# CosyVoice3 DEEPOPT — kernel-generation audit + optimization plan (NOT IMPLEMENTED)

> **STATUS: PLAN ONLY. No code has been written.** This document is the full audit of the
> CosyVoice3-0.5B port against the engine's current "#31" kernel generation (coalesced
> GEMV/GemmCoal, Kokoro Conv1DTile2/#33), plus a ready-to-execute implementation plan: cost
> models, eligibility tables, kernel sketches, call-site routing, parity gates and the
> validation checklist. A future session implements it cold from this file.
>
> Audit date: 2026-07-17. Baseline hardware: RTX 4060 Laptop (~236 GB/s DRAM, 32 MB L2,
> ~11–15 fp32 TFLOP/s — BENCHMARK.md hardware baseline). Sources read: SPEC.md, CosyVoiceLM.cs,
> CosyVoiceFlow.cs, HiFTVocoder.cs, CosyVoiceWeights.cs, CosyVoiceTTS.cs, CosyVoiceVoice.cs,
> CosyVoiceLMCS.compute, CosyVoiceFlowCS.compute, Gemma3CS.compute (#31 GVC_*/GemmCoal),
> Qwen3_5CS int8 variants, KokoroCS.compute (Conv1DTile2/LayerNormCoop), OPTIMIZATIONS.md #31,
> KOKORO_DEEPOPT.md, BENCHMARK.md, ProbeLogs/{cosyvoice_*.md, a6max_research.md}.

---

## 0. Executive summary

The premise "the port still runs on legacy kernels" is **half true**. The A6-max campaign
landed a lot before it was dropped (1307):

| Stage | Already modern (keep) | Still legacy (this plan) |
|---|---|---|
| LM decode (T=1) | **Phase-6b fused split-k GEMVs** (`DecQKV/DecOProjRes/DecGateUp/DecDownRes/DecHead`) — structurally ≥ #31 (§2.1) | — |
| LM prefill (T≤64 bursts) | — | `QProjBias/KProjBias/VProjBias/OProj/GateUp/Down/LmHeadPredict1Vec` — thread-per-(token,row), strided W walks. **#31 GemmCoal target.** |
| Flow DiT | batch-2 CFG, AdaLN-mod precompute, `BidirAttentionQT` (Q-tiled attn), `LinearTileBias2[Q8]` register-blocked GEMM | 13–14 dispatches/block, AdaLN/GateAdd/rope micro-dispatches, 8-dispatch pack chain per Euler step. **Fusion + GemmCoal target.** |
| HiFT vocoder | NSF/iSTFT kernel family (validated) | **every conv runs the naive `Conv1D`** (~0.7 % of fp32 peak measured). **Conv1DTile2-style [T,C] tile target — the single biggest lever in the pipeline.** |

**Headline predictions (fp16, offline, 4060):** HiFT 3131 → ~400–600 ms (**5–8×**), flow
4063 → ~2300–2900 ms (**1.4–1.8×**), LM prefill 1709 → ~550–700 ms (**2.5–3×**), LM decode
unchanged (already modern). E2E offline RTF **1.03 → ~0.5–0.6**; streaming RTF **1.31 → ~0.75–0.95**
(real-time becomes plausible, not guaranteed). int8 within ±10 % of fp16 on every stage
(dispatch/occupancy-bound, not DRAM-bound — measured precedent §2.4 of a6max_research.md).

---

## 1. Measured baseline (from ProbeLogs, current kernels)

| Metric | Value | Source |
|---|---|---|
| LM prefill 155 tok | **1709 ms** | cosyvoice_lm_parity.md |
| LM decode | **151.3 tok/s** (6.05× RT; 6.6 ms/tok) | cosyvoice_lm_parity.md |
| Flow solve, T=576 mel (10 Euler × batch-2) | **4063 ms wall**, 406 ms CPU issue | cosyvoice_flow_parity.md |
| HiFT offline, Tg=402 (8.04 s audio) | **3131 ms → RTF 0.389** | cosyvoice_hift_parity.md |
| E2E offline (10.88 s audio) | **RTF 1.03** (LM 2564 ms @106 tok/s; "vocoder 3426 ms" absorbs the flow GPU tail — known artifact, a6max §2.2) | cosyvoice_e2e_report.md |
| Streaming (5 chunks, 192 tok, 7.68 s) | **RTF 1.31, TTFA 2.48 s**; LM+sample-wait 6104 ms, chunk synth 3150 ms (47 % overlapped), finalize 804 ms | cosyvoice_stream_report.md |

---

## 2. Cost model

### 2.1 One LM decode token — ALREADY #31-CLASS, no work needed

`CosyVoiceLM.FastLM=true` path (A6-max Phase 6b): **146 dispatches/token**
(EmbeddingLookup + 24 × {DecQKV, DecRopeCache, FlashAttention, DecOProjRes, DecGateUp,
DecDownRes} + DecHead).

Design equivalence vs #31 `*1VecCoal` (Gemma3CS lines 1172–1302):

| #31 decode-GEMV property | Phase-6b kernels (CosyVoiceLMCS 1236–1669) |
|---|---|
| 256 threads = 8 rows × 32 lanes | DecQKV/DecOProjRes/DecGateUp/DecHead identical; DecDownRes = 4 rows × 64 lanes (deeper split for K=4864 — a superset, not a gap) |
| input staged once in groupshared | `dec_x[896]` / `dec_xl[4864]` |
| lane reads consecutive packed words (coalesced fp16/int8) | manual packed-uint reads, `st = lane; st += 32` — consecutive lanes → consecutive uints (Phase 6b reads 1 uint/lane/step vs #31's 2; both fully coalesced) |
| 32-lane fixed tree reduce | `dec_red` fixed trees |
| fp16 + int8 (+ int4 compile-only) | same `#pragma multi_compile` |
| — (#31 has none) | **extra fusions #31 lacks**: RMSNorm prologues, bias adds, residual epilogues, rope+K-cache+V-cache in one dispatch |

Bandwidth: weights/token = 363.9 MB int8 / 727.8 MB fp16 → floors at 236 GB/s = **1.6 / 3.1
ms/tok** (620 / 324 tok/s). Measured 151 tok/s fp16 = 47 % of floor — remaining gap is
attention + issue overhead, not GEMV design. **Verdict: leave decode alone; spend nothing here.**
(An A/B FastLM=false already exists: menu "A3 LM Parity LEGACY (FastLM off)".)

### 2.2 LM prefill (the TTFA floor) — legacy, #31 GemmCoal target

Per 64-token burst per layer, current `DispatchLayer` = 17 dispatches (Copy, RmsNorm, Q, K, V,
Rope×2, WriteCache×2, FlashAttn, O, Add, Copy, RmsNorm, GateUp, Down, Add); 155 tokens = 3
bursts × 24 layers + finals = **~411+ dispatches** and — the real problem — the projection
kernels are thread-per-(token,row) with K-strided W reads (~1/32 warp efficiency, the exact
pathology #31 fixed; Gemma3 GemmCoal precedent: prefill 417 → 1258 tok/s, **3.0×**).

Work per 155-token prefill: 155 × 358 M weights ≈ 55 GMAC; W traffic = 3 bursts × 716 MB fp16
= 2.1 GB → 9 ms DRAM floor. Measured 1709 ms ⇒ occupancy-bound ⇒ tiling fixes it.

Dispatches after (per layer per burst): QKV fused **1** (was 3), O 1, GateUp 1, Down 1 → 15/layer;
count barely matters — the win is warp efficiency.

### 2.3 One flow chunk (22 DiT layers × 10 Euler steps) — the RTF hot spot

Offline T=576 mel (2M = 1152 stacked CFG rows): **~248 GMAC per estimator forward** →
2.48 TMAC (≈5 TFLOP) per solve. Measured 4063 ms ⇒ ~1.2 fp32-TFLOP/s effective ≈ **8–10 % of
peak**. Per-forward MAC split: 22 blocks × (QKV 3.62 G + O 1.21 G + attn 1.36 G + FF1 2.42 G +
FF2 2.42 G) ≈ 243 G, convpos 4.7 G, proj/packs ~0.5 G.

Dispatches per Euler step, current (offline): concat chain 6 (Zero + Pack×4 + PackSpk) +
proj 1 + convpos 2 + Add 1 + **22 × 13** (AdaLN, Q, K, V, Rope×2, Attn, O, GateAdd, AdaLN, FF1,
FF2, GateAdd) + final AdaLN + proj_out + EulerCfg = **299/step → 2990/solve**.
Cached streaming step adds WriteFlowKV per block + apron packs/compaction ≈ **327/step**.

Planned (§4.2): PackEstIn 1 + proj 1 + convpos 2 + Add 1 + **22 × 8** (AdaLNStats, QKV-fused,
RopeQKPair, Attn, O+GateAdd, AdaLNStats, FF1, FF2+GateAdd) + final 2 (stats + proj_out-fused) +
EulerCfg = **184/step → 1840/solve** (cached ≈ 210/step). Plus each eliminated AdaLN/GateAdd
round-trip saves a [1152,1024] fp32 write+read (4.7 MB each way; ×4 sites × 220 block-steps ≈
8 GB of L2/DRAM traffic per solve).

### 2.4 One HiFT chunk — legacy naive convs, the biggest single lever

Tg=402 (8.04 s audio): **~135 GMAC** total —
f0 condnet 1.3 G, conv_pre 0.08 G, stage0 (L=3216, 256ch) ≈ 42.6 G, stage1 (L=16080, 128ch) ≈
50.3 G, stage2 (L=48241, 64ch) ≈ 40.7 G, conv_post 0.4 G, NSF/STFT/iSTFT ≈ 1 G.
Measured 3131 ms ⇒ **~86 GFLOP/s ≈ 0.7 % of fp32 peak.** The naive `Conv1D` gives each (t,oc)
one thread that serially walks in_ch × k with K·in-strided W reads. Not DRAM-bound (weights are
L2-resident; activation traffic ≈ 0.5 GB ≈ 2 ms) — pure warp-efficiency/ILP loss.
Kokoro precedent: Conv1DTile ≈ 19 % of peak, Conv1DTile2 better ⇒ even a conservative 8–12 %
of peak lands at **~300–500 ms**.

Dispatches per offline chunk: current ≈ **279** (each ResBlock = 20: Copy + 3j×{Copy, Snake,
Conv, Snake, Conv, Add} + Add/Copy; 12 resblocks total). Planned ≈ **~135** (fused ResBlock = 8;
Activate+Repeat+Conv collapsed into one repeat-fused tile conv per up stage).

---

## 3. Eligibility tables (K = reduction dim; GemmCoal requires K % 128 == 0)

### 3.1 LM (CosyVoiceLMCS.compute) — prefill GEMM sites

| Site | out × K | K%128 | Plan |
|---|---|---|---|
| q_proj (+bias) | 896 × 896 | ✓ (7×128) | **QKVProjBiasGemmCoal** (one dispatch, gid.x mode ranges 112/16/16 like DecQKV) |
| k_proj / v_proj (+bias) | 128 × 896 | ✓ | ↑ fused into same dispatch |
| o_proj | 896 × 896 | ✓ | **OProjGemmCoal** |
| gate/up | 4864 × 896 | ✓ | **GateUpGemmCoal** (two acc sets, silu/gelu writeback like legacy `GateUp`) |
| down | 896 × 4864 | ✓ (38×128) | **DownGemmCoal** |
| llm_decoder head | 6761 × 896 | ✓ (out 6761 needs only a `row < vocab` tail guard — 846 groups, DecHead precedent) | **LmHeadPredict1VecCoal** (T=1 GEMV — straight Gemma3 port, lm head always fp16) |
| embed lookups, RmsNorm, rope, KV write, FlashAttention | — | n/a | stay as-is |

Note: this export has **separate** W_Q/W_K/W_V and gate/up tensors (no concatenated q|k|v /
gate|up blobs), so the Gemma3 base-offset concat macros do NOT apply — the fused QKV kernel
routes by gid.x range over three bound buffers instead (DecQKV already does exactly this).
**Do not change the exporter.**

### 3.2 Flow DiT (CosyVoiceFlowCS.compute)

| Site | out × K | K%128 | Plan |
|---|---|---|---|
| input_embed.proj | 1024 × 320 | ✗ (320 = 2.5×128) | stays `LinearTileBias2[Q8]` (FastGemm) |
| attn.to_q/k/v (+bias) | 1024 × 1024 | ✓ | **DitQKVCoal[Q8]** — one dispatch, AdaLN-modulate fused into X staging (§4.2) |
| attn.to_out (+bias) | 1024 × 1024 | ✓ | **DitLinearCoal[Q8]** with gate-add epilogue (`resid += gate[c]·(acc+b)`) |
| ff.0.0 (FF1, GELU-tanh) | 2048 × 1024 | ✓ | DitLinearCoal, modulate staging + act 8 writeback |
| ff.2 (FF2) | 1024 × 2048 | ✓ | DitLinearCoal, gate-add epilogue |
| proj_out | 80 × 1024 | ✓ (out 80 → 10 row-groups, tail-guarded) | DitLinearCoal, modulate staging (norm_out order = SCALE,shift — reuse mod_scale_off/mod_shift_off uniforms) |
| adaLN mod Linear (6144×1024), time-MLP, spk affine (80×192) | T=1, precomputed once per instance | — | stay legacy (already hoisted by EnsureMods) |
| pre_lookahead conv1/conv2, convpos grouped convs | conv | n/a | stay legacy (once per chunk / 2 per step; convpos is grouped-64-in — optional later) |
| BidirAttention[KV]QT | — | n/a | stays (already Q-tiled Phase 3; out of #31 scope) |

### 3.3 HiFT (CosyVoiceFlowCS.compute) — Conv1DTileTC routing (stride==1 && K≤16 && (K−1)·dil ≤ 50)

| Site | ch out×in, k(,d) | Eligible | Notes |
|---|---|---|---|
| f0 condnet.0 | 512×80 k4 | ✓ | right-lookahead handled by generic in_len > seq_len |
| f0 condnet.2/4/6/8 | 512×512 k3 | ✓ | ELU already fused at writeback (act 7) |
| f0 classifier | Linear 1×512 | — | legacy (out_dim 1) |
| conv_pre | 512×80 k5 | ✓ | |
| ups.0 / .1 / .2 | 256×512 k16 / 128×256 k11 / 64×128 k7 | ✓ | **repeat-fused staging** (nearest-×stride + leaky on load) — kills Activate + RepeatTime dispatches AND the 15360·Tg-float intermediates |
| source_downs.0 / .1 | k30 s15 / k6 s3 | ✗ (stride) | legacy `Conv1D` (tiny: 0.7 GMAC total) |
| source_downs.2 | 64×18 k1 | ✓ | either; trivial |
| resblock convs1 | k∈{3,7,11}, d∈{1,3,5} | ✓ ((11−1)·5 = 50 = halo cap, exactly like Kokoro) | **snake fused into X staging** (formula must be `x + sin²(αx)/(α+1e-9)` — SnakeAct's exact expression, NOT Kokoro's exact-1/α) |
| resblock convs2 | k∈{3,7,11} d1 | ✓ | += writeback (conv_out_mode 1) into rbAcc |
| conv_post | 18×64 k7 | ✓ | oc=18 < tile width: guard, still wins (T≈48k) |
| STFT16/ISTFT16/NSF/MagPhase | — | — | stay (validated, cheap) |

---

## 4. Ranked levers — predictions and implementation sketches

Static switches (repo precedent: `FastLM`, `FastAttention`, `FastGemm`, `ForceLegacyGemv`),
all `public static bool`, default **true**, legacy path kept runnable:

| Tier flag | Class | Covers |
|---|---|---|
| `HiFTVocoder.FastConv` | HiFTVocoder.cs | lever 1 |
| `CosyVoiceFlow.FastDit31` | CosyVoiceFlow.cs | lever 2 |
| `CosyVoiceLM.FastPrefill` | CosyVoiceLM.cs | lever 3 |

### Lever 1 — HiFT `Conv1DTileTC` (predicted vocoder 3131 → ~400–600 ms, 5–8×; e2e −~2.6 s)

New kernel in **CosyVoiceFlowCS.compute**, adapted from KokoroCS `Conv1DTile2` to the flow
file's **[T,C] row-major layout** (Kokoro is channel-major — geometry must be transposed, it is
NOT a copy-paste):

- threads (32,8); tile = **64 output channels (tx, tx+32) × 64 timesteps (ty + 8r, r<8)**;
  16 named accumulators (`float2 c0..c7`; no dynamically-indexed arrays — they spill).
- groupshared: `ctc_x[114][4]` (XT = 64 + halo 50, CI slice = 4) + `ctc_w[64][4*16]` ≈ 18 KB.
- X staged per CI-slice with **fused prologue** via new uniform `conv_in_mode`:
  0 raw · 1 snake(α per in-channel, SnakeAct-exact) · 2 leaky · 3 leaky+nearest-repeat
  (new uniform `conv_repeat`; staging reads `X[((gt)/conv_repeat)*in_dim + ci]`, gt = global
  padded index, zero outside `[0, in_len)`); each element re-activated per tap (≤K reads) —
  same fp32 expression on the same value ⇒ identical sums (Kokoro-documented precedent).
- W staged per slice, coalesced (`readH`, torch [out,in,k] rows contiguous); bias-first accs.
- writeback `conv_out_mode`: 0 `Y=`, 1 `Y+=` (ResBlock x += conv2), through `apply_act`.
- **accumulation order is ci-slice-outer/k-inner vs Conv1D's k-outer/ic-inner ⇒ NOT bit-exact**
  — tolerance-gated (§6). Copy the "NOT [unroll]: fxc mis-schedules barriers in unrolled loops"
  comment onto any loop that contains a barrier.

HiFTVocoder.cs routing (all inside existing helpers — streaming/window/seam logic untouched):
- `Conv(...)`: route to tile when `FastConv && stride==1 && kernel<=16 && (kernel-1)*dilation<=50`.
- `ResBlock(...)` fast form: `Copy(rbAcc,x)` then per j: tile-conv(X=rbAcc, in_mode=snake α1j) →
  rbT2; tile-conv(X=rbT2, in_mode=snake α2j, out_mode +=) → rbAcc. 20 → 8 dispatches per
  resblock. **No SRV/UAV aliasing**: conv1 reads rbAcc/writes rbT2, conv2 reads rbT2/writes rbAcc.
- Up-chain fast form: one tile-conv(X=vA pre-repeat, in_mode=leaky+repeat(ups[i]), in_len=outLen)
  → vB, then swap — replaces Activate + RepeatTime + Conv and skips the 8–120·Tg×ch intermediate.
- conv_post chain: fuse the preceding leaky via in_mode 2.

Bandwidth math: 135 GMAC ⇒ at 10 % of 13 TFLOP/s ≈ 210 ms conv time + ~150 ms of
legacy/NSF/iSTFT/readback tail ⇒ 400–600 ms range. Even the pessimistic 5 %-of-peak case
(≈650 ms) is a 4.8× win.

### Lever 2 — Flow DiT fusion set in #31 GemmCoal form (predicted flow 4063 → ~2300–2900 ms)

Honesty note: unlike the LM/HiFT levers, the flow's GEMM baseline is **already tiled**
(`LinearTileBias2`, ~19 % of peak by Kokoro's measurement of the same design). A GemmCoal
GEMM-for-GEMM swap alone is likely a wash (GemmCoal even re-reads W more often: every 8-token
tile vs LTB2's 32). The real value is the **fusions**, which need new kernels anyway — the plan
ships them in #31 GemmCoal form (campaign directive, proven shape), and the A/B probe decides;
if GemmCoal loses to LTB2 on raw GEMM, port the same in/out modes onto LTB2 tiles instead
(fallback explicitly allowed, same C# routing).

New kernels in **CosyVoiceFlowCS.compute** (fp16 + Q8 twins, macro-generated bodies —
this file uses per-tensor Q8 kernels, NOT multi_compile; keep that convention):

1. `AdaLNStats` — copy `AdaLNModulate`'s exact 256-thread mean/var tree but write only
   `est_stats[row*2] = mean, [+1] = rstd` (new RW buffer). Same reduction structure ⇒ stats are
   bit-identical to today's AdaLN; the modulate expression moves into GEMM staging unchanged ⇒
   GEMM **inputs** are bit-identical, only GEMM sum order differs (tolerance-gated).
2. `DitQKVCoal` / `DitQKVCoalQ8` — GemmCoal (8 rows × 8 tokens, 128-col staged chunks) with:
   gid.x mode ranges 128/128/128 (1024 rows ÷ 8 per mode) over three weight buffers
   (new decls `W_k, W_v, W_k_bias, W_v_bias, W_k_scales, W_v_scales`; existing W/W_bias/W_scales
   = Q), outputs to Y (=qBuf) + new `Y2, Y3` RW decls; X staging applies
   `(x−mean)·rstd·(1+mod_vec[mod_scale_off+c]) + mod_vec[mod_shift_off+c]` per token from
   est_stats. Route only when q/k/v quant status matches (all fp16 or all q8 — `Sc()` is
   per-tensor); else 3 separate DitLinearCoal dispatches.
3. `DitLinearCoal` / `Q8` — generic GemmCoal with uniforms `dit_in_mode` (0 raw / 1 modulate)
   and `dit_out_mode` (0 `Y=apply_act(acc+b)` / 1 `Y += mod_vec[mod_gate_off+row]·(acc+b)`).
   Covers O (raw, gate-add), FF1 (modulate, act 8), FF2 (raw, gate-add), proj_out (modulate,
   plain). Out-mode-1 float ops match legacy GateAdd exactly given the same acc.
4. `RopeQKPair` — one dispatch ropes qBuf AND kBuf (new RW `inout_buf2`); RopeQK math verbatim
   (only head-0 dims 0..63 rotate, interleaved pairs). Bit-exact; saves 220 dispatches/solve.
5. `PackEstIn` — one dispatch builds the whole `[2·Le, 320]` estimator input. New dedicated
   buffers `est_x, est_tail, est_cond, est_mu, est_spk` (reads) + `est_out` (RW); uniforms
   `est_rows(Le), est_apron, est_xrow(F), est_cond_row(F−apron), est_tail_row`. **Must reproduce
   the legacy Zero+Pack chain exactly** (enumerated from EstimatorFull/EstimatorCachedStep):
   lane0 (ch 0–79) both halves: row < apron ? tail[tail_row+row] : x[F+row−apron];
   lane1/2 (cond/mu) cond-half only from row (F−apron)+row, uncond half **0**;
   lane3 (spk broadcast) cond-half only, uncond **0**. Offline = apron 0, F 0, Le = M
   (bind xBuf as est_tail — never read). Bit-exact data movement.

CosyVoiceFlow.cs routing (`FastDit31`):
- `RunBlocks`: stats → QKV-fused → RopeQKPair → [WriteFlowKV] → attention (unchanged) →
  O(out_mode 1, gate off+2·DIM) → stats → FF1(in_mode 1, offs off+4D/off+3D, act 8) →
  FF2(out_mode 1, gate off+5·DIM). 13/14 → 8/9 dispatches per block.
- `EstimatorFull` / `EstimatorCachedStep`: PackEstIn replaces the Zero+Pack chain; final =
  AdaLNStats + proj_out DitLinearCoal (in_mode 1 with modFSteps offsets — note norm_out order
  is (SCALE, shift), the existing uniforms already encode that at the call site).
  Compaction CopySlices, convpos, proj(320-K), EulerCfg, SaveXTail, WriteFlowKV: unchanged.
- Resolve `wPo/bPo/sPo` (proj_out) handles in `ResolveWeights` alongside the block arrays.
- New scratch: `statsBuf` = `2·curM·2` floats in EnsureScratch.
- **DebugTap contract unchanged** (h_lookahead / dxdt_cond_s0 / dxdt_uncond_s0 / mel_full all
  tap the same buffers at the same points), so the existing A2 probe grades the new path as-is.
- Streaming semantics: only kernel routing changes; hop schedule, K/V cache, x-tails, chunk
  masks, EulerCfg offsets byte-identical.

Prediction math: −40 % dispatches (2990→1840; at the streaming-measured ~50 µs/dispatch issue
+ queue gaps ≈ −0.5 s), −~8 GB/solve of eB/gate round-trip traffic (≈ −0.4 s at effective
L2/DRAM mix), + whatever the GemmCoal-vs-LTB2 A/B yields on the 243 GMAC of GEMMs (0.9–1.4×).
Range: 1.4–1.8× ⇒ 2300–2900 ms.

### Lever 3 — LM prefill GemmCoal (predicted 1709 → ~550–700 ms; TTFA −~1 s)

New section in **CosyVoiceLMCS.compute** under the existing `multi_compile _ INT8_WEIGHTS
INT4_WEIGHTS` (readW/wScale macros make one body serve all three; INT4 compiles but never
ships — same note as the Phase-6b kernels):

- Port Gemma3CS `GVC_*`/`GMM_*` macros with `GVC_XMAX 4864` (LM's largest staged K) and local
  names (`gvc_x/gvc_red/gmm_x` don't collide with `dec_*`).
- `QKVProjBiasGemmCoal`: Dispatch(144, ⌈seq/8⌉): gid.x 0–111 Q (row = gid.x·8+slot), 112–127 K,
  128–143 V; writeback `+ readH(proj_bias|proj_bias_k|proj_bias_v, row)`, output strides 896
  (Q_out) / 128 (K_out/V_out). All three bias buffers already declared (Phase-6 additions).
- `OProjGemmCoal` (112, ⌈seq/8⌉), `GateUpGemmCoal` (608, ⌈seq/8⌉ — two acc arrays ag/au[8],
  silu/gelu per `activation_type`, wScale per buffer), `DownGemmCoal` (112, ⌈seq/8⌉, K=4864 →
  38 staged chunks), `LmHeadPredict1VecCoal` (846,1 — vocab tail guard, fp16-only readH).
- Keep the `// NOT [unroll]: fxc mis-schedules barriers in unrolled loops` comment verbatim on
  every GMM token-writeback loop (Gemma3 lines 1349/1373/1400/1432 precedent).

CosyVoiceLM.cs routing: in `DispatchLayer`, when `FastPrefill && seqLen > 1` swap the six
matmul dispatch sites (Q/K/V → one fused; O; GateUp; Down); `seqLen == 1` keeps legacy kernels
(preserves the FastLM=false bisect arm untouched). `DispatchFinalLast` routes LmHead to the
coal twin under the same flag. Copy/RmsNorm/rope/cache/attention/Add stay (they are not the
bottleneck; keeping them makes the A/B attribution clean).

Effect on outputs: prefill logits shift by GEMM-reduction reorder (~ulp) ⇒ RAS-sampled token
sequences may diverge run-to-run vs old builds — **same contract FastLM Phase 6b already
established** (see its header comment); dump gates (corr/argmax) unaffected.

### Explicitly left legacy (with reasons)

- LM decode GEMVs — already ≥ #31 (§2.1).
- Flow attention — already Q-tiled (Phase 3); #31 covers matmuls, not attention.
- input_embed.proj (K=320 ✗ 128), t-embed/adaLN-mod/spk (T=1, precomputed), pre-lookahead +
  convpos convs (≤2 dispatches/step, grouped-conv tile is a separate design — optional later).
- source_downs.0/.1 (strided), NSF/STFT/iSTFT family (validated, ~1 % of vocoder MACs).
- GPU RAS sampler, KV cache formats, hop schedule, seam fades, residency loader: untouched.

---

## 5. Weight-packing notes (CosyVoiceWeights.cs — DO NOT change the exporter)

- Manifest-driven: `name\tfile\tdtype\tnumel\tshape`; fp16 packed 2-per-uint (`readH`), q8
  packed 4-per-uint (`readQ8`) + fp16 per-OUTPUT-ROW `.scales` sibling tensor, i32 raw.
- int8 selection is **per tensor** (`Has(name+".scales")`): LM matmuls + DiT matmuls in the
  int8 export; norms/embeddings/heads/conv weights/biases always fp16. New kernels must keep
  bias/scale reads as readH and apply row scale once per dot (existing wScale semantics).
- **No concatenated q|k|v / gate|up tensors exist** — fused kernels bind separate buffers and
  route by gid range (§3.1/§3.2). The Gemma3 concat-base-offset macros are NOT applicable.
- All new kernels consume the existing layouts exactly; zero export/import changes.

---

## 6. Parity gates, probe plan, validation checklist (for the implementing session)

### New probe file: `validation/Editor/CosyVoiceFastKernelsProbe.cs`

Follow `GemmaCpmGemvParityProbe` gate structure (isolated tight gate + full-path relative gate
+ argmax + same-run A/B timing). Menu items + batch-runnable statics, fp16 and int8 twins
(weightsDir swap like the existing probes):

1. **LM prefill gate** (`DeepUnity/CosyVoice/#31 LM Prefill Parity + A-B [INT8]`):
   load `llm/`, dump tokens; arm A = FastPrefill off, arm B = on (FastLM constant ON both):
   step-0 logits **rel maxAbs ≤ 2e-3, corr ≥ 0.999999, argmax MATCH**; then 8 greedy
   (argmax-of-logits) continuation steps per arm — token-for-token match expected, gate ≥ 7/8
   + per-step logits corr ≥ 0.99999 (near-tie escape). `[perf]` prefill ms per arm, same run.
2. **Flow gate** (`#31 DiT Fast Parity + A-B [INT8]`): injected dump speech tokens, offline:
   arm A FastDit31 off / arm B on → mel **maxAbs ≤ 5e-3 (fp16), corr ≥ 0.9999**; PLUS arm B
   vs dump refs must clear the existing A2 gates (h ≥ 0.999, dxdt ≥ 0.99, mel ≥ 0.99).
   `[perf]` solve wall + IssueMs per arm + printed dispatch counts (analytic: 299 vs 184 per
   step × NT, formula in §2.3).
3. **HiFT gate** (`#31 HiFT Conv Parity + A-B`): injected source; DebugTap per-stage capture on
   both arms → per-stage **corr ≥ 0.99999**, final wav **corr ≥ 0.999, maxAbs ≤ 2e-2**; plus
   arm B vs dump refs clears the existing A1 stage gates. `[perf]` vocode ms per arm +
   dispatch counts (279 vs ~135).
4. **Chain A/B** (`#31 Flow+HiFT E2E A-B`): dump tokens → mel → wav, all-fast vs all-legacy
   (deterministic — no sampler), wav corr ≥ 0.999 + total ms per arm.

### Validation checklist (main session, in order)

```
1. Open Unity — zero shader compile errors in Console (both .compute files), zero C# errors.
2. -executeMethod DeepUnity.CosyVoiceModeling.CosyVoiceFastKernelsProbe.RunLmPrefill      → PASS
3.               ...CosyVoiceFastKernelsProbe.RunLmPrefillInt8                            → PASS
4.               ...CosyVoiceFastKernelsProbe.RunFlow / RunFlowInt8                       → PASS
5.               ...CosyVoiceFastKernelsProbe.RunHift                                     → PASS
6.               ...CosyVoiceFastKernelsProbe.RunChain                                    → PASS
7. Existing dump gates on the new defaults:
   CosyVoiceLmProbe.Run + RunInt8 · CosyVoiceFlowProbe.Run + RunInt8 · CosyVoiceHiftProbe.Run
   · CosyVoiceE2eProbe.Run                                                                → PASS
8. Bisect arms still green: "A3 LM Parity LEGACY (FastLM off)" + new LEGACY menu twins
   (flip FastPrefill/FastDit31/FastConv off) — proves the fallback path survived.
9. RTF reads (same run, before/after lines into ProbeLogs):
   CosyVoiceStreamProbe.Run (A5) + RunInt8 (A6) → RTF/TTFA/breakdown vs §1 baseline;
   CosyVoiceE2eProbe.Run offline RTF line.
10. Listen check: ProbeLogs/cosyvoice_{flow,hift,e2e,stream}_unity.wav — no seams/buzz.
```

Failure policy: any gate FAIL → flip that tier's static to false (ship-safe), bisect with the
per-stage taps; flow GemmCoal-slower-than-LTB2 (not a parity fail) → keep fusions, swap tiling
per §4.2 fallback.

---

## 7. Predicted end state + risks

| Stage (fp16, offline 8–11 s utterance) | Now | Predicted | Factor |
|---|---:|---:|---|
| LM prefill (155 tok) | 1709 ms | ~550–700 ms | 2.5–3× |
| LM decode | 151 tok/s | 151 tok/s | 1× (already modern) |
| Flow solve (T=576) | 4063 ms | ~2300–2900 ms | 1.4–1.8× |
| HiFT (8 s audio) | 3131 ms | ~400–600 ms | 5–8× |
| **E2E offline RTF** | **1.03** | **~0.50–0.60** | ~1.8× |
| **Streaming RTF** | **1.31** | **~0.75–0.95** | ~1.5× (real-time plausible) |
| **TTFA** | **2.48 s** | **~1.4–1.7 s** | prefill + first flow chunk |

int8: same ±10 % (all three levers are occupancy-bound, not byte-bound; int8's win stays
VRAM/load, matching every prior A/B in this port).

Risks, ranked:
1. **Conv1DTileTC is new geometry** ([T,C] transpose of Kokoro's [C,T]) — coalescing/bank
   behavior must be A/B-proven, not assumed; snake-per-tap recompute at k=11 could eat margin
   (fallback: fuse snake only for k ≤ 7, keep standalone SnakeAct for k=11).
2. **Flow GemmCoal may not beat LTB2** on raw GEMM — mitigated by the explicit LTB2-fusion
   fallback; the fusion/dispatch win (~0.9 s) survives either tiling.
3. fp32 reorder propagation through 10 Euler steps — mel corr gate could land near threshold;
   if so, grade mel on the emitted band and tighten per-kernel gates instead.
4. fxc barrier gotchas in unrolled loops — mitigated by copying the reference comments and
   structures verbatim; no wave intrinsics anywhere (SM5.0).
5. Streaming byte-identical behavior rests on PackEstIn's zero-fill mapping (§4.2.5) — the
   enumerated mapping must be implemented exactly; the stream probe's seam/RTF lines are the
   backstop.
6. Two other agents own PocketTTS/Qwen3.5/benchmarking files — this plan touches ONLY
   CosyVoiceLMCS.compute, CosyVoiceFlowCS.compute and TTS/CosyVoice/** (both shaders already
   registered in DeepUnityMeta; no shared-file edits needed).
