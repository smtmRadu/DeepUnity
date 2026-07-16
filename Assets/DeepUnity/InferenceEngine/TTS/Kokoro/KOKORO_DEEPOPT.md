# Kokoro #33 deep-opt — throughput autopsy + FastKernels2

Second optimization pass over the Kokoro-82M GPU port (branch `kokoro-deepopt`), in the
pocket-tts #30 / Qwen #31 style: per-stage autopsy first, then kernels. Everything new is
routed by `KokoroModel.FastKernels2` (static, default ON, only active on top of
`FastKernels`) so probes bisect **v2 / v1 (#26) / legacy** without replumbing.
Written by the deep-opt workstream — NOT yet compiled/measured in Unity; the validation
checklist at the bottom is the main session's runbook.

## 1. Autopsy — where a synthesized second goes (post-#26)

Reference measurements (RTX 4060 laptop, editor, #26 same-session A/B on the t0/t1/t2
dumps = 3.7/7.2/8.4 s audio):

| stage | pre-#26 | post-#26 (current) | share of GPU time |
|---|---|---|---|
| PLBERT (`BertMs`) | 43–73 ms | 33–44 ms | ~17 % |
| decoder (`DecoderMs`) | 30–54 ms | 18–21 ms | ~8 % |
| generator (`GeneratorMs`) | 370/695/813 ms | 105/170/199 ms | **~75 %** |
| RTF (editor, incl. Mono CPU-LSTMs) | 0.375/0.245/0.235 | 0.301/0.166/0.152 | — |

CPU side: the 6 biLSTMs are ~0.8–1 s/utterance on editor Mono (scalar fallback) but
~30 ms in IL2CPP — per prior directive, the editor LSTM path is NOT to be re-optimized.
The tenc biLSTM however ran **serially** (GPU idle) inside `PredictorMs`; that is a
scheduling bug, not a kernel one — fixed below.

MAC model per audio-second (F = 40 duration frames, S = 24 000 samples, 120F+1 = 4 801
STFT frames):

- **Generator ≈ 24.5 GMAC/s — 97 % of it in the 8 Snake resblocks**
  (rb0–2 + noise_res0 @ 256 ch × 20F ≈ 8.8 G; rb3–5 + noise_res1 @ 128 ch × (120F+1)
  ≈ 15.1 G). ups0/1 ≈ 0.5 G, noise_conv0/conv_post ≈ 0.15 G, STFT/iSTFT noise-level.
- **Decoder ≈ 1.4 GMAC/s** (encode 0.21, decode0–2 0.92, decode3 0.24) — same kernels.
- **F0/N stacks ≈ 0.3 GMAC/s.**
- **PLBERT ≈ 66 MMAC/token** but T is tiny (50–200/chunk): at t0 it ran ~80 GMAC/s
  effective — 130+ small dispatches/chunk, thread-per-position LayerNorms that light up
  T (=50!) threads total, 24-group matmul waves. **Dispatch/latency-bound, not MAC-bound.**

Diagnosis of the top stage: generator throughput ≈ 1.03 TMAC/s = **~19 % of the 4060's
fp32 peak**, while its global traffic (~1.2 GB/audio-s) is only ~5 ms of a 256 GB/s bus →
**not DRAM-bound; the Conv1DTile inner loop is LDS-port-bound** (6 groupshared reads per
8 FMAs) and every 128-ch conv re-stages + re-activates the same X slice in all 8 oc-tile
rows (the fused AdaIN+**Snake sin()** prologue runs 8×). On the GTX 1650 (2.9 TFLOPS,
128 GB/s) the same shares hold with ~3–4× the wall time — no 1650 Kokoro numbers were
ever recorded; this pass must produce them.

## 2. Changes (all inside TTS/Kokoro/** + KokoroCS.compute, all behind FastKernels2)

| # | change | attacks | expected gain |
|---|---|---|---|
| C1 | **`Conv1DTile2`** — 128 t × **32 oc** tile, each thread owns 4 t × 4 oc = 16 accumulators (four float4 rows). Per (ci,k): 4 broadcast W reads + 4 conflict-free X reads → 16 FMAs (v1: 6 reads → 8). Halved oc-row count also halves redundant X staging + prologue Snake recompute. Same per-output accumulation order (bias first, ci asc in 8-slices, k asc) → bit-comparable to v1/oracle modulo fma noise. | LDS-bound Snake/Adain convs = ~90 % of GPU MACs | generator convs 1.5–1.9×; decoder ~1.3× |
| C2 | **Fused writebacks** on `Conv1DTile2` (`conv_out_mode`): 1 = `Y += conv` replaces the SnakeResBlock residual `AddBuf` (24/audio-s); 2 = `Y = (conv + buf_b)·scale` replaces the AdainBlock `(res+short)·rsqrt2` AddScale (shortcut computed before conv2). Same float ops, same per-element order → bit-exact vs the dispatches they replace. | dispatch count + full-buffer R/W passes | ~35 dispatches + ~100 MB traffic per audio-s |
| C3 | **`LayerNormCoop`** — one 256-thread group per position, row staged in groupshared (1 global read/elem instead of 3), tree-reduced mean/var; `ln_add=1` fuses the two per-layer bert residual adds (`LayerNormAdd`). Tree reduce reorders sums (like #31 GEMVs) → gated at maxabs < 1e-3, not bit-exact. Routed for ALL LNs (bert emb/attn/ffn + tenc). | PLBERT latency: thread-per-position LNs + 24 Add dispatches/chunk | bert 1.3–1.6× (33–44 → ~22–30 ms) |
| C4 | **Pipeline reorder (C#, identical math)** — tenc convs issued BEFORE PLBERT with an early `AsyncGPUReadback`; tenc biLSTM Task runs concurrently with the predictor Task (was serial-with-GPU-idle). F0 readback requested before the N stack (earlier NSF start). | serialized CPU stages | hides `TencCpuMs` entirely: ~0.3–1 s/chunk editor, ~10–15 ms IL2CPP; TTFA − same |
| C5 | **Activation prologues** — `ConvTranspose1DFast` gains `conv_in_mode` 2 (AdaIN+lrelu: up-path pool consumes the normalized activation directly, kills CopySlice+IN+Activate per up block) and 3 (plain lrelu: trunk pre-ups0/ups1 activations + `conv_post`'s 0.01-slope, and tenc's inter-conv 0.2 lrelu) — each fuses a full elementwise pass + dispatch. Recomputed per tap on the same fp32 value → identical sums. | elementwise passes + dispatches | ~10 dispatches + ~180 MB traffic per audio-s |

Net expectation (GPU stages, t2-shaped chunk on the 4060): **~264 ms → ~160–190 ms**;
proportionally larger relief on the 1650 where every ms is 3–4×. End-to-end editor RTF
will additionally drop by the hidden TencCpuMs.

Parked levers (diagnosed, deliberately not taken): one-pass InstanceNormStats sum/sumsq
(saves ~1 ms/audio-s on the 1650 but changes variance rounding feeding EVERYTHING —
bad risk/reward); batching the 78 per-forward style FCs into one dispatch (needs a
KokoroWeights arena change); iSTFT twiddle tables (measured trivial: ~6 M transcendentals
per audio-s ≈ tens of µs); resblock-mean copies; noise_conv0 (0.2 % of MACs).

## 3. What changed on disk

- `Assets/Resources/ComputeShaders/KokoroCS.compute` — new uniforms `conv_out_mode`,
  `ln_add`; new kernels `Conv1DTile2`, `LayerNormCoop`; `ConvTranspose1DFast` prologue
  modes (v1 kernels untouched byte-for-byte).
- `TTS/Kokoro/KokoroModel.cs` — `FastKernels2` static; `Conv`/`ConvT` optional
  `inMode/inSlope` (ConvT now ALWAYS sets `conv_in_mode` — stale-uniform guard);
  `ConvFused2`, `LayerNormAdd`, cooperative `LayerNorm` routing; v2 branches in
  `AdainBlockY`/`SnakeResBlockY`; reordered `ForwardYielding` (early tenc + concurrent
  tasks + early F0 readback + trunk/conv_post/tenc fusions); `Pending`
  `BeginReadback`/`ResolvePending` helpers.
- `TTS/Kokoro/validation/KokoroKernelProbe.cs` — serialized `fastKernels2` flag, routing
  line in the report, new part-A tests `2c LayerNormCoop +residual` and
  `9c ConvT +lrelu prologue` (v2-only).
- `TTS/Kokoro/validation/Editor/KokoroKernelBatchRunner.cs` — menu
  `Run Kokoro Kernel Probe V1 (FastKernels2 off)` for the 3-way bisect.

No edits outside the Kokoro folder + KokoroCS.compute. DeepUnityMeta untouched
(KokoroModel Resources.Load's the shader directly). Weights format untouched — fp16 AND
int8 folders both work unmodified (conv weights are always fp16; the q8 `LinearTileBias2Q8`
matmul routing is unchanged).

## 4. Validation checklist (main session runbook)

Gate 0 — compile: open the worktree project; expect 0 errors with only the files in §3
changed. `Conv1DTile2` uses ~16.6 KB groupshared — if dxc complains, that's a real bug,
report back (budget is 32 KB).

Gate 1 — kernel + stage probe, three-way bisect (each writes
`ProbeLogs/kokoro_kernel_report.md` + `.done`; also runnable headless via
`Unity.exe -batchmode -executeMethod DeepUnity.KokoroModeling.KokoroKernelBatchRunner.Run`
— batch runs v2 defaults):

1. menu `DeepUnity/TTS/Run Kokoro Kernel Probe` (v2) — expect **31/31 kernels PASS**
   (29 legacy-graded + 2c + 9c) at maxabs < 1e-3; stages: bert_dur/d_en/d/duration ≥ .999,
   en/F0/N ≥ .995, t_en/asr ≥ .999, dec_x ≥ .99, pred_dur exact (or ±1 ≤ 2 tokens),
   **t0 wav corr ≥ 0.99** with injected noise. Composite 13a/b/c grade the fused
   ConvFused2/outMode/pool-prologue paths directly.
2. menu `... Kernel Probe V1 (FastKernels2 off)` — must still pass (guards #26 regression).
3. menu `... Kernel Probe LEGACY (FastKernels off)` — must still pass.

Record the `[perf]` line of each t0/t1/t2 run for v2 vs v1: bert / predictor
(pred/tenc cpu) / decoder / generator / end-to-end / RTF. Expected on the 4060:
generator 105/170/199 → **~60–120 ms**, bert 33–44 → **~22–30 ms**, decoder 18–21 →
**~14–17 ms**, TencCpuMs → **~0** (hidden). FAIL the pass (and bisect with the menus) if
any stage regresses vs v1.

Gate 2 — listen QA: `ProbeLogs/kokoro_gpu_t0.wav` (fp16, from gate 1) — intelligible
af_heart, no clicks/buzz vs the reference dump wav.

Gate 3 — int8 + frame pacing: `QwenKokoroPerfRunner` (bridge Run/Finish/Restore pattern)
→ `ProbeLogs/qwen_kokoro_perf.md`. int8 weights exercise the q8 matmul twins alongside
the new conv/LN kernels. Expect B prefetch smooth (0 > 33 ms), D2 speak-alone ≤ #26's
avg 1.7 ms / 0 > 33 ms, D3 combined spikes only from Qwen decode. Listen to
`ProbeLogs/kokoro_int8_velmire_elder.wav` (int8 has no dump-parity oracle — INT8_NOTES).

Gate 4 — cross-GPU numbers (the point of this pass; record in BENCHMARK.md alongside the
#30/#31 A/Bs):
- **RTX 4060**: gate-1 v2-vs-v1 `[perf]` lines (above).
- **GTX 1650 laptop 4 GB (reference low-end tier)**: same three probe runs — these are
  the FIRST Kokoro numbers on that box. Record per-stage ms + RTF for v2/v1 at fp16 and
  the QwenKokoroPerfProbe int8 pass. Also note TTFA proxy = t0 end-to-end (first-chunk
  synth time ≈ time-to-first-audio for a non-AR model).

Gate 5 — E2E regression: NPC E2E 3D probe (S2 Kokoro voice speaks, S2b frame stats) —
ALL PASS, 0 errors, voice audible during Qwen generation.

Rollback story: any failure → flip `KokoroModel.FastKernels2 = false` (one static; v1
behavior is byte-for-byte the #26 dispatch list) and report which gate + which kernel row.

---

# Round 2 (#33-R2) — predictor biLSTM stack ON GPU (`FastKernels3`)

Round-1 landed (v0.15.7, 31/31 PASS, generator 199→95 ms, bert 44→18, tenc hidden) and
left the CPU predictor LSTM chain as the editor wall: **pred 844–894 ms/chunk on Mono**
(10–15 ms IL2CPP). Round 2 moves the whole `pred` stage — DurationEncoder (3× biLSTM +
AdaLayerNorm), the duration-head biLSTM + `dur_proj`, and the shared biLSTM — onto the
GPU behind `KokoroModel.FastKernels3` (default ON, layered on FastKernels2; the KokoroCPU
path stays intact as the A/B fallback and the parity oracle).

## R2.1 Design

**Two-kernel LSTM, mirroring the oracle's own two-phase split** (KokoroCPU.BiLstm =
parallel input projection + sequential recurrence):

- **`LstmInProjTile`** — `pre[t,r] = (bih[r]+bhh[r]) + wih[r,:]·x'(t,:)`, r in [0,1024).
  A LinearTileBias2 clone (32 tokens × 64 rows, both operands staged, each weight read
  reused 32× — naive per-(t,r) threading would have re-streamed the 1.25 MB wih T times)
  with the LSTM input rule fused into the X staging: `lstm_cat_dim` appends the 128-d
  style vector from `buf_b` (CatStyle never materialized), `lstm_gather` indexes rows
  through `gather_idx` (the `en = d@aln` alignment expansion fused into the shared-LSTM
  read — no [F,640] en buffer exists at all). Dual bias via new `W_bias2` (bhh); bias-sum
  first then k-ascending, the oracle's order.
- **`LstmBiRecur`** — the persistent recurrent half: ONE dispatch, **2 groups** (gid.x =
  fwd/bwd — bwd walks t = T-1..0), timesteps looped in-kernel. 256 threads/group; thread
  i owns hidden unit i = gate rows {i, 256+i, 512+i, 768+i} (torch i,f,g,o), so all four
  gates accumulate in registers and the cell state c never leaves a register; only
  h[256] is groupshared, swapped with 2 barriers/step. Per step each thread streams its
  4 whh rows (j-ascending, packed-word-unrolled; the 512 KB whh stays L2-resident across
  steps on both the 4060 and the 1650) against broadcast h[j] reads. Gate math =
  `c = σ(f)·c + σ(i)·tanh(g); h = σ(o)·tanh(c)` — RunLstmDir verbatim (HLSL exp/tanh vs
  the oracle's double Math.Exp/Tanh; recurrence is contractive, gated at 1e-3).
- **AdaLayerNorm** = `LayerNormCoop` with new `ln_style=1`: affine from the style fc's
  fp32 [2C] ((1+γ)·LN_noaffine + β, eps 1e-5) instead of fp16 ln_gamma/ln_beta.
- **`TransposeRC`** — shared-LSTM rows [F,512] → channel-major xf [512,F] (replaces the
  CPU transpose + upload).

**What stays CPU (deliberately):** the duration head's sigmoid-sum — the GPU computes
`dur_proj` logits and reads back only [T,50]; the **double-precision** sigmoid-sum,
`/speed`, round-half-even and clamp run as verbatim oracle code (rounding feeds the
alignment; keeping it bit-identical to KokoroCPU.DurationHead means pred_dur can only
differ through ~1e-6 logit noise at exact .5 boundaries — covered by the existing D
gate's exact-or-±1≤2 provision). Plus the integer frame2tok build (µs). The **tenc
biLSTM** and **NSF source** stay on worker Tasks (see R2.4).

**Pipeline shape under v3:** the d_en readback disappears (the durenc chain reads dEnBuf
directly); the bert scratch (qB/kB/atB, free after benc) hosts the [T,512] chain; new
grow-only `lstmPreF/lstmPreR` [max(T,F)×1024] hold the input projections. `BertMs` is
now dispatch-issue wall only (no sync point) — `PredictorMs` still covers everything up
to asr; `PredCpuMs` = GPU durenc/head wall + [T,50] readback + CPU head.

**Weights:** zero loader changes — KokoroWeights already uploads every manifest tensor,
including all LSTM sets (fp16 packed, torch layouts); its header comment was the only
edit. int8 export unaffected (LSTM tensors are f16 in both exports per INT8_NOTES).

## R2.2 Expected impact (both targets, honestly)

- **Editor (Mono):** the ~850 ms CPU-LSTM wall per chunk is replaced by GPU work
  estimated at **~5–20 ms** (in-proj ≈ 1.3 GMAC/chunk tiled + 10 persistent recur
  dispatches whose wall is T × per-step latency, ~0.2–0.5 ms per LSTM at T≈200) →
  editor chunk time should drop from ~1.0–1.2 s to **GPU-bound ~150–250 ms**
  (t2-shaped: bert 18 + pred GPU + decoder ~15 + generator ~95 + readbacks).
- **IL2CPP builds:** roughly **unchanged wall** (the SIMD CPU LSTM was already
  10–15 ms) but the predictor becomes CPU-free — those cores go back to the game (and
  to the tenc/NSF tasks), and the pipeline loses a readback→CPU→upload sync pair.
- Recur occupancy is 2 threadgroups by design (serial dependency) — latency we pay, not
  throughput; the 1650 pays ~the same microseconds per step as the 4060 (L2-resident
  whh, latency-bound loop), so the ABSOLUTE pred cost stays small on the low-end tier.

## R2.3 What changed on disk (round 2)

- `KokoroCS.compute` — uniforms `ln_style`, `lstm_cat_dim`, `lstm_gather`; buffers `W2`,
  `W_bias2`; kernels `LstmInProjTile`, `LstmBiRecur`, `TransposeRC`; `LayerNormCoop`
  styled-affine branch. All 28 kernels dxc-verified (`cs_6_0`).
- `KokoroModel.cs` — `FastKernels3`; `LstmBiY` / `AdaLayerNormOp` / `TransposeOp`
  helpers (public, probe-graded); v3 predictor block in ForwardYielding (CPU Task path
  intact under the else); `lstmPreF/R` scratch; `ln_style` stale-uniform guards on the
  LN helpers; timing-field doc updates.
- `KokoroWeights.cs` — comment only (LSTM GPU copies now consumed).
- `KokoroKernelProbe.cs` — `fastKernels3` flag, routing line, part-A tests
  **14a LstmBiY durenc0 (cat)**, **14b LstmBiY shared (gather+cat)**,
  **14c AdaLayerNorm**, **14d TransposeRC** — all vs the KokoroCPU oracle on real fp16
  weights, maxabs < 1e-3.
- `KokoroKernelBatchRunner.cs` — menu `Run Kokoro Kernel Probe V2 (FastKernels3 off)`;
  V1/LEGACY menus force the lower tiers off too.

## R2.4 Next targets (diagnosed, NOT done)

- **NSF source (CPU, nsf-wait 66–86 ms in round 1):** with pred on GPU it starts earlier
  (F0 readback lands sooner) and must hide behind decoder + generator (~110 ms GPU); if
  `NsfWaitMs` stays > ~10 ms in the round-2 numbers, the sine-gen phase pipeline is the
  next persistent-kernel candidate — but it is also the RNG parity-injection home, so it
  moves only with a dump-injected GPU-RNG plan.
- **tenc biLSTM (CPU, overlapped):** trivially routable through the same kernels
  (in_dim = cat_dim = 512, no gather, + two TransposeRC) if `TencCpuMs` ever surfaces;
  left CPU to keep round 2's blast radius on the measured bottleneck.

## R2.5 Validation checklist (round 2 — replaces gate 1 above)

1. **Compile** — files in R2.3 only.
2. **Kernel + stage probe, four-way bisect** (report header prints
   `kernel routing: v3 (#33 GPU-LSTM) / v2 / v1 / LEGACY`):
   - `DeepUnity/TTS/Run Kokoro Kernel Probe` (v3, default) — expect **35/35 PASS**
     (round-1 rows + 14a–d) at maxabs < 1e-3; stage gates unchanged incl.
     **pred_dur exact (or ±1 on ≤2 tokens)** on t0/t1/t2 and **t0 wav ≥ 0.99**;
   - `... V2 (FastKernels3 off)` — round-1 behavior untouched (31/31);
   - `... V1 (FastKernels2 off)` and `... LEGACY` — still PASS;
   - batch: `KokoroKernelBatchRunner.Run` (v3 defaults, exit 0/1/2).
3. **The number that matters** — `[perf]` per text, v3 vs v2: pred CPU wall 844–894 ms →
   expect **PredCpuMs ≲ 20 ms** and the predictor stage GPU-bound; **end-to-end editor
   ~1.0–1.2 s → ~150–250 ms**. Record `nsf-wait` — if > ~10 ms, R2.4 is next.
4. **Duration integrity** (the user-facing risk): the probe already gates pred_dur vs
   the dumps; ALSO diff v3-vs-v2 pred_dur on the same texts (run both modes, compare the
   reports' pred_dur lines — must match exactly, else flag the boundary token count).
5. **Audio + pacing**: listen `ProbeLogs/kokoro_gpu_t0.wav` (v3 run); QwenKokoroPerfProbe
   int8 — D2 speak-alone ≤ round-1 frame stats (the predictor no longer competes with
   Unity's main/worker threads at all), B/C2 unchanged.
6. **1650**: same v3/v2 probe pair — record per-stage + RTF; expect the pred stage to be
   nearly as cheap in absolute ms as on the 4060 (latency-bound persistent loop).
7. Rollback: `KokoroModel.FastKernels3 = false` restores round-1 v2 byte-for-byte.
