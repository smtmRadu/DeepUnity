# pocket-tts #31-P deep-opt — coalesced GEMV/GEMM + fused flow head (`FastKernels2`)

Kernel-level optimization round on the pocket-tts GPU port, targeting the **AR FlowLM loop**
(~72% of generation: int8 801 ms / fp16 963 ms for the 10.4 s BENCHMARK.md clip on the 4060).
Same playbook as LLM #31 (Qwen 2.37×, Gemma 2.6-3.6×, MiniCPM 3.1-5× decode) and Kokoro #33:
autopsy first, additive kernels behind one static switch, parity probe before anything ships.
Written by the deep-opt workstream — **NOT yet compiled/measured in Unity**; the validation
checklist at the bottom is the main session's runbook.

## 1. Diagnosis (verified in source before writing anything)

- `PocketTTSCS.compute` `LinearBias` / `LinearBiasQ8` are naive GEMVs: `[numthreads(1,8,32)]`,
  ONE thread per output element, scalar loop over the whole `in_dim`; adjacent lanes read weight
  rows `in_dim` elements apart → fully uncoalesced weight traffic (~1/8-1/32 bus efficiency).
  EVERY matmul of the FlowLM backbone, flow head and mimi decoder_transformer dispatches there
  (`PocketTTSFlowLM.Linear/LinearRows`, `PocketTTSMimi.Linear/LinearRows`).
- Per AR frame the decode reads ~151 MB (fp16) / ~76 MB (int8) of weights — at ideal bus
  efficiency that is ~0.6 / 0.3 ms on a 4060, yet the measured loop runs ~6-7 ms/frame:
  the loop is **weight-bandwidth-bound at terrible coalescing**, exactly the #31 pathology.
- The flow head (SimpleMLPAdaLN, 1 Euler step/frame) issues ~66 dispatches of 512-wide ops
  (<1 MMAC each) per frame — dispatch/latency overhead rivals the math. Additionally the legacy
  path recomputed two **constant** timestep embeddings (s=0, t=1 every frame: 6 dispatches +
  CPU cos/sin + a transient ComputeBuffer) and re-materialized SiLU(y) per res_block (12 more).

## 2. What was built (all additive, gated by `PocketTTS.FastKernels2`, default ON)

### New kernels — `PocketTTSCS.compute`, "#31-P" section at the end
| kernel | replaces | shape rules |
|---|---|---|
| `LinearBiasCoal` / `LinearBiasQ8Coal` | LinearBias/Q8 at T=1 (AR decode GEMVs) | K % 128 == 0, K ≤ 4096 |
| `LinearBiasGemm` / `LinearBiasQ8Gemm` | LinearBias/Q8 at T>1 (prefill / mimi xf) | K % 128 == 0; `elem_offset` = first token row |
| `FlowResBlockFused` / `...Q8` | 8 dispatches per res_block (Copy, SiLU, adaLN GEMV, LN, Modulate, mlp0, mlp2, GateAdd) | one 256-thread group; D = 512 in groupshared (~15 KB) |
| `FlowFinalFused` / `...Q8` | 6 dispatches (Copy, SiLU, adaLN, no-affine LN, Modulate, linear) | same |

- GEMV/GEMM are straight ports of the proven Gemma3CS/Qwen3_5CS GVC/GMM patterns
  (256 threads = 8 rows × 32 lanes, input staged once in groupshared, lane reads 4 CONSECUTIVE
  packed weights, 32-lane tree reduce; GEMM adds 8-token tiles staged in 128-column chunks so
  every coalesced weight read is reused 8×). fp16/q8 are explicit kernel twins (pocket picks
  quant per TENSOR from the manifest `.scales` sibling — no multi_compile), consuming the
  existing weight buffers **as-is** (fp16 2-per-uint `readH`; q8 raw signed 4-per-uint + per-row
  fp16 scale factored out of the dot). fxc gotchas copied verbatim (per-token reduce loop NOT
  `[unroll]`; constant 5-step reduce IS; braces around macro locals; unique loop var names;
  group-uniform barrier flow with a `norm_dim == 0` prewarm early-out).
- Fused flow head: one persistent group per res_block — activations never leave groupshared;
  work split 64 rows × 4 lanes per tile (a warp covers 8 consecutive rows, each row's lanes read
  one full 32 B sector per step in both quants; 2 barriers per 64-row tile). SiLU(y) computed
  in-kernel from the raw cond vector. Kokoro `LstmBiRecur` is the precedent that fxc handles
  in-kernel multi-stage loops with barriers.
- New shader buffer slots `W2/W_bias2/W_scales2`, `W3/W_bias3/W_scales3` (fused kernels bind
  adaLN + mlp0 + mlp2 in one dispatch). No existing kernel touched; legacy section byte-identical.

### Routing — `PocketTTSFlowLM.cs`, `PocketTTSMimi.cs` (module files only)
- `Linear()` / `LinearRows()` in both classes route eligible shapes (K % 128 == 0 — i.e. K ∈
  {256, 512, 1024, 1536-in, 2048, 4096}; K=32 `input_proj` and 1-out `out_eos` stay legacy/CPU)
  to the coalesced kernels. `T==1` → GEMV, else GEMM. Slicing semantics UNCHANGED: the #29
  MAC-budget slicer and #30 mimi tail-restriction pass their row offset through `elem_offset`
  exactly as before (a ragged 8-token tail tile recomputes ≤7 rows with the same kernel —
  idempotent, the documented #29 overlap rule). Pump ticks, FrameBreak/GpuWait yields and the
  `GpuMacsPerTick` budget are untouched — InferencePerf AutoTune just re-measures cheaper ticks.
  (2026-07-27: nothing re-measures any more — `GpuMacsPerTick` is a fixed Backend Tradeoff row.)
- `FlowHeadIssue` gains a fused branch (`FlowHeadIssueFused`): input_proj → cached
  0.5·(temb_s+temb_t) (`fTimeComb`, computed once with the SAME legacy dispatches — (s,t) is
  constant (0,1) every frame, so caching is bit-identical) → cond_embed + assemble → 6×
  `FlowResBlockFused` → `FlowFinalFused`. The legacy body is untouched and still used when
  `FastKernels2 == false` **or `FlowTap != null`** (the P3 localization taps need per-stage
  intermediates) or on a (never-occurring) mixed-quant flow head.
- `PocketTTSMimi.ForceLegacyKernels` now ALSO forces legacy linears → it remains the
  "full pre-#30 baseline"; `PocketTTS.FastKernels2` bisects the #31 axis independently.
- `PocketTTS.PrewarmKernels()` covers the 8 new kernels + 6 new buffer slots (all degenerate at
  zeroed uniforms).

### Dispatch counts per AR frame (decode step + flow head)
| stage | before | after |
|---|---|---|
| FlowLM transformer step | ~85 (24 GEMVs naive) | ~85 (24 GEMVs coalesced) |
| flow head | ~66 | ~11 (+7 one-time time-embed cache) |
| **total** | **~151** | **~96** |

## 3. Expected speedups (RTX 4060, BENCHMARK.md clip, to be MEASURED not trusted)

| stage | int8 now | expected | lever |
|---|---|---|---|
| AR loop | 801 ms | **~320-420 ms (≥2×)** | 24 GEMVs/frame at ~4-8× bus efficiency (#31 measured 2.4× on the GEMV-dominated Qwen int8 decode) + flow head 66→11 dispatches |
| prefill | 82 ms | ~35-55 ms | GEMM tiles (gemma prefill measured 3.0×) |
| mimi decode | 236 ms | ~180-210 ms | decoder_transformer linears only (convs already #30-tiled — untouched) |
| e2e RTF | 0.108 | **~0.055-0.075** | sum |

fp16 gains proportionally more on the GEMVs (2× the bytes). Live streaming RTF (~0.29) carries
the same per-frame costs; the pump self-calibrates so the gain shows up as fewer/coarser ticks.

## 4. Parity contract

Unlike #30 (bit-exact), tree/lane reductions REORDER float sums — same acceptance as LLM #31:
- per-kernel legacy-vs-new gates: corr ≥ 0.999999, maxAbs ≤ 1e-2 (outputs O(1)-O(10));
- whole flow head velocity: corr ≥ 0.9999;
- mimi window decode wav: corr ≥ 0.99999, maxAbs ≤ 1e-3;
- mimi tail-restricted vs full decode UNDER the coal path: kept tail BIT-exact (the streaming
  flush contract — a GEMM token's value is elem_offset-invariant, tokens share no math);
- e2e offline A/B (same injected noise): **same frame count (same EOS step)**, latents
  early-frame corr ≥ 0.9999 / all-frame corr ≥ 0.999, wav mel-corr ≥ 0.99 (phase-invariant;
  raw corr informational — AR feedback turns ~1e-5 reorder noise into phase drift, not quality).
- The #30 tiled-vs-legacy gates still run BIT-exact: the probe forces `FastKernels2 = false`
  during them so the axes stay pure.

## 5. Validation checklist (main session runbook)

**Gate 0 — compile.** Open the project; expect 0 errors with only these files changed:
`PocketTTSCS.compute`, `PocketTTSFlowLM.cs`, `PocketTTSMimi.cs`, `PocketTTS.cs`,
`validation/Editor/PocketTTSKernelParityProbe.cs`, this file. If fxc complains about a fused
kernel (groupshared ~15 KB, well under 32 KB) that is a real bug — report back, don't patch
around it.

**Gate 1 — #30 regression.** Menu `DeepUnity/PocketTTS/Kernel Parity (tiled vs legacy)` —
must still print PASS (bit-exact) on both sub-gates. This proves the #31 additions did not
disturb the #30 axis.

**Gate 2 — #31 parity + [perf], fp16.** Menu
`DeepUnity/PocketTTS/Kernel Parity #31 (coal GEMV+flow, fp16)` — expect ALL rows PASS:
9 GEMV shapes, 5 GEMM shapes (incl. two mimi shapes), flow head, mimi window A/B +
bit-exact tail-restriction under coal, E2E A/B (same frame count + latent/mel gates).
Record the two `[perf]` lines (mimi window speedup; offline prefill/loop/mimi/total speedups).

**Gate 3 — #31 parity + [perf], int8.** Menu `... (coal GEMV+flow, int8)` — same gates; this
is the run that exercises every Q8 twin. Record `[perf]`.

**Gate 4 — RTF benchmark.** Menus `DeepUnity/PocketTTS/RTF Benchmark (int8)` and `(fp16)` —
compare against BENCHMARK.md (int8: prefill 82 / loop 801 / decode 236, RTF 0.108). The AR
loop must be **≥ 2× faster**; nothing may regress. For an in-place A/B set
`PocketTTS.FastKernels2 = false` and rerun. Update BENCHMARK.md pocket rows (main session owns
bench docs).

**Gate 5 — listen QA.** `ProbeLogs/pockettts_names_unity.wav` from
`DeepUnity/PocketTTS/P4 Offline E2E` (fp16 gates unchanged — note P4/P5 dump gates compare
against Python at corr 0.99, which the ~1e-5 reorder noise does not threaten) — no clicks,
buzz, or truncation vs the previous run.

**Gate 6 — live streaming.** Play-mode NPC talk (the standard talk-perf scene): voice audible,
no new frame spikes; `PocketTTS.GpuMacsPerTick` converges upward (coarser slices) — that is the
expected AutoTune response to cheaper ticks, not a bug. Streaming RTF should drop from ~0.29
proportionally to the AR-loop gain.
(Gate 6 as written expired 2026-07-27: `GpuMacsPerTick` no longer converges, it is a fixed Backend
Tradeoff row per tier. Re-run the gate by comparing frame spikes at a FIXED slice instead.)

**Rollback:** any failure → `PocketTTS.FastKernels2 = false` (one static; restores the exact
pre-#31 dispatch list — legacy kernels and the legacy flow-head body are byte-untouched) and
report which gate + which probe row failed.

## 6. Risks / non-goals

- **EOS flip risk:** ~1e-4 logit noise on `out_eos` could flip a frame whose margin sits inside
  it; the E2E gate hard-fails on frame-count mismatch so this cannot ship silently.
- **fxc scheduling** of the fused kernels is the main compile risk; every known gotcha from
  Gemma/Kokoro is copied, but only Gate 0 proves it.
- Deliberately NOT taken (diagnosed, low value/high risk): fusing QKV slice+RoPE+AppendKV
  (~6 tiny dispatches/layer-frame — next lever if dispatch overhead still dominates after
  measurement), attention micro-costs (CausalAttentionKV is already one dispatch/layer),
  a whole-Euler-loop mega-kernel (needs a weight arena — violates "consume buffers as-is"),
  and anything in the mimi conv stack (#30 owns it; untouched).
- No files outside `TTS/PocketTTS/**` + `PocketTTSCS.compute` were modified. Other models'
  shaders/kernels untouched — their performance is provably unaffected by file isolation.

---

# Round 2 (#31-R2) — GPU-resident AR frame (`FastKernels3`)

## R2.0 Why (what the R1 measurements proved)

R1 parity was flawless but the AR loop stayed flat (fp16 334→328 ms 1.02×, int8 283→225 ms
1.26×; prefill 1.93×/1.57× and mimi 1.21× DID materialize). The numbers falsify the R1
bandwidth diagnosis: int8 legacy at 6.2 ms/frame moves ~76 MB of weights that cost ~0.32 ms at
roofline — both legacy AND coal run 20-50× under the bus limit, so the loop is dominated by
**fixed per-frame overhead**, not traffic. Source audit (this round, verified before writing):

- **TWO BLOCKING READBACKS PER FRAME** in the offline loop — `DecodeStepKV` reads back c [1024]
  for the CPU EOS check, `FlowHead` reads back velocity [32] for the CPU latent add. Each
  `GetData` drains the entire GPU pipe: ~96 dispatches never pipeline across a frame.
- **THREE UPLOADS PER FRAME** — token embedding (CPU `input_linear` matmul → `d1In.SetData`),
  noise (`fNoiseIn.SetData`), and c re-uploaded (`fCondIn.SetData`) for cond_embed right after
  its own readback (a full GPU→CPU→GPU roundtrip of the conditioning vector).
- Streaming pays the same twice per frame as **async waits** (≥1 pump tick each).
- The flow-head fusion (66→11 dispatches) producing exactly the int8 1.26× confirms
  dispatch/serialization count is the lever.

## R2.1 What was built (additive, gated by `PocketTTS.FastKernels3`, default ON, layered on FK2)

**Instrumentation (item 1)** — `PocketTTS.PerfCounting` + counters: every FlowLM dispatch is
funneled through `Disp()` (exact counts), blocking/async readbacks and uploads counted at their
sites with wait-ms, legacy-loop CPU split (token / decode-call / flow-call) stopwatched, and
`StatLoopStart*` marks isolate the AR loop from prefill. The R2 probe prints per-frame numbers
for the R1 loop and the R2 loop back-to-back — the diagnosis is confirmed by measurement before
the speedup is graded.

**Zero per-frame syncs (item 2)** — the whole frame chain runs GPU-resident
(`PocketTTSFlowLM.DecodeFrameGpuIssue`):
- token = `input_linear(prev latent | bos_emb)` on GPU (K=32 legacy GEMV; feedback latent lives
  in `d1Lat`, bos uploaded once) — kills the CPU matmul + `d1In` upload;
- `AREosNorm(Q8)` fuses out_norm LN + the eos dot and writes the logit into the frame's slot of
  `eosLat` (stride 33 = [eos | latent[32]]) — kills readback #1;
- the flow head reads its conditioning DIRECTLY from the on-GPU c buffer (mode-2 Gemv epilogue
  adds the cached time embedding) — kills the c re-upload;
- noise is pre-uploaded in K-frame blocks (`noiseK`, ONE SetData per block; per-frame CopySlice
  into the [32] row the flow head consumes) — kills the noise upload;
- `ARCommit` computes latent = velocity + noise (bitwise-equal commutative add), writes the
  feedback buffer AND the frame's slot — kills readback #2.

Offline: K = `ArBatchFrames` (8) frames issue back-to-back with ZERO readbacks, then ONE
blocking readback of the K slots; the CPU EOS scan reproduces legacy semantics exactly (eos
checked before a frame's latent is emitted; post-EOS overshoot latents are DISCARDED before
mimi — emitted audio identical by construction, frame f's latent never depends on later frames;
up to K-1 frames of throwaway compute ≈ 2% of a 130-frame clip). Streaming keeps per-frame
pacing (pump semantics untouched) but makes ONE combined async readback instead of two.

**Fused transformer step (item 3)** — per layer 14 → 6 dispatches:

| step | was | now |
|---|---|---|
| norm1 + qkv | LayerNormT + GEMV | `GemvLN16/Q8` (LN folded into the staging pass) |
| q/k/v prep | 3×SliceCols + 2×ApplyRoPE + AppendKV | `ARQkvPrep` (RoPE math copied VERBATIM — bit-identical) |
| attention | CausalAttentionKV | unchanged |
| out_proj + residual | GEMV + AddResidual | `Gemv16/Q8` mode 1 (Y += r epilogue) |
| norm2 + linear1 | LayerNormT + GEMV | `GemvLN16/Q8` (GELU in epilogue) |
| linear2 + residual | GEMV + AddResidual | `Gemv16/Q8` mode 1 |

`gemv_mode` (0 write / 1 add / 2 write + buf_b[row]) exists ONLY on the new R2 kernels — the
proven R1 kernels never read it (no staleness risk). GemvLN's per-group LN recompute is ~1024
reads + ~18 barriers per group — trivial against killing a whole dispatch; identical reduce
order in every group ⇒ identical stats in every group.

**Dispatch count per AR frame:** legacy ~151 → R1 ~96 → **R2 ~49**
(1 token + 6×6 transformer + AREosNorm + noise slice + input_proj + cond_embed + 6 res_blocks +
final + ARCommit), with **0 readbacks / 0 uploads inside a block** (R1: 2 blocking reads + 3
uploads per frame).

## R2.2 Expected impact (to be MEASURED, not trusted)

Per-frame cost model: syncs gone (they were the bulk of the apparent ~50 µs/dispatch), ~49
back-to-back tiny dispatches ≈ 1-2 ms GPU + a 1/K-amortized drain. For the 10.4 s clip
(~130 frames): **AR loop int8 783 → ~150-250 ms, fp16 920 → ~180-300 ms** → RTF int8 toward
~0.05-0.06. Streaming: halved per-frame waits + no CPU token matmul → live RTF and TTFA both
improve; the pump self-calibrates (AutoTune) as before.

## R2.3 Files touched (round 2 — same territory)

- `PocketTTSCS.compute` — "#31-P ROUND 2" section: `Gemv16/GemvQ8/GemvLN16/GemvLNQ8`,
  `ARQkvPrep`, `AREosNorm/Q8`, `ARCommit`, `gemv_mode` uniform, `pvc_stat` (R1 section untouched).
- `PocketTTSFlowLM.cs` — `Disp()` instrumentation funnel (every dispatch), `BlockingRead`/upload
  counters, `CanRunGpuFrames`, `EnsureAr`/`UploadNoiseBlock`/`DecodeFrameGpuIssue`/
  `ReadEosLatBlock`/`ReadEosLatYielding`, `EnsureTimeComb` + `FlowBlocksIssue` refactor (the R1
  fused path emits IDENTICAL dispatches), probe hooks (`ReadCondForProbe`, `RunLNLinearForProbe`).
- `PocketTTS.cs` — `FastKernels3`, `ArBatchFrames`, `PerfCounting` + Stat* statics, K-batched
  offline loop (legacy loop kept verbatim + attribution stopwatches), streaming single-readback
  frame branch (legacy branch kept verbatim), prewarm += 8 kernels + `gemv_mode`.
- `PocketTTSKernelParityProbe.cs` — R1 menus now PIN `FastKernels3 = false` (axis purity); new
  menus `Kernel Parity #31-R2 (GPU-resident AR, fp16 / int8)`.
- This file. Mimi untouched this round.

## R2.4 Validation checklist (main session runbook, round 2)

1. **Compile** — files in R2.3 only; 40 kernels in PocketTTSCS.compute.
2. **R1 regression** — `Kernel Parity (tiled vs legacy)` bit-exact PASS; `Kernel Parity #31
   (coal GEMV+flow, fp16/int8)` all PASS (unchanged numbers — those menus now pin FK3 off).
3. **R2 parity + instrument** — `Kernel Parity #31-R2 (GPU-resident AR, fp16)` then `(int8)`:
   - `[instrument]` lines: the R1 loop must show ~2 blocking reads + ~3 uploads/frame and ~96
     dispatches/frame; the R2 loop ~49 dispatches/frame and ~1/K reads+uploads/frame — the
     sync-point hypothesis confirmed IN NUMBERS (brief item 1);
   - gates: 3× GemvLN rows, single-frame composite (c corr ≥ 0.9999, |Δeos| ≤ 1e-2, latent
     corr ≥ 0.9999), E2E same frame count + latents early ≥ 0.9999 / all ≥ 0.999 + mel ≥ 0.99;
   - `[perf]` line: the AR-loop R1→R2 ratio is THE acceptance number (target ≥ 2× — combined
     with R1's flat result that finally delivers the original ≥ 2×-vs-legacy bar).
4. **RTF benchmark** — `RTF Benchmark (int8)`/`(fp16)`: expect loop ≪ 783/920 ms, RTF int8
   toward ~0.05-0.06; update BENCHMARK.md (main session owns bench docs).
5. **Listen QA** — P4 Offline E2E (fp16 + int8): dump gates unchanged (corr 0.99 vs Python has
   ~2 orders of margin over the LN-reorder noise); listen to `pockettts_names_unity.wav`.
6. **Live streaming** — talk-perf scene: voice audible, no new spikes, TTFA should drop (one
   wait per frame instead of two); `ar_frame` replaces `ar_decode`/`ar_flowhead` in the
   LastHeavyTick attribution (diagnostics label only — pump contract unchanged).
7. **Rollback** — `PocketTTS.FastKernels3 = false` restores the exact R1 dispatch list;
   `FastKernels2 = false` on top restores pre-#31. Three-tier bisect, one static each.

## R2.5 Risks (round 2)

- **EOS values** now come from the tree-reduce LN (AREosNorm) — same flip-risk class as R1;
  hard-gated by frame-count equality in the E2E gate plus the explicit |Δeos| print in the
  single-frame composite.
- **Non-deterministic RNG consumption**: the offline batch pre-generates noise for issued (not
  emitted) frames — subsequent clauses see a different RNG phase than the legacy path. Sampling
  only; injectNoise (all parity paths) indexes absolute frames and is unaffected.
- **TTFA offline proxy** is now block-granular (first K frames land together) — informational
  only; streaming owns the real TTFA and improves.
- fxc risk concentrated in GemvLN (barrier-heavy staging) — patterns copied from the proven
  FRB_LNSTATS/PVC_REDUCE forms; checklist step 1 (compile) proves it.

---

# Round 3 (#31-R3) — mimi/AR overlap + TTFA ramp (`OverlapMimi`, `ArBatchRamp`)

## R3.0 Where R2 landed and what is left

R2 measured: AR loop fp16 329→233 (1.41×), int8 240→165 (1.45×); RTF int8 0.073 / fp16 0.099;
instrumentation confirmed the syncs are dead (R2 loop: 0.13 reads + 0.13 uploads/frame, CPU
split 0/0/0) and the loop is now GPU-execution-bound: the block readback wait is ~95% of loop
time and is simply the GPU running 51 tiny dependent kernels — **inter-dispatch dependency
bubbles**, not math (effective weight bandwidth still ~30 GB/s of a 236 GB/s bus). Two levers
remain that don't require new kernels; a third (whole-layer fusion) was assessed and SKIPPED.

## R3.1 Lever 1 — overlap mimi with the AR loop (scheduling only, `OverlapMimi`, default ON)

The offline path ran mimi strictly AFTER the loop: 189-199 ms of fat, high-occupancy conv work
that could instead fill the AR chain's bubbles. Now (`FastKernels3` offline path only):

- As soon as `MIMI_OVERLAP_CHUNK` (= 64 = `DecodeWindowed`'s chunk) latents are scanned, that
  window's decode (ctx `MIMI_DECODE_CTX`, tail-restricted — the streaming-proven path) is
  ISSUED with **no readback** (`PocketTTSMimi.DecodeIssueTo`); its kept tail is harvested
  GPU-side into a persistent assembly buffer (`wavAccum`) by a `CopySlice` (pure copy — bit-
  identical values, one existing kernel). The window executes interleaved with the NEXT AR
  blocks and is drained by their eos readbacks — off the critical path to the extent the
  driver overlaps independent dispatches (disjoint buffers; shared-cbuffer updates are
  versioned, not synchronizing).
- After the loop, the final ragged window is issued and ONE `GetData` of `wavAccum` drains
  everything.

**Determinism contract (the strongest defensible, probe-gated):** the windows and their
dispatch parameters are IDENTICAL to `DecodeWindowed(chunk 64)` on the same latents — only the
interleaving with AR dispatches differs, and window outputs are state-independent (the #30
tail-restriction never reads garbage-permitted regions — gates 2/B4b). Therefore:
- overlap wav == sequential `DecodeWindowed` wav **BITWISE** (gate C4-b, forced 160-frame run);
- for T ≤ 64 the single window IS the plain full decode → **BITWISE** equal to the old
  sequential tail (gate C4-a);
- latents are **BITWISE** unperturbed by overlap+ramp (gate C4-c — AR buffers are disjoint
  from mimi's);
- only 64 < T ≤ 128 differs from the OLD path (full decode) — by the established windowed
  past-receptive-field fp-noise (corr 1.0, maxAbs ~3e-5; mel-gated). T > 128 used
  `DecodeWindowed` before, so long clips are bitwise-comparable to the old path too.

**Accounting note:** under overlap, LoopMs absorbs hidden mimi work (it executes inside the
block-readback waits) and DecodeMs shrinks to the final windows + readback — the TOTAL is the
comparable number; the probe's `[perf]` lines say so explicitly.

## R3.2 Lever 2 — TTFA ramp (`ArBatchRamp = {2, 4}`, streaming `StreamFirstChunkFrames = 2`)

Flat K=8 regressed the offline TTFA proxy 57→79 ms (first readback waits for 8 frames). The
first offline blocks now follow `ArBatchRamp` (2, then 4, then K) — block 0's eos/latent
readback lands after ~2 frames of GPU; expected proxy ≈ prefill + ~2 frames ≈ high-50s ms
(int8). Ramp neutrality is bit-gated (C4-c compares ramped vs flat latents). Streaming gets
the twin: the FIRST flush fires after `StreamFirstChunkFrames` (2) instead of
`StreamChunkFrames` (8) — first audio reaches the ring ~0.5 s earlier at 12.5 Hz; only the
flush BOUNDARY moves (windowed decode is tail-exact, emitted samples unchanged); applied on
the GPU-frame path only so the legacy A/B stays byte-identical.

## R3.3 Lever 3 — whole-layer persistent fusion: assessed, SKIPPED

To fuse GemvLN(in_proj) → attention → out_proj into one dispatch, the wide GEMV stages (3072 /
1024 output rows across 384 / 128 groups) and the attention stage (16 groups) would need a
device-scope barrier INSIDE a dispatch — D3D11 has none, so the whole layer must collapse into
ONE threadgroup. That serializes ~12.6 M MACs + a 16-head × ~700-row KV attention into 256
threads at single-SM bandwidth ≈ 3-6 ms/layer — provably SLOWER than the current 6-dispatch
chain. The dependency structure (wide → narrow → wide) forces inter-dispatch barriers by
construction; the only remaining fusible pairs are micro (noise-slice into ARCommit, token
GEMV into GemvLN staging: ~2-3 dispatches/frame, <5%). Not worth the fxc risk this round —
skipped deliberately, per the brief's escape clause.

## R3.4 Files touched (round 3)

- `PocketTTSMimi.cs` — `DecodeIssueTo` (issue-only decode + GPU-side tail harvest; no kernels).
- `PocketTTS.cs` — `OverlapMimi` + `MIMI_OVERLAP_CHUNK`, `ArBatchRamp`, `StreamFirstChunkFrames`,
  `wavAccum`/`GrowWav`/`IssueMimiWindow`, ramped+overlapped FK3 offline loop, overlap-aware
  LoopMs/DecodeMs tail, streaming first-flush schedule (legacy cadence provably identical when
  firstChunk == chunk), Dispose.
- `PocketTTSKernelParityProbe.cs` — C0-C3 pin the R3 knobs OFF (pure R2 axis); new gate set C4
  (bitwise overlap determinism ×3, ramp neutrality, TTFA print, normal + forced-160-frame
  `[perf]`); `BitGate` helper.
- This file. Shader untouched this round (zero new kernels).

## R3.5 Validation checklist (round 3)

1. **Compile** — files in R3.4 only.
2. **Regression** — `Kernel Parity (tiled vs legacy)` and `#31 (fp16/int8)` unchanged PASS;
   `#31-R2` C0-C3 rows must reproduce the R2 numbers (knobs pinned off there now).
3. **R3 gates** — `#31-R2 (fp16)` + `(int8)` menus, new C4 section: all three BitGates PASS
   (bit-exact), TTFA proxy OFF→ON prints ~79 → high-50s, long-run `[perf]` total ratio is the
   overlap acceptance number (expect int8 735→~560-620 ms total on the RTF-shaped 160-frame
   run; flat = the driver serializes independent dispatches — report it, don't force it).
4. **RTF benchmark** — `RTF Benchmark (int8)`/`(fp16)`: expect int8 total ≈ 560-620 ms → RTF
   ~0.055-0.06, TTFA proxy back under ~60 ms; update BENCHMARK.md (main session owns it).
5. **Listen QA** — P4 Offline E2E gates unchanged (46-frame clip: overlap wav is bit-identical
   to the R2 wav); listen to `pockettts_names_unity.wav`.
6. **Live streaming** — talk-perf scene: first audio audibly earlier (first flush at 2 frames);
   no new spikes; flush cadence after the first chunk unchanged.
7. **Rollback** — `OverlapMimi = false` and/or `ArBatchRamp = null` restore the exact R2
   offline behavior; `StreamFirstChunkFrames = StreamChunkFrames` restores the R2 streaming
   flush schedule. All independent of the FastKernels2/3 tiers.

## R3.6 Risks (round 3)

- **Overlap win depends on the D3D11 driver overlapping independent dispatches.** Worst case
  it serializes: total GPU work is UNCHANGED for T ≤ 64 and T > 128 (identical dispatches to
  the old path) and mildly higher for 64 < T ≤ 128 (windowed vs full decode of a short clip —
  bounded by the tail-restriction savings). So the lever is upside-with-bounded-downside; the
  C4 long-run [perf] measures the truth.
- **VRAM**: `wavAccum` ≤ maxFrames×1920×4 ≈ 3.9 MB — negligible.
- **LoopMs/DecodeMs semantics shift under overlap** — documented in-code and in the probe
  prints; TOTAL is the cross-round comparable.
- Streaming first-flush ramp changes flush BOUNDARIES only; the ring consumes variable-size
  pushes by design (same mechanism as the existing end-of-stream partial flush).

# Round 4 (#32) — retained voice-prompt KV across clauses (no flag, no new kernels)

## 32.0 Why (what the streaming TTFA line proved)

Every clause re-prefilled the flow LM from scratch: `ResetKV()` then a block
`PrefillKVYielding(prefix, Lp, Lp + maxFrames)` over `prefix = [bbv (1 row) | voicePrompt
(125 rows) | textEmb (~25 rows)]`. Rows 0..125 are byte-identical AND position-identical on
every clause of a reply — only the text rows differ. That waste was the measured **392-604 ms
`synth→first-audio` dead window** in the `[TTFA]` line, during which playback drains and the
ring starves.

The cost is NOT mainly GPU time: `PrefillKVYielding` yields ~4 ticks per layer × 6 layers, and the
pump admits only `maxHeavyTicks` of those per frame — 6 on Very Smooth/Smooth, 2 on Very Fast.
So ~24 ticks spreads over 4-12 frames, **67-200 ms** at 60 fps. (An earlier draft of this section
said "one tick = one frame, so ~24 ticks ≈ 400 ms": wrong on both counts — `PumpPipeline` counts
FrameBreaks and breaks at the cap, it does not end the frame on each one. The measured 63-78 ms
compute saving stands; only the pacing figure was inflated.) Cutting rows cuts ticks, which is
what the ring actually feels.

**The trade this buys, stated honestly (review 2026-07-28).** The per-row path is CPU-*issue*
bound where the block path was GPU bound, so clause-start main-thread issue time goes UP ~6.5×
and lands in ONE frame instead of spread over 4-12: measured 20.8 ms across 3 MoveNexts for a
55-row clause versus 3.2 ms across 24 for the equivalent block prefill. `ROWS_HARD_CAP` bounds the
per-TICK cost (~9 ms), not the per-FRAME cost — the pump's own 12 ms `frameBudgetMs` is what bounds
that, plus one tick of overshoot. Net: one likely dropped frame at each clause boundary, in
exchange for removing ~75 ms and 4-12 frames of dead window in which the ring drains and the voice
starves. On the low tiers that is the right direction (audio continuity over frame smoothness is
the whole premise of those rows), but it IS a regression in frame time and should not be described
as a free win.

## 32.1 What was built

Retain ONLY the speaker conditioning — no text, no latents, no EOS — so the model still sees
exactly one utterance per clause and nothing goes out of distribution.

- `PocketTTSFlowLM.cs`: `promptKey`/`promptRows` + `CanReusePromptKV` / `RetainPromptKV` /
  `InvalidatePromptKV` / `BeginFromRetainedPromptKV` / `RetainedPromptRows`, and
  `AppendRowsKVYielding(rows, count)` — pushes the text rows through the EXISTING per-row
  `DecodeStepKVIssue`, batching rows per tick against `PocketTTS.GpuMacsPerTick` the way
  `LinearRows` does. No new compute kernels; the shader is untouched.
- `PocketTTS.cs`: the clause start branches on `CanReusePromptKV`. On a hit it appends only the
  text rows; on a miss it does the old full prefill and then `RetainPromptKV(voicePrompt, Lv)` —
  free, because the prefix STARTS with the prompt, so rows [0,Lv) already hold its K/V.

**Measured** (GTX 1650, fp16, prompt-cache probe 2026-07-28; 125-frame prompt ⇒ Lv = 126 — the
same 126 prompt rows are skipped either way):

| clause | prefill rows | prefill `FrameBreak` ticks | whole-clause synth |
|---|---|---|---|
| 14 tokens | 140 → **14** | 24 → **0** | 541 → **467 ms** (−74) |
| 25 tokens | 151 → **25** | 24 → **1** | 1016 → **953 ms** (−63) |

≈ 11.4 → 1.9 GMAC. The tick column is the second, larger win: the pump ends the frame on a
`FrameBreak` and allows 2-6 heavy ticks per frame at a clause start, so ~4 frames of pacing
disappear on top of the 63-74 ms of compute. This is NOT the whole 392-604 ms window — the AR
frames up to the first flush and that flush's Mimi decode are the rest — but it is the part that
was pure waste.

Cost side, measured and bounded: the per-row path is CPU-ISSUE bound (~40 tiny dispatches/row,
~0.4 ms of issue each) where the block path was GPU bound (~0.04 ms of issue per tick buying
~15 ms of GPU). A 25-row clause is ~10 ms of issue, inside the pump's 12 ms clause-start budget
(`gpuBudgetMs` 6 × `TtsSilentRefillBudgetScale` 2); `ROWS_HARD_CAP = 24` in
`AppendRowsKVYielding` keeps one tick under that budget at the Very Fast tier, where the MAC dial
alone would have handed it a 52-row (~21 ms) tick.

## 32.2 Invalidation (the whole risk — the flow LM is SHARED between voices)

Keyed on the **identity of the `voicePrompt` array**, never on "have I prefilled before": two
NPCs with different voices alternate on one engine, and `SetVoice`/`CloneVoice`/
`BindRawVoicePrompt` all assign a FRESH array, so a swap fails `ReferenceEquals` and falls back.
`CanReusePromptKV` additionally requires: the same row count; `kvCap >= maxTotal` (because
`EnsureKV` releases and re-creates the caches when it grows, and `maxTotal = Lp + maxFrames`
varies with text length); and live `ComputeBuffer`s (play-mode exit / device loss). Invalidated
by `EnsureKV` on growth, `PrefillKVYielding` (rewrites row 0), `ResetKV`, `Dispose`,
`PocketTTS.Defetch` and `LoadCpuTensors` (it re-reads `bbv`, which is row 0). Anything
uncertain ⇒ full prefill.

## 32.3 Parity contract — BIT-EXACT, and why that bar is reachable

Unlike #31 (tolerance-gated), this round is bit-exact, because the per-row path is the same
arithmetic in the same order as the block path for a single token:

- `CoalEligible` keys on `in_dim` only, so both T=1 and T>1 route to the same tier; and
  `LinearBiasCoal`'s per-lane order (4 consecutive, stride 128, same `PVC_REDUCE` tree) is
  exactly `LinearBiasGemm`'s. Same for the q8 twins.
- `CausalAttentionKV` over `kv_len` rows IS `CausalAttention`'s last-row output (same
  j-ascending accumulation, same online-softmax constants — see the kernel comment).
- `LinearBiasGemm` keeps one accumulator per token, so a row's value is independent of `Lp`
  and of the `LinearRows` slicing: rows cached during a SHORTER prefill are the rows a longer
  prefill would have written.
- LayerNorm/RoPE/residual are per-row ops; RoPE gets the same absolute position either way.

## 32.4 Validation

`PocketTTSPromptCacheProbe` — menu `DeepUnity/PocketTTS/#32 Retained Voice-Prompt KV Parity`
(+ `(int8)`), batch entries `Run` / `RunInt8` (self-exiting 0/1, report in
`ProbeLogs/pockettts_prompt_cache.md`). Six runs on ONE engine and ONE injected noise block:
cold voice2/A, full voice1/A, full voice1/B (grows kvCap), **retained** voice1/A, **retained**
voice1/B, then a voice swap back to voice2/A to fire the fallback. `LastPrefillRows` is asserted
on every run so a silent reuse (or a silent fallback that would make parity trivial) fails, and
an `Alive` gate rejects a "parity" of two silent runs.

Result 2026-07-28, both quant tiers, **maxAbs exactly 0** on all three comparisons:

| gate | fp16 | int8 |
|---|---|---|
| clause A retained vs full | 51840 samples, maxAbs 0 | 48000 samples, maxAbs 0 |
| clause B retained vs full | 103680 samples, maxAbs 0 | 96000 samples, maxAbs 0 |
| voice swap fallback vs cold full | 59520 samples, maxAbs 0 | 63360 samples, maxAbs 0 |

Regression: `Kernel Parity #31-R2 (GPU-resident AR, fp16)` re-run after the change — all #31-R2
and #31-R3 gates still PASS (bit-exact ones included). The OFFLINE path is untouched: it still
does the full prefill every call (`GenerateOffline` → `ResetKV`, which now also drops the
retained marker).

## 32.5 Known limitation (not a correctness issue)

Retention is a SINGLE slot, so in a two-NPC conversation the first clause of each turn pays the
full prefill (the voice swapped) and only clauses 2..N of that reply are cheap. A per-voice slot
would need either a second set of cache buffers (2× the KV VRAM) or a prompt-region allocator —
deliberately out of scope here.

# Round 5 (#33) — the blips: panic band + K-blocked streaming AR

## 33.0 Why (what the 2026-07-30 logs proved)

Two findings from the same day, one reply pattern: `in-reply silence 2.36-4.28 s` per long reply,
split by the `pause after drain` line into a **re-gated** term and a **dry** term.

1. **Re-gate was chunk-quantized.** Audio reached the ring only in whole-chunk lumps (chunk 16 =
   1.28 s), so after a dry event the 0.25 s `TtsRegateSeconds` re-gate actually waited for the
   next full lump: measured 1.20-2.80 s of re-gated silence carried by dry spells half that size.
2. **Production sat at ~1.0× playback.** The streaming AR path read back every frame
   (`#31-R2 "streaming always runs per-frame (latency)"`), and that per-frame readback latency —
   ~73 ms/latent while the LLM decodes, ≈ 1 latent per Unity frame at decode-era frame times —
   capped production at almost exactly playback speed. The ring limit-cycled off zero; after the
   panic band (below) removed the long holes, what remained was 2-15 bursts of ~80 ms per reply:
   the "blips".

## 33.1 Panic band (`InferencePerf.TtsPanicFloorSeconds = 0.25`)

Below it (or while playback is gated), two things change:
- **hurry-flush** — `PocketTTS.StreamHurry` (a voice-owned hook, null in probes) suspends the
  chunk cadence and decodes every `StreamHurryMinFrames = 4` pending latents. Delivery
  granularity while it matters becomes 0.32 s, which is what a 0.25 s re-gate can actually
  resume on. Above the band the chunk-16 cadence (and its tick amortization) is untouched.
- **the LLM waits** — a third `NoteTtsStarving` site in `PumpPipeline`: mid-reply low ring still
  does not hold the LLM (reverse-arbiter reasoning stands), but below the band a hole is a
  certainty at playback speed, not a risk.

Result (2026-07-30 log, same GTX 1650): re-gated silence **1.20-2.80 s → 0.00 s on every reply**;
5 of 9 replies fully clean; `buffer-gate` 1175-1248 → 644-687 ms on LLM-idle replies.

## 33.2 K-blocked streaming AR (`StreamArBatchFrames = 4`, ramp `{1, 2}`)

The streaming loop now issues K chained GPU-resident frames per combined `[eos|latent]` readback
(the offline #31-R2 block construction, applied to streaming): the readback latency that WAS the
production cap is amortized K-fold. The outer loop scans buffered frames one per iteration, so
EOS semantics, the flush schedule and the hurry-flush are untouched; overshoot is bounded at K-1
discarded frames per clause; issued frames never exceed maxFrames, so KV capacity needs are
unchanged. Pacing stays honest — one `FrameBreak` per issued frame (between issues, so K=1
degenerates to exactly the old schedule) — the block only removes readback stalls.

## 33.3 Validation

- `PocketTTSStreamBatchProbe` — menu `DeepUnity/PocketTTS/#StreamArBatch K-Block Parity`
  (+ `(int8)`), batch `Run` / `RunInt8`. Same text + same injected noise: K=1 ramp-off (the exact
  old per-frame schedule) vs K=3 flat (ragged blocks, EOS mid-block) vs K=4 ramped (shipping
  config). **maxAbs exactly 0, identical frame counts, both quant tiers** (2026-07-30).
- `PocketTTSPromptCacheProbe` re-run after both changes: all gates PASS, and its per-run
  frames/samples/rms line matched the per-frame path's morning run digit for digit —
  cross-implementation identity, not just within-run parity.
- **Async-path incident (2026-07-30, in-game):** the first shipped build crashed in
  `ReadbackYielding` — `NativeArray.CopyTo(dst)` demands `dst.Length ==` the request's length,
  and the ramped blocks read `count = blk*33` into a steady-K-sized dst. Every probe above runs
  `AsyncReadback = false`, whose sync `GetData(dst,0,0,count)` is partial-copy tolerant, so none
  of them could see it; the voice prewarm hit it on the first block and the voice was dead for
  the session. Fixed with an explicit-length `NativeArray.Copy(..., count)` (the method's stated
  contract), and the probe gained a third entry — `#StreamArBatch Async-Path Parity` — that runs
  the SAME synthesis with `AsyncReadback = true` pumped from `EditorApplication.update`
  (`AsyncGPUReadback.WaitAllRequests()` closes each fence: an idle unfocused editor stops pumping
  readbacks entirely, and the hardwait guard counts `Time.frameCount`, frozen in edit mode — a
  harness constraint, play mode pumps every frame). Result: 94080 samples, **maxAbs 0** vs the
  sync path. Rule this bought: any change on the streaming path must run the async gate too.
- P5 (`PocketTTSStreamProbe`) needs Python reference dumps (`validation/dump/*.npy`) absent on
  this machine — not runnable here, before or after.

## 33.4 Open

The panic band and K-blocking fix delivery and rate; what they cannot fix is fps itself — at
sub-15 fps (sync LLM decode) production margin is thin whatever the schedule. If blips return on
longer replies: raise `StreamArBatchFrames` toward 6-8 (watch per-frame GPU: the tick cap
already bounds it), or revisit the Smooth speaking-ticks column.

## 33.5 Addendum (#33b) — clause-lifetime preallocation (the 174/286 ms stalls)

The armed FrameSpikeProbe run (2026-07-30, `ProbeLogs/frame_spikes.csv`) attributed the felt
"load-up freezes" to ALLOCATION, not streaming: 174.8 ms (warmup's cold `EnsureKV` + prefill
scratch) and 286.4 ms (first real clause regrowing kvCap ~138 → ~700 — ~45 MB re-created in one
frame). Fix: `PocketTTSFlowLM.PreallocateYielding` + `PocketTTS.PrewarmAllocationsYielding`,
drained by `PocketTTSVoice.PrewarmRoutine` in the walk-up — one REAL driver allocation per frame
(a buffer is atomic; per-create cost calibrated from the log at ~6-40 ms, placed where the player
isn't looking), yields only on frames that allocated, `kvCap` published last, growth refused
while `kvLen > 0` (a live clause owns the rows — verifier finding A). Bounds are voice-independent
worst cases: clone-cap prompt rows (135) + 192 text rows (a verbose two-sentence clause measures
~109 tokens; the 1000-char emergency comma cut ≈ 279 stays deliberately out) + DefaultMaxFrames.
Side effect worth knowing (verifier finding B): `CanReusePromptKV` demands `kvCap >= maxTotal`,
so pre-fix every clause longer than all previous ones re-prefilled COLD (mid-reply `prefill`
spikes at t=32/t=39 in the log); pinned at the bound, #32 retention now serves every clause under
it. Adversarially verified (opus agent, 8 attack surfaces): two must-fix findings (the `warmed`
latch burning on defetch-mid-prealloc; the original one-buffer-PAIR-per-frame spread still being
6 x 40-77 ms frames) and two should-fix (the 118-row effective bound; the kvLen race) — all four
landed. Deferred with eyes open: Mimi's decode scratch re-creates (~47-110 MB across the first
clause's flushes) are unproven as a visible spike against the decode-contention plateau; ~55 MB
of prealloc VRAM is held from first zone entry for the session. All five parity probes PASS
after the change (PromptCache fp16/int8, K-Block fp16/int8, Async-Path).

## 34 (#34) — the CRUISE band: the renderer's seat at the #29 table (2026-08-02)

Two findings from the armed FrameSpikeProbe on a real Velmire session, after fixing the probe's
own attribution bug (`LastHeavyTick` read WITHOUT clearing — after a reply's final flush the tag
said "flush_push" for the rest of the session, which is what the "163 s flush_push storm" in the
first CSV actually was; the field now stamps `LastHeavyTickFrame` and readers ignore stale tags):

1. **Speaking cost ran unthrottled on a fat ring.** With the LLM idle and the ring holding
   3.7-7.3 s of banked audio, the pump still issued the tier's full speaking column (Smooth: 4
   ticks ≈ 16-24 ms GPU on the 1650) every frame — the reported 60-70→25-35 fps dips WHILE an
   NPC speaks. The #29 arbiter only ever ceded frames to the LLM; nothing ever ceded to the
   RENDERER. Fix: above `ttsCedeHeadroomSeconds` (+`InferencePerf.TtsCruiseEnterExtraSeconds`
   to enter, headroom itself to leave — hysteresis, or the integrator parks at the boundary),
   the tick cap drops to speaking / `InferencePerf.TtsCruiseTickDivisor` (Smooth: 2). Floor,
   panic and hurry-flush below the band untouched.
2. **The prefill boost fired with 7.3 s banked.** The 2026-07-26 "never cede through a dead
   window" reasoning dates from floor-hover, where a ~1 s bank in 1.28 s lumps genuinely could
   not cover a ~0.5 s dead window; in cruise the bank covers it by construction, so
   `clausePrefilling` no longer counts toward pushHard while cruising.

Measured (NpcTalkPerfProbe, 4 turns, vsync off, same protocol): the dominant SPK+AUD band —
5825 of 6779 frames — runs 17.25 ms mean / 22.9 p95 (was 24-40 ms sustained pre-fix), audio
still clean: `in-reply silence 0.00s` on three replies, one 0.04 s dry burst on the longest.
What remains and is NOT this fix's business: the GEN+SPK+AUD collision band (~283 frames at
~67 ms mean — sync LLM decode + TTS at reply start, the 33.4 note above still applies) and the
one-time cold-clause prefill singletons (up to ~310 ms). Pump telemetry for all of this is now
first-class: `PocketTTSVoice.Pump*` statics + `ProbeLogs/fps_timeline.csv` (one row/second) —
what the pump did, was allowed, and held banked, on every frame a probe cares about.

## 34.1 Addendum (#34b) — voice-prepare: the dialogue-open hitch (2026-08-02)

With the storm gone, the user's own session isolated the OTHER dip: dialogue open ran ~1 s at
17-25 fps with one 311-365 ms frame, tagged `pump=prefill`. Root cause, in ONE pump frame at the
first real clause: `BindVoice` → `CloneVoice(clip)` — `ClipToMono`'s blocking decompress-wait on
the reference MP3 (a `Thread.Sleep` loop!), resample 44.1→24 kHz, SHA-256 over ~1 MB, the
Resources cache load — then `InvalidatePromptKV` forcing the full ~125-row voice-prompt prefill
at the silent-refill boost. The session prewarm never touched any of it: its "Hi." synth runs on
whatever voice the engine happens to hold, not this NPC's clone.

Fix: `PocketTTSVoice.PrepareVoiceNow()` — kicks `LoadAudioData` async immediately, then (weights
resident, engine free) runs `BindVoice` + one tiny DISCARDED synth with the real voice, so #32
retention holds the voice-prompt KV and the first real clause prefills only its text rows.
Called from NPCChatBase at zone enter (hides in the walk-up) and at StartInteraction (covered by
the open animation + ~2 s first-token LLM latency). The pump gates on `prepareJob` exactly like
`prewarmJob`; the prepare aborts the moment real speech is queued; a static
`s_engineSideJobBusy` (+ per-voice `sideJobHeld`, released in OnDisable — the mute-latch failure
class) keeps prewarm/prepare synths from ever interleaving on the shared engine's single KV.

Verified (talk-perf, same protocol): turn-1 worst frame 91.8 ms (was 311.8), no frame over
~162 ms anywhere (was 250-312), first-reply `synth→first-audio` 392 ms (was 534-701), all four
replies `in-reply silence 0.00s`, zero starves. Remaining, unchanged and known: the
GEN+SPK+AUD low-ring collision (~100 ms frames while the LLM decodes INTO a clause start) — the
sync-decode plateau of 33.4, the next hunt if anyone reopens it.

## 34.2 Addendum (#34c) — shared-frame split + the boosted-open protocol (2026-08-02)

The GEN+TTS collision (97-162 ms frames): on frames the LLM issued GPU work, EVERY pump band now
divides its tick cap by `InferencePerf.TtsSharedFrameTickDivisor` (2) — via the new
`FramePacing.LlmIssuedRecently` (the raw mark; `LlmBusy` = mark + cede ration, and the ration is
the cede sites' business only). First split run measured the cost of splitting blindly: three
0.08 s dry bursts, clause dead windows losing the refill race by exactly the halved ticks. So the
CRITICAL band is exempt — audible with the ring under 2× `TtsPanicFloorSeconds` keeps the full
boost (under the floor itself the LLM is held outright; the exemption widens the sprint zone from
"hole open" to "hole one chunk away"). Verified: zero starve warnings, `0.00s dry` on every turn;
collision band mean 70→62 ms, worst in-conversation frame 311.8→~122 ms across the three rounds.

The probe now tests what the author actually does (his call — the old protocol waited for
LlmReady at the zone edge, so the "boosted model loading" dips were never exercised): phases
settle → walk-up (3.5 s of slow prefetch only) → BOOSTED OPEN (StartInteraction with weights
still streaming) → turns, recording continuously. First measurement of the open: ~4.7 s at
~50 fps mean (20.1 ms, p95 30.8) with singleton hitches (one 235 ms in boot+warmup) — the
sustained-collapse era is over; what remains there is one-shot load noise.

Two ledger notes. (1) A once-per-turn-boundary `0.80s re-gated` in the drain line with zero
starves and zero dry is Unity's stream-clip reader prefetching zero-fill into a PAUSED clip while
the next reply's text is already feeding — counted, never heard. The pause-keeps-clip comment in
StopSpeaking documents the buffering; treat the number as gate-time only when a starve warning
brackets it. (2) Multi-second `idle` frames with no GC flag, no tick, no pump snapshot
(2.9 s / 0.6 s mid-conversation on 2026-08-02) are machine-level stalls — the box was
simultaneously running a dataset upload; nothing in the engine was on the frame.

## 35 (#35) — sliced decode: the GEN burst finally splits (2026-08-02)

The sync-decode plateau 33.4 and 34.1 kept deferring to — "the next hunt if anyone reopens it" —
got reopened by the author the same day, with the criteria list rewritten first: latency no
longer counts ("nu mai conteaza delayul, doar sa fie smooth — niciun frame peste 50 ms, ideal
nici peste 33"). That matters more than the fix, because the 2026-07-26 always-sync verdict
(BackendTradeoff.cs) was never wrong on its own terms: async's right answer depended on the
hardware (the fps/3.5 dribble on weak GPUs, the old `MinUsableTokS = 12` floor), and a fixed
table cannot measure. Strike tok/s from the criteria and the dependence goes with it — the knob
becomes const-shaped again, just pointing the other way.

**Cause.** Sync decode issues a token's ENTIRE forward in one frame and blocks on the readback:
~1.06 GB of weights read per token on the 1650 (0.55 GB of INT8 layers + 0.5 GB of fp16 lm_head)
= 30-55 ms of GPU in ONE frame's queue. No readback strategy softens a lump already in the
queue, which is why #20's burst_async never killed the display hitch either — it only cleaned
the CPU-side numbers. Measured baseline (talk-perf, 2026-08-02 10:56): GEN 32.68 ms mean /
55.02 p95 / 57.3 max; GEN+SPK+AUD 57.32 mean / 86.24 p95 / 122.3 max; decode 12.0 tok/s.

**Fix.** Decode ISSUE is sliced across frames; the result never was splittable and still is not.
`Qwen3_5Model.ForwardYielding`'s seqLen==1 path now yields every
`InferencePerf.LlmDecodeSliceLayers` (6) layers — 24 layers = 4 slices of ~6-9 ms — with the
lm_head alone in a fifth frame (~11-15 ms; one GEMV, indivisible without a kernel change, the
true frame-cost floor of the scheme). The caller's async `SampleYielding` readback gates the
next token, bounding the in-flight backlog to ONE token — which is precisely why #20's
CPU-out-issues-GPU queue backup does not return at this width. `BackendTradeoffTable.UseSyncDecode`
flipped true→false (still a const, still not a dial — the right answer now depends on nothing);
`DecodeStep`'s sync branch survives for A/B archaeology. Dispatch order is untouched, so parity
is exact by construction and gated as such: `QwenDecodeProfileProbe.RunSlicedDecodeParity`
(burst vs sliced, 32 greedy tokens) — PASS, tokens identical, final logits bit-exact
(maxAbs 0.0E+0); the #31 GEMV tolerance gate re-run — PASS.

**Measured after** (same talk-perf protocol, clean run 11:27; an earlier run hit a cluster of
`idle+GC` machine stalls — 0.8-7.9 s frames on the no-pagefile box — that belong to ledger note
34.2(2), not to any bucket's physics):

| bucket | before (10:56) | after (11:27) |
|---|---|---|
| GEN | 32.68 mean / 55.02 p95 / 57.3 max | **14.55 / 17.13 / 33.6** |
| GEN+SPK+AUD | 57.32 mean / 86.24 p95 / 122.3 max | **17.64 / 23.65 / 82.8** |
| decode | 12.0 tok/s | 5.4-8.7 tok/s (per-turn 6.1/8.7/6.4/6.2) |

Audio through the slower decode: all four turns `in-reply silence 0.00s (0.00s dry + 0.00s
re-gated) in 0 bursts`, zero starve warnings, pump ceded 24-88 LLM frames/turn (the #29 arbiter
working, not fighting). The GEN+SPK+AUD max above ~60 is three cruise-band singletons
(70-83 ms, `ar_frame @cruise` with 6-7 s of ring banked — far exceeding any slice's possible
cost, i.e. more machine noise, not burst physics); the 95-123 ms `ar_frame @prefill ring 0.00`
frames in GEN+SPK are the known cold-clause-prefill singletons of 34.1, untouched here. The
tok/s halving is the accepted price: speech at ~3 words/s paces the conversation, and ~6 tok/s
≈ 4.5 words/s still feeds it with slack.

Ported the same frame (same gate, same constant, comments point back at Qwen): Gemma3Model and
MiniCPM5Model ForwardYielding — both only ever had the pre-#20 per-layer spread, so for them
slicing is ALSO a decode speedup (~numLayers+2 frames/token → ~numLayers/6 + 2). Compile-clean;
their GEMV parity probes need `weights_gemma3_270M_*` / `weights_minicpm5_1B_*`, which the
Pavilion does not hold — by-construction argument only until a weights-bearing machine re-runs
`GemmaCpmGemvParityProbe`.

## 36. The 2026-08-02 afternoon sweep — the last conversation-time spike classes (#36)

Three distinct killers fell in one afternoon, each unmasked by the previous one's fix, all found
by cross-reading npc_talkperf.md worst-20 against the scene probe's frame_spikes.csv (the tag
columns RACE — both probes read-and-clear LastHeavyTick, so trust ticks/GC/llm_phase, not tag,
when both run):

**(a) Voice-prepare racing the session warmup / LLM boot** (329-374 ms at zone entry, the
"second visit is always smooth" report). PrepareVoiceRoutine gated only on the TTS side; it now
also waits for `LLM.CurrentPhase == "idle"`, runs PrewarmAllocationsYielding itself (idempotent —
first visit can outrun the session warmup, and the mini-synth then paid the KV/scratch driver
allocations: the same 174+286 ms pair the 2026-07-30 hunt measured), and paces at ONE heavy
tick/GpuWait per frame (two ticks was half the 110→70 fps walk-up dip).

**(b) Canvas.ForceUpdateCanvases per reveal step** (82-163 ms, zero-tick frames, growing with
transcript length — NOT an engine cost at all). The audio-synced typing effect pops+re-adds the
bubble near every frame, and each AddMessage forced a synchronous rebuild of the whole
transcript. SoulsChatWindow now defers streaming-mutation scroll pinning one frame and never
forces; genuinely new lines keep the same-frame settle. >22.2 ms frames: 306 → 111 on the spot.

**(c) Text rows billed at MACs in the #32 clause start** (96-169 ms `ar_frame @prefill ring 0`,
the last standing family). A retained-prompt text row is ~76 MMAC but ~40 tiny GEMV dispatches —
2-4 ms of real GPU on a latency-bound card. Smooth's 900 MMAC tick bought 11 rows, the pump's
4-tick clause-start allowance stacked 44 ≈ 90-170 ms in one frame. AppendRowsKVYielding now
bills rows at max(realMacs, `InferencePerf.TtsTextRowDispatchEquivMacs` = 400 M): a cost-model
shape, tier dial untouched.

**Verified (12:18 run, focused, vsync off):** worst conversation-time frame 50.1 ms (from 169);
aggregate >33.4 ms: 42 over 15,176 frames, every one in the load window; audio `0.00s dry +
0.00s re-gated, 0 bursts` on all four turns — not even the paused-clip phantom.

**Still open (once per SESSION, mid-walk-up, GC-clean):** 135 ms (first voice-prompt prefill) +
215 ms (first mimi decode) inside the voice-prepare — driver-side one-shots that survive the
frame-0 zeroed-uniform PrewarmKernels, i.e. whatever the driver defers until the first REAL-sized
dispatch. Next lever: make PocketTTSVoice.PrewarmKernels' real "Hi." pass (or the prepare itself)
the designated payer and slice it finer, or prove the cost is driver heap growth and pre-size it.
Plus the scene-load GC cluster (4.0 s + 0.6 s at t≈4, editor asset load) — not engine's to fix.

## 36.1 The walk-up pair: four rounds, verdict — platform residency, not engine scheduling

Target: the two once-per-session ~150-290 ms frames at t≈13-15 s (zone entry, TTS just turned
ready, voice-prepare running). Four controlled rounds on fresh editor boots:

1. **Whole-buffer zero-touch in the walk-up prealloc** (commit the pages before the first real
   prefill/flush): the stalls LEFT the synth path (tags gone) but kept their magnitude — the
   touch frames inherited them. Cost tracks BYTES per RESOURCE, not dispatch count.
2. **Chunked touch** (`TtsFirstTouchElemsPerFrame`, 4 MB/frame, ZeroBuffer made
   elem_offset-aware): no change — WDDM residency is per-resource; the first chunk's dispatch
   references the whole buffer.
3. **Frame-0 static create pool** (SEANet scratches made in the scene-load blackout, adopted by
   Grow): no change — residency happens at first dispatch REFERENCE, not create. Rolled back.
4. **Frame-0 create + touch** (reference each in the blackout, card empty): no change
   (247+218 ms) — the ~1 GB Qwen upload between frame 0 and the walk-up evicts idle scratches;
   first real use re-pays the migration. Rolled back with round 3.

What stays from this hunt: the prealloc coverage + chunked ZeroTouch (rounds 1-2 — they moved
the cost OFF the synth path and shaved ~20%), the elem_offset-aware ZeroBuffer, and the
constant. The residual pair is the OS paying VRAM eviction on a saturated 4 GB card that holds
an editor + a 0.8B LLM + a TTS at once; one of the two frames often carries a gen2 GC on top
(BindVoice's ~1 MB managed churn — a slice-BindVoice follow-up could shave that half). On any
card with headroom the phenomenon should not exist; it is NOT reachable by engine scheduling.
Conversation-time remains clean through all rounds (worst ≤ ~97 ms singleton at a clause start,
typically ≤ 50; audio 0.00 s dry on every turn of every controlled run).

Editor-measurement caveat, learned the hard way at 14:01: play-session N>2 on one editor boot
drifts noisier (VRAM/heap residue) — 159/120 ms cruise "regressions" vanished on a fresh boot.
Compare runs ONLY across fresh boots.

## 36.2 The weights-residency pass + the pair's true residue (2026-08-03)

Round 5, prompted by the author pinning the stall to "immediately after pocket-tts finishes
streaming": SetData only STAGES a tensor — residency is paid at first dispatch REFERENCE, and the
warmup synth's first prefill tick binds dozens of flow-LM tensors at once. So
PrewarmAllocationsYielding now runs a WEIGHTS residency pass first: CopyBuffer reads one element
of every uploaded tensor into the 1-elem sync scratch, a few MB of resource per frame
(`ResidentBuffers()` on the weights registry). The pass itself measures INVISIBLE (no `w_touch`
frame above 40 ms) — correct and kept — but the pair survived it too, now cleanly attributed by
the tag columns: **~177 ms tagged `prefill`** (the warmup's 126-row block prefill, one layer per
tick) and **~242 ms untagged, prime suspect BindVoice's synchronous managed churn** (tagged
`bind` for the next run to confirm). Theories falsified so far: synth scheduling (34.2), scratch
commit timing (36.1 ×4), weights residency (this). Remaining levers, both real: slice BindVoice
into the prepare coroutine; batch the warmup's block prefill finer than one layer per tick.

Probe hygiene, hard-won: NpcTalkPerfProbe now SELF-DEDUPES (static driver guard in Awake — a
scene-armed copy plus the runner's spawn each drove the protocol, the doubled AskNPC cutting
every other reply: the "turn 1: 2.1 s" runs) and reads LastHeavyTick NON-clearing with the frame
stamp, same as FrameSpikeProbe — read-and-clear raced every other consumer and one duplicated
probe turned a whole attribution run to garbage.

## 36.3 The first-interaction pair — eight attacks, the honest state (2026-08-03 evening)

Landed and KEPT (each individually correct, none regressed; conversation clean and audio
0.00s-dry through every run): sliced CloneVoiceYielding (mono/resample/SHA one frame each) used
by the prepare, async Resources.LoadAsync warm of the tier-1 baked cache, off-thread
PreloadBakedVoiceAsync for SetVoice's cold ReadFloats, yield-wait on the reference clip's
decompress (ClipToMono's sleep-wait no longer blocks a frame), the real-mini second PrewarmKernels
pass, the weights-residency touch pass, prompt-cache dictionary in SetVoice, and the one-reply
probe protocol.

NOT fixed: the zone-entry pair (`prefill` + `bind` tags) measured 128+201, 129+195, 138+210,
224+249 across four IDENTICAL fresh-boot runs the same afternoon — run-to-run variance (±80%)
now exceeds any fix's plausible effect size, and every targeted theory (synth pacing, scratch
commit, resource creation, weights residency, driver finalize, cold asset IO, clip decompress
wait) has been individually falsified by a controlled run. Blind fixing is over.

**Armed next step — instrument, don't guess:** wrap the voice-prepare/warmup stages and the
first prefill tick in per-stage Stopwatch logs (ms per stage, printed once per session). One run
then NAMES the cost instead of tagging the frame it lands on. Only after that, fix.

## 36.4 Instrumentation names everything — the first-interaction pair, solved and residual (2026-08-03)

The #36.3 per-stage instrumentation (cpu-vs-frame per clone stage; max-tick cpu+tag per pump; a
one-shot first-prefill sub-profile; fine pf:L<i>.<sec> tags) turned five days of guessing into
three named costs in two runs:

**KILLED — the `bind` frame (~200-250 ms):** `AudioClip.GetData` on the MP3 reference DECODES on
the CPU at call time (~190 ms) — load type and loadInBackground change nothing about it. The
yielding clone now reads CHUNKED (48000 samples per frame, buffer sized EXACTLY per chunk —
GetData wraps past the clip end): measured 9.8 ms total. Every other stage was already cheap
once named (prep 4.4, sha 15, resload async 11, tail 3.4). Both reference clips also got
loadInBackground=1 (the zone-entry LoadAudioData kick used to decompress ON the main thread —
a separate ~430 ms frame at t≈12, also gone).

**HALVED — the warmup-prefill tick (~183-224 → ~120-140 ms):** ~120 ms of pure CPU inside ONE
arbitrary API call, position-stable (~16th tick, L1), operation-independent (follows the tag
around when ticks are split), first MoveNext measured at 1.5 ms total. GL.Flush() per warmup
tick (submit early, many small flushes) took it from 183 to ~110-120. Splitting the tick and
double-spacing the early ticks (driver wall-time) did NOT move the rest — a driver-internal
one-shot (deferred pipeline specialization) charged to whichever call outruns it. RESIDUAL: one
~130 ms frame per session, mid-walk-up, down from the original 2×~300 ms. Levers exhausted at
reasonable risk; on a healthier driver/GPU it should not exist.

The instrumentation STAYS (one log line per session each) — it is how any future regression
names itself on the first run. Removing it would re-buy five days of tag archaeology.

## 36.5 Coda — the JIT frame moved into the orientation window (2026-08-03)

The atomic ~130 ms driver-JIT tick would not shrink, so it moved: NPCChatBase.Awake now runs the
whole session voice-warm cycle at scene start (EnsureVoice → PrefetchNow full-rate →
PrewarmKernels — via EnsureVoice because the voice component is built lazily and there is
nothing to Find at Awake). SlowPrefetch gained an in-flight guard (never slows a faster stream),
and SlowPrefetchDivisor went 8 → 16 for the walk-up's remaining Qwen stream. Verified 17:42: TTS
resident at t≈5, JIT paid before t≈9, and the playable session's worst frame is a 92 ms editor
GC — nothing else above 74 ms anywhere. `warmed` being a session static means zone defetch/
refetch cycles never re-pay any of it.

## 36.6 The zone-entry freeze — the touch pass learns residency cycles (2026-08-03)

The one residual the author still FELT after ##36.5: a small freeze at prefetch-zone ENTRY.
His 18:33 session's frame_spikes.csv named it — `w_touch` frames of 51/40 ms at t≈11 and
another 21/21/31 ms walk at t≈16.5: the #36.2 weights-residency pass re-running IN FULL on
every zone entry AND again at StartInteraction (PrepareVoiceRoutine → PrewarmAllocationsYielding
walks the whole registry each call), on weights resident since the Awake warm cycle at t≈5.
A repeat walk is at best ~30 wasted walk-up frames and at worst a REAL re-migration bill —
the saturated card evicts under the concurrent Qwen stream, and ##36.1 round 4 already proved
a touch cannot PIN what the OS evicts afterward; first real use re-pays regardless, so the
repeat bought nothing the prepare's own layer-paced mini-synth doesn't.

Fix: once per RESIDENCY CYCLE, not once per caller. PocketTTSWeights exposes `LoadEpoch`
(bumps exactly when a Defetch actually starts — the only path that releases uploaded buffers
while the store lives), and the touch pass latches on it: skip while (still ready, same
epoch); re-run in full after any defetch→re-stream, where SetData has only STAGED the fresh
buffers and the pass is doing its designed #36.2 job again. The latch burns only on a pass
that COMPLETED under an unchanged epoch — a mid-pass defetch aborts unlatched and the next
cycle re-touches everything. The scratch preallocations below it stay unlatched (covered
buffers are same-frame no-ops; their idempotence IS the guarantee). No dials moved: no tier
row, no InferencePerf constant — the rare fix that is structural on every card.

Measured (fresh boots: 19:03 on a freshly rebooted quiet box, 18:44 on a loaded one, both
agreeing): `w_touch` now appears exactly ONCE per session — the boot warm cycle's legitimate
first pass at t≈7, all slices ≤ 23 ms — and NEVER at zone entry. The zone-entry window's
worst frame is the known #36.3 residual cluster (37-50 ms, bind/GC neighborhood, tags
shuffling per the stale-tag caveat), no new spike class; conversation clean (worst 35.5 ms);
`in-reply silence 0.00s (0.00s dry + 0.00s re-gated) in 0 bursts`. Ledger note: the 18:44
run's 0.68 s dry belonged to the BOX, not the fix — the no-pagefile machine was minutes from
RAM-exhaustion restart (the 34.2(2) class, again) and the clean-box re-run shows 0 bursts.
