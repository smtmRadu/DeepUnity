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
