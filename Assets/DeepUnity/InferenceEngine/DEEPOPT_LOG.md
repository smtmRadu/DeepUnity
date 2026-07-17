# Kernel deep-opt campaign log — 2026-07-17 (Victus RTX 4060 Laptop)

The running record of the multi-model kernel-optimization campaign: what was attempted, what was
**measured** (only headless-validated numbers land here), and where each workstream stands. Per-model
design documents (written by the implementing agents) hold the kernel-level detail:

- `TTS/PocketTTS/DEEPOPT.md` — pocket-tts rounds 1–3 design + validation runbook
- `LLM/Qwen3_5/DEEPOPT2.md` — Qwen3.5 post-#31 audit (implementation in flight)
- `TTS/CosyVoice/DEEPOPT.md` — CosyVoice3 audit + plan (**NOT implemented, shelved by decision**)
- `LLM/OPTIMIZATIONS.md` — the historical LLM log this campaign builds on (#31 = coalesced GEMV/GEMM)

**Method (same for every workstream):** one Fable agent per model, write-territory limited to the
model's own shader + module folder (structural guarantee that no other model can regress); kernels
strictly additive behind static switch tiers with the legacy path kept as fallback; a parity gate for
every new kernel + full-path E2E gates (same-EOS/argmax, latent/logit corr, mel-corr) + same-run A/B
`[perf]` timing; all Unity validation run serialized by the main session, headless (`-batchmode
-executeMethod`, Unity closed), chain scripts in `ProbeLogs/run_pocket_deepopt2.sh` /
`run_pocket_r2.sh`. Agents never launch Unity.

---

## pocket-tts (Kyutai, DEFAULT chat TTS) — #31-P rounds

Offline-KV RTF probe, 66 ids → ~10.2–10.4 s of 24 kHz audio, warm shaders. Baseline = 2026-07-16
BENCHMARK.md entries (pre-campaign).

| metric (int8) | baseline | R1 coal+flow-fuse | **R2 GPU-resident** | R3 overlap+ramp |
|---|---:|---:|---:|---:|
| **RTF** (10.4–10.6 s clip) | 0.108 | 0.103 | **0.073** | 0.076–0.078† |
| **short-clip total** (46 f ≈ chat clause, same-run A/B) | — | — | 1× ref | **2.05×** |
| AR loop ms | 801 | 783 | **495** | ~640† |
| prefill ms | 82 | 52 | 52 | 52 |
| mimi decode ms | 236 | 230 | 189 | ~125† (in-loop windows) |
| TTFA proxy ms | 88 | 57 | 79* | **47–60** |

| metric (fp16) | baseline | R1 | **R2** | R3 |
|---|---:|---:|---:|---:|
| **RTF** | 0.132 | 0.114 | 0.099 | **0.095** |
| short-clip total A/B | — | — | 1× ref | **1.75×** |
| AR loop ms | 963 | 920 | **756** | ~990† |
| TTFA proxy ms | 156 | 62 | 94* | **65** |

\* offline proxy became block-granular in R2 (K=8 frames per readback block) — informational only;
streaming owns real TTFA. Fixed by the R3 ramp. † under `OverlapMimi` the loop/mimi split shifts
(mimi windows are issued inside the AR loop) — only total/RTF is cross-round comparable.

Per-frame anatomy (from the R2 probe's `[instrument]` mode, 46-frame run):

| | legacy | R1 | **R2** |
|---|---:|---:|---:|
| dispatches / frame | ~151 | 104.0 | **51.1** |
| blocking readbacks / frame | 2.02 | 2.02 | **0.13** |
| uploads / frame | 3.02 | 3.02 | **0.13** |
| CPU work in loop | input_linear + eos dot | same | **none** |

- **R1 (`FastKernels2`)** — #31 coalesced GEMV/GEMM ported (8 new kernels: `LinearBias(Q8)Coal/Gemm`,
  `FlowResBlockFused(Q8)`, `FlowFinalFused(Q8)`); flow head 66→11 dispatches; constant time-embed cached.
  Finding: the AR loop was **not** bandwidth-bound (ran at ~12–15 GB/s vs 236 GB/s roofline) — coalescing
  alone moved almost nothing (AR 1.02×/1.26×); the wins were prefill (1.9×/1.6×) and TTFA. Parity: all
  gates PASS, bit-exact #30 regression, E2E same EOS, mel-corr ≥0.999998.
- **R2 (`FastKernels3`)** — instrumentation first, which proved the real bottleneck: **2 blocking
  readbacks + 3 uploads per frame** (EOS check + latent add on CPU) = ~88% of the loop was pipeline-drain
  wait. Fix: fully GPU-resident AR frame — eos/latent written to per-frame GPU slots, K=8 frames issued
  back-to-back, ONE readback per block, noise uploaded per block, latent feedback on GPU (`ARCommit`);
  plus LN folded into GEMV staging (`GemvLN16/Q8`), residual/assemble epilogues (`Gemv16/Q8` modes),
  QKV prep collapsed (`ARQkvPrep`, RoPE bit-identical). A/B: AR 1.41× fp16 / 1.45× int8 over R1. Parity:
  all gates PASS (eos |Δ|≤4.3e-6, latents corr ≥0.9999998, mel-corr ≥0.999985).
- **R3 (`OverlapMimi` + `ArBatchRamp {2,4}` + `StreamFirstChunkFrames=2`)** — mimi windows issued
  INSIDE the AR loop between frame blocks (scheduling only, zero new kernels; wav proven **bit-exact**
  vs the sequential path at T≤64 and T=160, latents bit-unperturbed) + first-block K ramp. Measured:
  **chat-clause-length runs (≈3.7 s) 2.05× int8 / 1.75× fp16 total; TTFA proxy →47–60 ms**; long runs
  neutral (0.99× — the AR chain leaves no usable GPU bubbles at this size; overlap just relocates the
  work). Streaming twin: first flush at 2 frames → first audio ~0.5 s earlier in the demos. Lever 3
  (whole-layer persistent fusion) **assessed and skipped with proof**: D3D11 has no device-scope
  barrier, so the fused layer collapses to one threadgroup ≈ 3–6 ms/layer — slower than the 6-dispatch
  chain (DEEPOPT §R3.3).
- Rollback ladder: `OverlapMimi=false`/`ArBatchRamp=null` → exact R2; `FastKernels3=false` → R1;
  `FastKernels2=false` → pre-campaign.
- **CLOSED (2026-07-17, user decision): pocket-tts optimization stops here.** Where it landed for the
  chat workload: clause-length synth ~2× faster than R2's structure, TTFA(proxy) 47–60 ms (was 88–156),
  10-s RTF int8 0.076 (was 0.108), all quality gates green across three rounds. Remaining play-mode QA:
  listen to `ProbeLogs/pockettts_names_*.wav` + a live NPC talk session. Next investment: Qwen.

## Qwen3.5 (flagship LLM) — post-#31 round (in flight)

Audit-first mandate: per-token dispatch/cost model vs the 236 GB/s roofline before any code. Status:
`DEEPOPT2.md` audit written + kernels/probe started (`QwenDeepOpt2ParityProbe.cs`), agent paused by a
session limit mid-implementation — WIP snapshot preserved, resumes after pocket R3 validation.
Scope: dispatch fusion (norm/RoPE/residual into GEMV), attention KV micro-chain, LmHead (~151k vocab
GEMV + GPU-side sampling readback), all behind a new switch tier with #31-style parity gates.
Rationale: LLM decode speed is what the user feels in chat (clause arrival rate + frame pacing while
the NPC talks).

## CosyVoice3 — audit only, SHELVED (user decision 2026-07-17)

Agent audit corrected the premise: the LM decode already runs A6-max fused split-k GEMVs (≥#31-grade);
the genuinely legacy surfaces are **HiFT convs (naive kernel at ~0.7% of fp32 peak → 5–8× available,
3131→~400–600 ms)**, the flow DiT micro-dispatch chain (1.4–1.8×), and LM prefill (2.5–3×). Projected
e2e: offline RTF 1.03→~0.50–0.60, streaming 1.31→~0.75–0.95 (real-time plausible), TTFA 2.48→~1.4–1.7 s.
Even fully optimized it stays ~5–10× costlier per audio-second than pocket-tts (and 2.1 GB vs 209 MB),
so it's parked; its unique value (multilingual) wasn't the current priority. Full plan in
`TTS/CosyVoice/DEEPOPT.md` — executable cold by a future session.

## Context: where the TTS tiers now sit (4060, offline RTF)

| model | RTF | note |
|---|---:|---|
| Kokoro-82M int8 (FK3) | 0.041 | fastest tier; no cloning, non-AR (TTFA ∝ chunk) |
| **pocket-tts int8 (R3 defaults)** | **0.076** | DEFAULT: AR streaming, voice cloning, **TTFA(proxy) 47–60 ms**, chat clauses ~2× vs R2 |
| pocket-tts fp16 (R3 defaults) | 0.095 | |
| CosyVoice3 fp16 | ~1.03 | shelved; multilingual; plan would reach ~0.5–0.6 |

*(BENCHMARK.md gets the final refreshed tables once R3 validates; this log is the campaign narrative.)*
