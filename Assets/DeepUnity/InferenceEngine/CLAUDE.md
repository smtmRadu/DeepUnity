# DeepUnity InferenceEngine — full-GPU model inference inside Unity

The subsystem of DeepUnity that runs PRETRAINED neural models (LLM / TTS / STT) entirely on the
GPU through dedicated HLSL compute kernels — **no ONNX Runtime, no Sentis, no native plugins, no
Python at runtime**. A shipped game contains only C# + .compute shaders + exported `.bin` weights.
This file is the source of truth for the architecture, conventions, and progress. Update it as
milestones land.

## Structure (F3 move landed 2026-07-12)

```
Assets/DeepUnity/InferenceEngine/           ← F3 restructure landed 2026-07-12
  CLAUDE.md                ← this file
  import_params.py         ← THE unified exporter + model pool registry (`--list`); LLM+TTS+STT
  LLM/
    Base/  (LLM.cs, LLMConfig.cs, BPETokenizer.cs)  Qwen3_5/  Gemma3/  MiniCPM5/
  TTS/
    TTS.cs (family base)   Chatterbox/  CosyVoice/  Kokoro/  VoiceLab/  validation per module
  STT/
    STT.cs (family base)   QwenASR/  Parakeet/
Assets/DeepUnity/Main/      ModelBase.cs, TokenizerBase.cs (family roots — shared beyond InferenceEngine)
Assets/Resources/ComputeShaders/  one .compute per model family, ALL registered in Main/DeepUnityMeta.cs
Assets/Resources/Weights/weights_<model>_<size>_<quant>/   exported manifests, FLAT (gitignored)
```

Note (F3, 2026-07-12): the three families were moved from `Assets/DeepUnity/{LLM,TTS,STT}/`
into `Assets/DeepUnity/InferenceEngine/`. Namespaces stayed `DeepUnity` (folder ≠ namespace in
C#), so no code references changed — but ~77 hardcoded asset-path STRINGS (tokenizer JSONs,
validation dump dirs, KokoroG2P lexicon dir, VoiceLab presets, legacy weight fallbacks) were
rewritten to the new roots. `ModelBase` / `TokenizerBase` stay under `Main/` as engine-wide roots.

## Architecture (the rules every module follows)

1. **Class hierarchy** — `ModelBase` → abstract family (`LLM`/`TTS`/`STT`) → concrete model.
   Tokenizers: `TokenizerBase` → `TextTokenizer`/`SpeechTokenizer` → concrete (WS-F2).
2. **GPU residency lifecycle on ModelBase** ("latent loading"; dissertation feature):
   `Unloaded ⇄ Prefetching → Ready`, all transitions anti-frame-drop.
   `Prefetch() / SlowPrefetch(sec) / BoostFetch() / PausePrefetch() / Defetch(Full|Slow) /
   DefaultDefetchMode / LoadProgress / Residency`. Loader-side: live per-instance byte budget
   sampled every frame; epoch-invalidated IO so mid-load Defetch is race-free; prefetch during
   defetch is queued, never lost. Reference impl: `TTS/CosyVoice/CosyVoiceWeights.cs`.
3. **Weights**: exported by `import_params.py` (pool registry: `python import_params.py
   --list`) into per-tensor `.bin` + `manifest.tsv` (`name\tfile\tdtype\tnumel\tshape`).
   fp16 packed 2-per-uint (`readH`); int8 = per-row-scale q8 4-per-uint (`readQ8`) + `.scales`
   sibling — matmuls only; norms/embeddings/heads ALWAYS fp16. Embeddings sharded ×16.
4. **Kernels**: one `.compute` per family, kernels follow the cache-index/SetBuffer/ceil-dispatch
   idiom; every shader registered in `Main/DeepUnityMeta.cs` (lazy Resources.Load) — shared file,
   main-session edits only.
5. **Frame discipline**: everything is a coroutine yielding under budgets — weight upload
   (bytes/frame), kernel warmup (one/frame at Prewarm), inference (yield per token/layer; sync
   vs async readback samplers). GPU work interleaves across models (LLM + TTS same frame).
5b. **GPU-only runtime** (user directive 0711): models never ship a CPU inference backend —
   CPU implementations (e.g. KokoroCPU) exist strictly as validation oracles for kernel/parity
   grading. Sole exception: microsecond-scale sequential cells (tiny LSTMs in Kokoro's
   predictors, Parakeet's TDT decode loop) run CPU-side inside the GPU pipeline where dispatch
   overhead would dominate — a documented hybrid boundary per model, not a fallback.
6. **Validation**: every port ships a parity harness — Python `dump_reference.py` per-stage .npy
   (fixed seeds/noise exported as tensors when possible) → Unity probe compares corr >0.99 →
   headless batch runner (exit-code + done-marker) → ProbeLogs/ report + listen/QA probes.
7. **Licensing**: everything portable & GitHub-safe (Apache-2.0/MIT/CC0/CC-BY only; espeak-GPL
   rejected); weights folders gitignored, regenerated via import_params.py. THIRD_PARTY_NOTICES
   pending (task #12).

### Conversation compaction (LLM family)

`LLM.Compact(system_prompt, onSummary, max_summary_tokens)` — **virtual on the LLM base** (0711):
the model summarizes its own conversation in-context (greedy), then the chat re-initializes as
[system prompt + summary briefing], shrinking the KV cache while preserving what mattered.
Coroutine — runs behind gameplay between turns. Overridable per model (token-level history
splicing, keep-last-K-turns).
**Roadmap idea (dissertation-adjacent):** a small FINETUNED background compactor — a dedicated
tiny model (or LoRA of the chat model) trained specifically to compress game dialogues, running
as a low-budget background coroutine while the main model keeps talking, à la Claude Code's
background compaction. Not scheduled yet; capture data (dialogue → good-summary pairs) once the
demos are live.

### NPC-AI systems roadmap (post-CosyVoice3; shared philosophy: heavy work hides where the
### player isn't looking — same thread as SlowPrefetch/Defetch)

- **AutoCompact** — LLM watches its own token count vs maxModelLength, triggers Compact()
  between turns at a threshold. Cheap; lands with WS-F.
- **SaveMemory()/LoadMemory()** — persist the compacted summary per NPC to disk; next session's
  InitializeChat injects it as [Past conversation] → cross-session NPC memory. Cheap; WS-F.
- **SnapshotConversation()/RestoreConversation()** — full conversation state (history + KV via
  the existing disk-cache machinery) as savegame data; dialogues resume exactly.
- **Type-ahead prefill** — prefill the player's draft text incrementally WHILE they type; on
  send only the delta prefills → near-instant reply start (perceived-latency killer).
- **Background compaction v2** — compaction runs only while the NPC is idle, abortable via the
  epoch pattern if the player engages, atomic history swap on completion; later upgraded by the
  finetuned compactor above.

## Model pool (2026-07-12)

| Model | Family | Status |
|---|---|---|
| Qwen3.5 0.8B/2B | LLM | ✅ shipped — fp16/int8/int4, KV fp16/int8, benchmarked 4060+1650 |
| Gemma3 270M | LLM | ✅ shipped |
| MiniCPM5 1B | LLM | ✅ shipped |
| Chatterbox-Turbo 0.5B | TTS | ✅ shipped — parity PASS; clause-streaming; superseded by CosyVoice3/Kokoro for RT |
| Kokoro-82M v1.0 | TTS | ✅ shipped — (superseded as default by pocket-tts); 29/29 kernels corr 1.000000; int8; GPU kernel speedup (vocoder ~4×, **RTF 0.15–0.30**); 0 frame drops speaking; CPU-LSTM predictor is the only non-GPU stage (~30ms in IL2CPP) |
| CosyVoice3-0.5B | TTS | ✅ shipped — full A0→A7 + A6-MAX campaign: **streaming RTF 2.90→1.15** (int8), LM 151 tok/s, TTFA 2.49s, offline RTF 1.03, parity corr 0.9999 argmax MATCH, seams 4/4 ≤ nat. Voice cloning offline (make_voice.py). Phase 7 (batch readback → RTF<1) optional |
| **pocket-tts (Kyutai) ~100M** | TTS | ✅ **DEFAULT NPC TTS (#27)**. FlowLM (SentencePiece cond → 6L RoPE causal transformer → SimpleMLPAdaLN 1-Euler flow head, ldim32) + Mimi codec (SEANet + 2L decoder-transformer, 250-key window) → 24kHz; **voice cloning** (audio_prompt prefix). **P1–P5 bit-exact vs PyTorch (fp16 corr 1.0 / wav 0.999999)**; KV-cache decode **10× speedup**, **RTF 0.15–0.18 offline / 0.31 streaming fp16 · 0.15/0.34 int8**, TTFA ~100–335ms, streamed==offline bit-identical (0 clicks/underflows). **int8 shipped** (103 MB, −46%; per-stage ≥0.998, mel-corr 0.925 vs fp16 — user-accepted 2026-07-13). **C# SentencePiece encoder** (P7, ids exact-match) → Say(string) Python-free; TtsModel registry + default switched. Names pronounced correctly (user QA). P8 IN PROGRESS: runtime voice-clone cache (Mimi encoder → CloneVoice(AudioClip) → speaker_proj → audio_prompt, disk-cached by audio hash) |
| Qwen3-ASR 0.6B/1.7B | STT | ✅ shipped — **GPU validated 6/6 EXACT** (RTF 0.22–0.66); QwenASRSTT:STT |
| Parakeet-TDT 0.6B v2/v3 | STT | ✅ shipped — **GPU validated 6/6 EXACT** (RTF 0.08–0.09); ParakeetSTT:STT (v3 = Romanian) |

Cross-cutting DONE: ModelBase/TTS/STT bases · residency lifecycle · **F2 tokenizer hierarchy
(TokenizerBase)** · **F3 folder move under InferenceEngine/** · unified import_params pool ·
LLMRegistry (auto-extending NPC picker) · NPCChatBase (2D+3D) + LLMPool + prefetch zones + KV disk
persistence · demos ChatDemo3D / ChatDemo2D (farm) / ForestFork / VoiceLab (all E2E green).

## Task board (2026-07-12) — source of truth, update as tasks land

**Done:** A0–A7 CosyVoice3 · Kokoro port + speedup · STT Qwen-ASR + Parakeet (GPU-validated) ·
import_params registry · THIRD_PARTY_NOTICES · NPCChatBase · dissertation TeX · chat thinking-dots +
think-filter · **F2 tokenizer hierarchy** · **F3 InferenceEngine restructure** · chat text-jitter fix ·
Kokoro question-prosody probe.

**Remaining (priority order):**

| # | Task | Priority |
|---|---|---|
| ~~20~~ | ✅ **DONE** Qwen3.5 decode frame-pacing — VALIDATED headless 2026-07-13 (`LMFramePacingProbe`, 3-arm A/B, 4060 int8): pre-#20 spread+sync = a 20-30 ms hitch EVERY token; shipped burst+`SampleYielding` = **0 frames >20 ms in 9608, p95 1.65 ms, max 15.9 ms, tok/s unchanged (30.8)**. Results in BENCHMARK.md; `DebugSpreadDecode` = A/B toggle | ✅ done |
| ~~27~~ | ✅ **DONE** pocket-tts GPU port → DEFAULT NPC TTS. P1–P8 all validated: fp16 bit-exact, int8 shipped (mel-gate, user-accepted), C# SentencePiece exact, registry + default set, names QA'd, voice-clone cache (Mimi encoder → CloneVoice(AudioClip) corr 1.0; 3 tiers: Resources bake / persistent / encode; **inspector precompute button** on NPCChatBase; ref auto-capped 10 s = native prompt len). **Long-reply safe**: 2D dispatch guard (>65535 groups) + windowed streaming decode (CTX 40, O(1)/chunk, corr 1.0 vs direct). **Integration polish live-QA'd**: drain-grace pause (last words audible), OnClauseSpoken audio-synced text reveal, spread prefill + async readbacks (no Interact hitch). ChatDemo3D: Velmire → PocketTTS/jean + user-cloned voices tested | ✅ done |
| 29 | **Talk-time frame drops** — dips to ~45 FPS while an NPC speaks (pocket-tts streaming + Qwen decode concurrently). Monitored multi-turn play-mode session (walk up → several turns → watch per-frame ms + which pump/stage correlates with the dips), then optimize the culprit (suspects: chunked Mimi re-decode burst at flush, clone-prefix prefill ticks, LLM+TTS pump contention in the same frame budget) | 🔥 **NOW** |
| 24 | A6-MAX Phase 7 — CosyVoice3 sub-1.0 RTF (batch token readback) | optional |
| 25 | Paper benchmark matrix (RTF/TTFA/load/VRAM, all engines × quant) on 4060 | unblocked |
| 15 | WS-F — LLM↔ModelBase unification (needs the 3 LLM loaders rewritten to the epoch/BeginLoad/Defetch contract — RISKY, touches the validated load path) | deferred |
| 28 | **Background KV compaction + HistoryMode.ResumeFromCompact** — run compaction in the background (idle/zone-exit), store the compacted summary+recent-turns state, resume from it on reopen. Activates the reserved `ResumeFromCompact` enum (replaced the removed `KeepAliveInBackground` — residency is the prefetch zone's job alone; LLM.Compact's summary-briefing seed is the starting point) | 🔥 **next after demo-polish commit** |
| 18 | WS-G rest — NPC-AI primitives (per-NPC memory, conversation snapshots, type-ahead prefill) | backlog |

(Full per-task history + gotchas live in the session memory `project_deepunity_cosyvoice3_port.md`.)
