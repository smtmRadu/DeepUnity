# DeepUnity InferenceEngine — full-GPU model inference inside Unity

The subsystem of DeepUnity that runs PRETRAINED neural models (LLM / TTS / STT) entirely on the
GPU through dedicated HLSL compute kernels — **no ONNX Runtime, no Sentis, no native plugins, no
Python at runtime**. A shipped game contains only C# + .compute shaders + exported `.bin` weights.
This file is the source of truth for the architecture, conventions, and progress. Update it as
milestones land.

## Target structure (physical move happens at the WS-F merge point — see Progress)

```
InferenceEngine/
  CLAUDE.md                ← this file
  ModelBase.cs             (today at Main/ModelBase.cs; moves here)
  LLM/                     (today at Assets/DeepUnity/LLM/)
    Base/  (LLM.cs, LLMConfig.cs, BPETokenizer.cs)  Qwen3_5/  Gemma3/  MiniCPM5/
    import_params.py       ← THE exporter + model pool registry (`--list`)
  TTS/                     (today at Assets/DeepUnity/TTS/)
    TTS.cs (family base)   Chatterbox/  CosyVoice/  Kokoro/  validation per module
  STT/                     (today at Assets/DeepUnity/STT/)
    STT.cs (family base)   QwenASR/  Parakeet/
Resources/ComputeShaders/  one .compute per model family, ALL registered in Main/DeepUnityMeta.cs
Resources/DeepUnity/{LLM,TTS,STT}/<Family>/weights_<model>_<quant>/   exported manifests (gitignored)
```

## Architecture (the rules every module follows)

1. **Class hierarchy** — `ModelBase` → abstract family (`LLM`/`TTS`/`STT`) → concrete model.
   Tokenizers: `TokenizerBase` → `TextTokenizer`/`SpeechTokenizer` → concrete (WS-F2).
2. **GPU residency lifecycle on ModelBase** ("latent loading"; dissertation feature):
   `Unloaded ⇄ Prefetching → Ready`, all transitions anti-frame-drop.
   `Prefetch() / SlowPrefetch(sec) / BoostFetch() / PausePrefetch() / Defetch(Full|Slow) /
   DefaultDefetchMode / LoadProgress / Residency`. Loader-side: live per-instance byte budget
   sampled every frame; epoch-invalidated IO so mid-load Defetch is race-free; prefetch during
   defetch is queued, never lost. Reference impl: `TTS/CosyVoice/CosyVoiceWeights.cs`.
3. **Weights**: exported by `LLM/import_params.py` (pool registry: `python import_params.py
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

## Model pool & progress (2026-07-11)

| Model | Family | Status |
|---|---|---|
| Qwen3.5 0.8B/2B | LLM | ✅ shipped (fp16/int8/int4, KV fp16/int8, benchmarked 4060+1650) |
| Gemma3 270M | LLM | ✅ shipped |
| MiniCPM5 1B | LLM | ✅ shipped |
| Chatterbox-Turbo | TTS | ✅ shipped (parity PASS 0307; clause-streaming retrofit; RTF 1.42 → superseded by CosyVoice3 for real-time) |
| **CosyVoice3-0.5B** | TTS | 🔄 **WS-A (main focus)**: A0 ✅ · **A1 ✅ PASS** (CausalHiFT: all stages corr 1.000000, wav 0.999989, RTF ~0.34@4060; gotcha: nearest-upsample scratch = outLen×IN-ch, 2× the ConvT donor max — OOB writes drop silently) · **A2 ✅ PASS 0711** (DiT flow: h_lookahead/dxdt-s0/mel ALL corr 1.000000, flow-mel→HiFT wav 0.99975; offline=ONE full-attention pass; kernels added: Conv1DGrouped/AdaLNModulate/GateAdd/RopeQK/EulerCfgStep/PackBroadcastCh + gelu-tanh act 8; **RoPE = x_transformers INTERLEAVED pairs (2j,2j+1), flat pre-head-split → only head 0 rotates**; naive-kernel flow = 18.7s @ T=576, perf work at A5/A6). **A3 ✅ PASS 0711** (LM: all 24 layers corr 1.000000, logp-s0 0.999998 argmax MATCH, decode 49.8 tok/s = 2.0× RT; CosyVoiceLMCS = Qwen3_5CS copy w/ plain-gamma RMSNorm + biased QKV; gotcha: buffer-name mismatch in SetBuffer = silent [Error] not exception). **A4 ✅** (tokenizer EXACT; e2e 10.9s audio; SineGen2 staircase source) · **A5 ✅ 0711** (ref-exact chunk schedule 25→50→100 + attn_chunk mask + finalize trims + CosyVoiceVoice ring buffer; perf: coalesced tiled GEMM + cooperative AdaLN + handle caching = flow 23s→7.0s @T=576; streaming TTFA 4.65s RTF 2.86 — int8+batch-CFG close the RT gap at A6) · **A7 ✅ 0711** (voices/velmire baked via make_voice.py from a Kokoro-am_onyx EN prompt; NPCInteractor3D TtsEngine enum; ChatDemo3D scene rebuilt: Velmire speaks CosyVoice3/velmire, play-QA pending). **A6 int8 ✅ 0711** (996MB vs 1.9GB; LM logp 0.999867 / flow mel 0.9996 vs fp32 — int8 wav-corr vs ref is phase-drift-decorrelated, MEL is the int8 gate; demo CosyVoiceVoice defaults to int8; gotcha: append `^voices/` manifest lines when cloning a weights dir). **PERF FINDING**: int8 = NO speed win on 4060 (LM decode is dispatch-overhead-bound ~240/token → 51.6 tok/s; tiled-GEMM flow is compute-bound → 6.9s @T=576 both quants; streaming RTF 2.90≈2.86). Remaining A6 = structural: batch-2 CFG single pass · AdaLN mods are t-only → precompute 22×10 once/synthesis · incremental chunk re-solve (freeze old DiT K/V, HiFT left-context re-vocode) · LM dispatch fusion. Probes: `CosyVoice{Hift,Flow,Lm,E2e,Stream}Probe.Run`, `CosyVoiceE2eProbe.RunVelmire`, `{Lm,Flow,Stream}Probe.RunInt8` |
| Kokoro-82M v1.0 | TTS | ✅ **GPU port VALIDATED 0711** (26/26 kernels PASS, stages corr 1.000000, RTF 0.24-0.43 @4060, CPU-LSTM hybrid per rule 5b) · **production polish 0711**: int8 (78 q8 matmuls via per-tensor LinearBiasQ8/.w.scales, 143MB, err≤0.011) · frame-spread decode (KokoroModel.SliceMacs Tick budget + AdainBlockY/SnakeResBlockY; KokoroCPU ParOpts=cores/2) · KokoroVoice FeedText/FlushText + PrewarmKernels + SlowPrefetchNow/DefetchNow · **ChatDemo3D Velmire = velmire_elder blend** (12 packs baked; blends = staging .pt avg + reimport) · NPCInteractor3D prefetch ZONE (10m gizmo sphere, enter=slow-prefetch qwen+kokoro, exit=deload, toggleable) · perf harness QwenKokoroPerfProbe (bridge Run/Finish/Restore): speak-alone 2/5544 frames >33ms; remaining spikes = QWEN decode (38/1267 >33ms → async token readback, task #20) |
| Qwen3-ASR 0.6B/1.7B | STT | ✅ WS-D D1 code-complete (0711): CPU twin corr 1.000000 all gates both sizes, transcripts EXACT; QwenASRSTT:STT + 26-kernel shader (fxc-checked, int8-ready); Unity compile clean; PENDING: in-Unity GPU probe + latency + int8 |
| Parakeet-TDT 0.6B v2/v3 | STT | ✅ WS-E E1 code-complete (0711): corr 1.000000 every stage, tokens/transcripts EXACT both variants (v3 incl. Romanian = default); ParakeetSTT:STT + 9-kernel shader; CPU TDT decode loop; Unity compile clean; PENDING: in-Unity GPU probe + latency |

Cross-cutting: ModelBase+TTS+STT bases ✅ (0711) · residency lifecycle in CosyVoiceWeights ✅ ·
legacy rebase (LLMs, Chatterbox → ModelBase; per-instance budgets; BeginLoad/Defetch on the 3 LLM
loaders) = **WS-F, right after CosyVoice3** · tokenizer hierarchy = WS-F2 · physical folder move
under InferenceEngine/ = WS-F3 (agents are writing to the old absolute paths mid-flight; move
only at the merge point, Unity closed, .meta pairs together) · unified import_params pool ✅
(kokoro/STT exporters fold in at their merges).

Related demos: ChatDemo3D (witch NPC = Chatterbox → CosyVoice at A7) · ChatDemo2D being rebuilt
as a Stardew-like farm demo (WS-C agent; old demo archived as ChatDemo2D_OLD).
