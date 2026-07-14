# NPC infrastructure — architecture + tweak recipes

Operational map of the NPC dialogue stack (`Assets/DeepUnity/NPC/`) and the exact steps for the
common modifications. Written so ANY model/agent can execute a tweak without re-deriving the
design: each recipe lists the files to touch and the invariants that must hold. The task
board / session gotchas live in `Assets/DeepUnity/InferenceEngine/CLAUDE.md` — this file is the
stable how-it-works + how-to-change-it reference.

## 1. System map

```
                         NPCChatBase (abstract MonoBehaviour, NPC/NPCChatBase.cs)
                         ├─ NPCInteractor3D (Tutorials/ChatDemo3D)   presentation subclasses:
                         └─ NPCInteractor2D (Tutorials/ChatDemo2D)   camera, window type, triggers, anim
   serialized per NPC:   npc_name · system_prompt · approach_text
                         conversationMode {LlmOnly, LlmPlusTts}
                         historyMode {ResetEveryTime, ContinueWhereLeftOff, ResumeFromCompact}
                         cacheKVCache · compactionTriggerTokens (ResumeFromCompact only)
                         model (string id → LLMRegistry dropdown) · quantization · allowThinking
                         smoothVsSpeed ("Reply Pacing" Smooth⇄Speed dial, per NPC)
                         sampling fields (-1 = model Config preset)
                         ttsModel {PocketTTS(default), Kokoro, CosyVoice3, Chatterbox} · ttsVoice ·
                         voicePitch · ttsQuantization · clonedVoiceClip (PocketTTS clone)
                         usePrefetchZone · prefetchRadius · slowPrefetchSeconds

   LLM side                                        TTS side
   ─────────                                       ────────
   LLMRegistry  [LLMEntry] reflection catalog      PocketTTSVoice / KokoroVoice /
   LLMPool      refcounted shared instances        CosyVoiceVoice / ChatterboxVoice
                keyed (id, quant, kvQuant)         (added to the NPC GameObject on demand;
   LLM (base)   Chat/Generate/InitializeChat/      FeedText per token, FlushText at reply end)
                Compact/Save+RestoreConversationKV
   concrete:    Qwen3_5ForCausalLM (flagship: disk-KV, thinking, coalesced kernels),
                MiniCPM5, Gemma3-270M

   cross-cutting: FramePacing (LLM⇄TTS frame arbiter) · InferencePerf (tuning board + #32
   AutoTune; fed by the NPC's Smooth⇄Speed dial) · DiskKVCache files in persistentDataPath
```

Chat windows implement `INPCChatWindow` (SoulsChatWindow 3D, ChatWindow2D). Several NPCs share
one window; only the NPC in interaction reacts.

## 2. Lifecycle (what calls what)

1. **Approach** — prefetch zone entry (distance check in `Update`) or trigger contact
   (`OnPlayerContact`): TTS `SlowPrefetchNow`/`PrefetchNow` + `EnsureLlm()` →
   `LLMPool.Acquire` (shared instance; second NPC on the same model = free) + slow-budget
   weight streaming paced by `slowPrefetchSeconds`.
2. **Open** — `StartInteraction` → `OpenConversation()`: WAITS out any in-flight background
   compaction on this model (never cancels — the bubble pulses "Compacting…", input blocked),
   then per `historyMode`: tier (a) live-KV reuse (owner check via
   `LLMPool.OwnsConversation`), tier (b) disk restore `TryRestoreConversationKV` (Qwen only),
   tier (c) re-prefill `BuildResumePrompt()` (= system + HISTORY block + turns since compact).
3. **Talk** — `AskNPC` → `Talk` coroutine → `llm.Chat(..., enable_thinking: allowThinking)`;
   tokens stream to the window (think-tokens filtered by `SplitThink`, never voiced) and to
   the voice via `FeedVoiceText`. Decode pacing: `InferencePerf` AutoTune decides sync/async
   per session; `FramePacing` arbitrates frames against the speaking voice.
4. **Close** — `CloseConversation(interrupted)`: ResetEveryTime wipes now; continue modes save
   the whole conversation KV to disk in the background (`SaveConversationKvRoutine`,
   `kvSaveInFlight` gate); ResumeFromCompact additionally runs `CompactConversationRoutine`
   past `compactionTriggerTokens` (single-shot compact → `[system + HISTORY:]` re-seed, #28/#31).
5. **Zone exit (Idle only)** — `ReleaseLlmAfterKvSave()`: waits for any in-flight compaction
   AND the KV save (the model never leaves the GPU until the compact lands), then
   `LLMPool.Release` (frees only at refcount 0) + TTS `DefetchNow`.

## 3. Recipes

### R1 — Integrate a fine-tuned LLM (e.g. your own Qwen3.5-0.8B SFT)

Same-architecture fine-tunes are a 3-step drop-in; NO engine code changes beyond one entry:

1. **Export** the HF checkpoint to engine weights (WSL or Windows, needs the checkpoint dir):
   ```bash
   python Assets/DeepUnity/InferenceEngine/import_params.py \
          D:/checkpoints/my-finetuned-qwen --quant int8 \
          --out Assets/Resources/Weights/weights_qwen3.5_0.8B_mysft_int8
   ```
   Arch is auto-detected (qwen3_5 / gemma3 / minicpm5; size by hidden dim). Prefer int8
   (fp16-equivalent quality, half VRAM). The folder MUST live under `Assets/Resources/Weights/`
   (players resolve it through StreamingAssets via `DeepUnityMeta.ResolvePath`).
2. **Register** it — add ONE static method anywhere in the engine assembly (convention: next to
   `Qwen3_5ForCausalLM`, see the existing entry at `Qwen3_5.cs` ~line 80 as the template):
   ```csharp
   [LLMEntry(5)]   // dropdown order; ties sort alphabetically
   static LLMRegistry.Entry MySftEntry() => new LLMRegistry.Entry
   {
       id = "Qwen3.5-0.8B-mysft",   // the STABLE string scenes serialize — never rename casually
       create = (q, kv) => new Qwen3_5ForCausalLM(quantization: q, kv_quant: kv,
                    params_path: "Assets/Resources/Weights/weights_qwen3.5_0.8B_mysft_" +
                                 (q == LLMQuant.INT8 ? "int8" : q == LLMQuant.INT4 ? "int4" : "fp16")),
       prewarm = () => Prewarm(),
   };
   ```
3. **Select** it — the id now appears automatically in every NPC's `Model` dropdown
   (`NPCChatBaseEditor.DrawModelPopup` reads `LLMRegistry.Ids`). Pick it, done: pooling,
   disk-KV persistence, thinking, prefetch zones, compaction all work unchanged because they
   key off the LLM base API + the (id, quant, kvQuant) tuple.

Notes: a fine-tune with a DIFFERENT chat template needs its template mirrored in the model's
`Chat()` (see MiniCPM5's `<think>` wiring for the pattern). A genuinely new ARCHITECTURE is a
full port — copy the closest model folder (Qwen3_5 for GQA/hybrid, MiniCPM5 for vanilla llama)
and its compute shader, then parity-gate against reference dumps (see the port history in
`InferenceEngine/CLAUDE.md`).

### R2 — Add a new NPC to a scene

Copy an existing NPC block in the scene's BUILDER (builders are the source of truth — scenes
are rebuilt from them; hand-edits to .unity files get overwritten). 3D: `ChatDemo3DBuilder`
(`BuildWitchNpc` is the template), 2D: `ChatDemo2DBuilder.BuildNpc(...)`. Set: name, prompt,
conversationMode, historyMode, model id, ttsModel+ttsVoice (or clonedVoiceClip), prefetch zone
radius. Rebuild via the `DeepUnity/...Build...` menu.

### R3 — Voices

- **Pick a baked voice**: the `Tts Voice` dropdown lists what exists on disk under the engine's
  `Assets/Resources/Weights/weights_<engine>*/voices/` (pocket/CosyVoice3 = tensor dirs,
  Kokoro = `.bin` voicepacks). The string IS the on-disk key — no alias mapping, ever.
- **Clone a voice (PocketTTS only)**: dropdown → "Clone (reference clip)" → assign an
  AudioClip (import Load Type: *Decompress On Load*). Clip is pause-aware-cropped near 10 s
  (never mid-word). Press *Precompute voice-clone cache* → writes
  `Assets/Resources/Cache/<sha256>.bytes` (ships in builds; runtime = pure load). A non-null
  clip always overrides the baked name.
- **New baked Kokoro voice**: blend/bake with `import_kokoro.py` (voicepacks land in both
  weight folders); pocket/CosyVoice3 baked voices come from their `make_voice.py`/exporter.

### R4 — Conversation memory behavior

- `ResetEveryTime` — wiped the moment the chat closes.
- `ContinueWhereLeftOff` — fully persistent: live KV while resident → disk KV
  (`persistentDataPath/DeepUnity/qwen35_conv_<npc>_<hash>.kv`, Qwen only) → transcript
  re-prefill fallback.
- `ResumeFromCompact` — Continue + background compaction after close once the estimated
  history exceeds `compactionTriggerTokens` (~4 chars/token): the model answers a bare
  "Compact the conversation." with a one-shot compact, the chat recomputes as
  `[system + HISTORY: compact]` and the compacted KV is saved. NEVER cancelled once started:
  reopening waits behind a "Compacting…" pulse (input blocked) and zone exit defers the GPU
  release until the compact + its KV snapshot land. Tune `compactionTriggerTokens` to the model's
  KV budget (default 512; Qwen capacity 8192).

### R5 — Performance: which knob for which symptom

All cross-engine knobs live in `InferencePerf.cs` (documented statics); the intent is NOBODY
hand-tunes per GPU — measure-driven AutoTune (#32) decides per session:

- **"Replies stutter the framerate" / "text too slow"** → the ONE user dial:
  `smoothVsSpeed` — the "Reply Pacing" Smooth ⇄ Speed slider ON THE NPC (applies while
  talking to that NPC). The auto-detection always computes for a stable 60+ fps; the slider
  only biases around it (hard ends force the implementation limits: async + 1 layer/frame
  vs sync + bulk prefill). Moving it mid-dialogue re-probes on the next reply. Do NOT
  hardcode `LlmDecodeTokensPerFrame`/sync mode — AutoTune owns them.
- **"Hitch when the dialog opens"** → prefill pacing is automatic (adaptive pack, 60 fps
  anchor; the Smooth ⇄ Speed ends force 1 layer/frame vs bulk). If opens still feel slow,
  shorten the system prompt (prefill cost is linear in it).
- **"Voice underruns / word-dribble"** → nothing to tune by hand: `PocketTTSVoice` escalates
  prebuffer/chunk itself, persists PER-GPU (v3 keys), and walks back one rung after each clean
  session — a contended session can't permanently degrade a device.
- **"Loading hitches on approach"** → `slowPrefetchSeconds` / `prefetchRadius` (give the
  stream more walk-up time), `LLM.UploadBudgetBytes` only for global load-rate policy.

### R6 — Scripted events (quests, gifts, reactions)

- `AskNPCSilent(prompt)` — inject a hidden bracketed prompt (no visible player line, still in
  history). Example: the 2D GIVE button sends `[The player hands you: 3 carrots... pay 6
  coins]`.
- `protected virtual OnReplyFinished()` — fires when a reply completes (never on interrupt);
  the 2D demo pays the coins here. Subclass hook — no base changes needed for new events.

### R7 — New chat window / new demo

Implement `INPCChatWindow` (see ChatWindow2D for the minimal shape: Open/Close, AddMessage,
PopLast streaming mutation, SetSendLoading, InputField). Wire it in the presentation subclass.
Builders create everything from code — follow ChatDemo2DBuilder's UI section.

## 4. Invariants (do NOT break)

1. Builders are the source of truth for demo scenes — edit the builder, rebuild, never the
   .unity by hand.
2. `model` ids are serialized in scenes — renaming an `[LLMEntry]` id orphans existing scenes
   (they warn + fall back to the first entry). Add new ids; deprecate old ones deliberately.
3. One weight-quant mode per session per shader (keyword on the shared ComputeShader).
4. Pooled LLMs are SHARED: never assume exclusive KV ownership — always go through
   `LLMPool.ClaimConversation/OwnsConversation` (the tier-a check).
5. Voice-clone clips must be readable (*Decompress On Load*) and the clone cache key is the
   SHA of the cropped wav — changing the crop algorithm requires re-baking all caches
   (menu: `DeepUnity/PocketTTS/Bake Voice-Clone Cache (all Voices clips)`).
6. Think-tokens never reach the TTS or the transcript; they render only behind the window's
   `showThinkingTokens` debug toggle.
7. Weight folders live under `Assets/Resources/Weights/` (gitignored, re-exportable via
   `import_params.py`); caches under `Assets/Resources/Cache/`; runtime KV snapshots under
   `persistentDataPath/DeepUnity/`.
