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
                         backendTradeoff ("Backend Tradeoff" dial, 5 levels → BackendTradeoffTable;
                           2nd field of the CONVERSATION group, right under chatWindow)
                         conversationMode {LlmOnly, LlmPlusTts}
                         historyMode {ResetEveryTime, ContinueWhereLeftOff, ResumeFromCompact}
                         cacheKVCache · maxContextLength (sizes the KV; halt/compact threshold)
                         model (string id → LLMRegistry dropdown) · quantization · allowThinking
                         sampling fields (-1 = model Config preset)
                         ttsModel {PocketTTS(default), Kokoro, CosyVoice3, Chatterbox} · ttsVoice ·
                         voicePitch · ttsQuantization · clonedVoiceClip (PocketTTS clone)
                         usePrefetchZone · prefetchRadius · slowPrefetchSeconds
                         enableAskUserQuestion · enableGiveItem (the TWO interactive tools)
                         decisions (List<NPCDecision> — the decision-binding table, see R6b)

   LLM side                                        TTS side
   ─────────                                       ────────
   LLMRegistry  [LLMEntry] reflection catalog      PocketTTSVoice / KokoroVoice /
   LLMPool      refcounted shared instances        CosyVoiceVoice / ChatterboxVoice
                keyed (id, quant, kvQuant)         (added to the NPC GameObject on demand;
   LLM (base)   Chat/Generate/InitializeChat/      FeedText per token, FlushText at reply end)
                Compact/Save+RestoreConversationKV
   concrete:    Qwen3_5ForCausalLM (flagship: disk-KV, thinking, coalesced kernels),
                MiniCPM5, Gemma3-270M

   cross-cutting: BackendTradeoffTable (every per-frame budget that is a statement about the
   machine — LLM fetch/prefill/decode AND the voice's ticks/prebuffer/chunk/cede-headroom/tick-MACs)
   · FramePacing (LLM⇄TTS frame arbiter) · InferencePerf (what is left: arbitration rates and
   shapes) · DiskKVCache files in persistentDataPath
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
4. **Context limit (ResumeFromCompact)** — THE COMPACTION STANDARD: when a reply pushes the
   conversation past `maxContextLength`, that reply is still delivered IN FULL — decoded,
   typed and SPOKEN to the end (+8192 KV headroom absorbs the overshoot). Only after the
   voice goes quiet does `CompactConversationRoutine` run behind the "Compacting…" pulse
   (input blocked; single-shot compact → `[system + HISTORY:]` re-seed). The window KEEPS the
   whole visible conversation; the NEXT open collapses it to one dimmed compact block.
   Sending or leaving during the pre-compact speech wait stands the pass down (a new ask
   re-triggers after ITS reply; a close leaves it to the next open's crash-recovery).
   Interrupts (send/leave mid-reply) cancel generation cooperatively at a token boundary
   (`LLM.CancelChat`) — the KV keeps the truncated turn as if the model had stopped there.
5. **Close** — `CloseConversation(interrupted)`: ResetEveryTime wipes now; continue modes save
   the whole conversation KV to disk in the background (`SaveConversationKvRoutine`,
   `kvSavesInFlight` per-LLM gate; skipped while a compaction is forwarding the model — it re-saves
   itself when it lands).
6. **Zone exit (Idle only)** — `ReleaseLlmAfterKvSave()`: waits for any in-flight compaction
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
- `ResumeFromCompact` — Continue + self-compaction at the context limit. THE STANDARD
  (user spec 2026-07-15):
  1. The reply that hits `maxContextLength` is delivered IN FULL — decoded, typed and spoken
     to the end (the KV is allocated `maxContextLength + 8192`, so the overshoot fits).
  2. "Compacting…" appears ONLY once the NPC finished talking; input stays blocked until the
     compact lands. The model answers a bare "Compact the conversation." with a one-shot
     compact, the chat recomputes as `[system + HISTORY: compact]` and the compacted KV is
     saved (overwrites the pre-compact snapshot).
  3. The window keeps the ENTIRE visible conversation through the compaction; the next open
     starts visually EMPTY (+ any turns since the compact) — the compact itself is never
     rendered, it lives only in the system prompt's HISTORY block.
  4. NEVER cancelled once started: reopening waits behind the pulse and zone exit defers the
     GPU release until the compact + its KV snapshot land. If it never got to run (player
     left during the speech wait / game closed), the next open compacts behind the same pulse.
  Verified end-to-end by `NpcCompactProbe` (menu `DeepUnity/NPC/Run Compact Probe`); the
  interrupt/cancel machinery by `NpcInterruptProbe` (`Run Interrupt Probe`).

### R5 — Performance: which knob for which symptom

EVERY per-frame budget that is a statement about the MACHINE is one of the five fixed rows of
`BackendTradeoff.cs` (`BackendTradeoffTable`) — the LLM's fetch/prefill/decode AND the voice's
heavy ticks per frame (speaking + refilling), prebuffer seconds, decode chunk frames, cede
headroom and MACs per tick. What is left in `InferencePerf.cs` is only the arbitration's rates and
shapes (cede stride, refill floor, budget scale, readback spin). NOTHING self-tunes any more: the
two controllers behind the old `smoothVsSpeed` slider went 2026-07-26, and the voice's three
adaptive loops — the PlayerPrefs-persisted prebuffer/chunk escalation ladder, the tick-cost
calibrator, the refill-rate EMA — went 2026-07-27 (BackendTradeoff.cs documents why at length):

- **"Replies stutter the framerate" / "text too slow"** → the ONE user dial: **Backend Tradeoff**
  (`backendTradeoff` ON THE NPC, second field of the CONVERSATION group, five levels Very
  Smooth…Very Fast; applies while talking to that NPC, and the level in force is logged in the LLM
  boot summary). One pick sets fetch bytes, prefill steps, decode tokens AND the whole of the
  voice's pacing. Moving it mid-dialogue lands on the next frame (the two fields pushed onto the
  voice component are written in `EnsureVoice`, so those follow at the next scene load). Do NOT
  hardcode any of those numbers anywhere — the table is the only place they exist.
- **"Hitch when the dialog opens"** → that IS the prompt prefill, paced by the dial's
  prefill-steps row (a lower level = smaller frames, a longer open). If opens still feel slow at
  a high level, shorten the system prompt (prefill cost is linear in it).
- **"Voice underruns / word-dribble"** → move the dial DOWN a level, which is counter-intuitive
  and correct: the tts tick rows run OPPOSITE to the LLM ones, so a lower tier spends MORE frames
  on the voice (4 ticks/frame at Very Smooth vs 1 at Very Fast). At 1 tick/frame synthesis runs
  0.35-0.8× real-time on a 1650 and the ring can only drain, whatever the prebuffer is. The voice
  no longer tunes itself — it just logs `ring starved mid-reply` and leaves the decision to you.
- **"Loading hitches on approach"** → `slowPrefetchSeconds` / `prefetchRadius` (give the
  stream more walk-up time), or a lower **Backend Tradeoff** level (its fetch row is the ceiling on
  bytes per frame; the walk-up governor runs at that ÷8). `LLM.UploadBudgetBytes` is the LIVE
  value the governor writes — read it, don't set it.

### R6 — Scripted events (quests, gifts, reactions)

- `AskNPCSilent(prompt)` — inject a hidden bracketed prompt (no visible player line, still in
  history). Example: the 2D GIVE button sends `[The player hands you: 3 carrots... pay 6
  coins]`.
- `protected virtual OnReplyFinished()` — fires when a reply completes (never on interrupt);
  the 2D demo pays the coins here. Subclass hook — no base changes needed for new events.
- For anything the PLAYER decides, use the decision-binding table below rather than either of
  the above.

### R6b — Decision bindings (how the game reacts to what the player chose)

The NPC has exactly two INTERACTIVE tools — `AskUserQuestion` (a choice) and `GiveItem` (an
offer with Accept/Decline) — and both produce the same thing: a **player decision**. A decision
is bound to game behaviour through the `Decisions` list on `NPCChatBase` (`NPC/NPCDecision.cs`),
which is a serialized `List<NPCDecision>` drawn in the inspector.

One binding row:

| field | meaning |
|---|---|
| `id` | stable, designer-authored key (`"sell_sword"`). What the game branches on. |
| `aliases` | how the NPC might WORD this decision's subject. |
| `onResolved` | `UnityEvent<NPCDecisionResult>` — runs when the binding resolves. |
| `gate` | optional `Component` implementing `INPCDecisionGate`; offers only. |

**The subject** is what gets matched: the **question text** for a question, the **item** for an
offer. Both go through one resolver, `NPCDecisionTable.Resolve`:

1. normalize both sides — trim, lowercase, collapse whitespace, strip surrounding punctuation,
   drop a leading `a`/`an`/`the`;
2. exact match on the binding's own `id`; then
3. exact match on an alias; then
4. an alias contained in the subject, or the subject contained in an alias
   (`"this old blade of mine"` → alias `"blade"`).

The first tier that hits anything decides, and within a tier the **first declared row wins** —
so resolution is deterministic and depends on table order and nothing else. When more than one
row could have hit, a warning names the ones that lost.

**Unmatched is not an error.** The global hooks (`ToolQuestionAnswered`, `GiveItemAcceptGate`,
`GiveItemAccepted`) fire for every decision, bound or not — the table is **additive**, it
replaced nothing — and the engine logs one `Debug.LogWarning` naming the unresolved subject and
listing every declared id, so a designer can see why their event did not fire.

**Gating** (offers only): the resolved binding's `gate` decides `canAccept` if it has one, else
the NPC-wide `GiveItemAcceptGate`, else true. False draws Accept disabled; Decline is never
gated. A gate that throws is reported and read as "yes" — it is a button state, not the
transaction.

**Ordering:** `onResolved` fires BEFORE the decision goes back to the model, in the same place
the old gated action ran, so the world is already updated when the reply streams. Subscribers
must not block.

Wire a binding in the scene BUILDER (builders are the source of truth) with
`UnityEditor.Events.UnityEventTools.AddPersistentListener`, so the hookup is visible and
editable in the inspector — see `ChatDemo3DBuilder.BindVelmireSwordSale`, which gives Velmire
one row (`sell_sword`, gate + event both on `NPCGearOffer`). Because the aliases are matched
against the model's own words, the NPC's persona should NAME the item plainly; Velmire's asks
for the bare word `sword`, and the aliases are the safety net for the runs where a 0.8B
embroiders.

Guarded headless by `NpcGiveItemProbe` (menu `DeepUnity/NPC/GiveItem Guard`), which asserts the
schema pins, the `{"accepted": …}` result bytes, every resolution tier, two bindings on one NPC
routing to different ids, the gate precedence, and the once-on-accept / never-on-decline
contract.

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
