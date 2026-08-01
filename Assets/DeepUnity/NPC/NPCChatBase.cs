using System.Collections;
using System.Collections.Generic;
using System.Text;
using System.Text.RegularExpressions;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity
{
    /// <summary>
    /// The chat-window surface NPCChatBase drives, so the NPC can talk to any environment's
    /// window without knowing the concrete type. Do NOT implement this directly: derive the
    /// window from <see cref="NPCDialogueWindow"/>, which implements the whole surface (plus the
    /// AskUserQuestion popup) and leaves only the presentation to the subclass.
    /// </summary>
    public interface INPCChatWindow
    {
        Button SendButton { get; }
        TMP_InputField InputField { get; }
        void Open();
        void Close();
        void Clear();
        void SetTitle(string title);
        void SetInfoText(string text);
        void SetSendLoading(bool loading);
        void AddMessage(string username, string message);
        void PopLastMessage();
        /// <summary>Render reasoning (&lt;think&gt;) content in the window, dimmed. It is never
        /// spoken by the TTS either way. Serialized on the window (off by default).</summary>
        bool ShowThinkingTokens { get; }
        /// <summary>Context-fill bar target (0..1 of the NPC's Max Context Length) — the golden
        /// bar above the input row; fed live per frame while a dialogue is open. Windows without
        /// a bar treat this as a no-op.</summary>
        void SetContextFill(float fill01);
    }

    /// <summary>
    /// Chat-window capability: the question+options panel behind the NPC's AskUserQuestion tool. It
    /// REPLACES the typing chrome while it is up — input field, context bar, Speak and Leave give
    /// way to the question and its options, so the window never offers both ways to answer at once.
    /// <see cref="NPCDialogueWindow"/> implements it for every environment, so deriving a window from
    /// that class is all any new environment needs; NPCChatBase feature-detects it
    /// (<c>Window as INPCToolQuestionWindow</c>) and drops the tool call with a console warning on a
    /// window that somehow lacks it. Implement it yourself only to replace the panel wholesale — to
    /// merely restyle it, override the base class's ToolQuestion* theme hooks; to change WHICH chrome
    /// it stands down, override CollectToolQuestionChrome.
    /// </summary>
    public interface INPCToolQuestionWindow
    {
        /// <summary>Show the choice panel (question + 2-4 clickable options) in place of the typing
        /// chrome. <paramref name="onPick"/> fires ONCE with the picked option's exact text; the panel
        /// tears itself down on pick and restores what it hid.</summary>
        void ShowToolQuestion(string npcName, string question, IReadOnlyList<string> options, System.Action<string> onPick);
        /// <summary>Tear the popup down without a pick (dialogue closed underneath it).</summary>
        void HideToolQuestion();
    }

    /// <summary>
    /// What a <b>GiveTool</b> call offers the player: an item, optionally a price, optionally a
    /// quantity. The two optional fields are NULLABLE and not defaulted, because "no price named"
    /// (a gift) and "priced at 0" are different offers and the panel renders them differently — the
    /// schema makes both parameters optional for exactly that reason.
    /// <para>Carried verbatim into <see cref="NPCChatBase.ToolGiveAcceptGate"/> and
    /// <see cref="NPCChatBase.ToolGiveAccepted"/>, so a host game reads the model's own numbers.</para>
    /// </summary>
    public struct ToolGiveOffer
    {
        /// <summary>What is being handed over, in the NPC's own words. Never null/empty — a call
        /// without it is refused before an offer is ever built (the schema requires it).</summary>
        public string item;
        /// <summary>The price the NPC named, or null when the offer carries none (a gift).</summary>
        public int? price;
        /// <summary>How many, or null when the NPC did not say (one of it).</summary>
        public int? quantity;
    }

    /// <summary>
    /// Chat-window capability: the item+Accept/Decline panel behind the NPC's <b>GiveTool</b> tool —
    /// the SECOND and last interactive tool the dialogue window handles (the other is
    /// AskUserQuestion). It replaces the typing chrome exactly like the choice panel does, shows the
    /// item (with its quantity and price when the NPC named them) and returns the player's decision
    /// as a bool. <see cref="NPCDialogueWindow"/> implements it for every environment, so deriving a
    /// window from that class is all any new environment needs; NPCChatBase feature-detects it
    /// (<c>Window as INPCToolGiveWindow</c>) and drops the call with a console warning on a window
    /// that somehow lacks it. Restyle through the base class's ToolQuestion* / ToolGive* hooks rather
    /// than implementing this yourself.
    /// </summary>
    public interface INPCToolGiveWindow
    {
        /// <summary>Show the offer panel (item line + Accept | Decline) in place of the typing chrome.
        /// <paramref name="onDecide"/> fires ONCE with true (accept) or false (decline); the panel tears
        /// itself down on the click and restores what it hid. <paramref name="canAccept"/> false renders
        /// Accept DISABLED — the host game's accept-gate said no (no money, no room) — while Decline
        /// always works, so the exchange can never dead-end.</summary>
        void ShowToolGive(string npcName, ToolGiveOffer offer, bool canAccept, System.Action<bool> onDecide);
        /// <summary>Tear the panel down without a decision (dialogue closed underneath it).</summary>
        void HideToolGive();
    }

    /// <summary>
    /// A component that gives its NPC EXTRA tools on top of the built-in AskUserQuestion popup.
    /// A provider is a MonoBehaviour on the same GameObject as the <see cref="NPCChatBase"/>, so
    /// which tools an NPC has is authored in the SCENE (Velmire can hand his gear over; Morwenna
    /// cannot) instead of being hard-coded in the base class.
    /// <para>Providers own <b>internal</b> tools: world-state reads the player never sees. The
    /// returned JSON goes straight back to the model as the &lt;tool_response&gt; and generation
    /// resumes in the same breath — no window, no player turn. Anything the player must decide
    /// belongs in AskUserQuestion instead, and any real value transfer belongs in the engine code
    /// that reacts to the pick (<see cref="NPCChatBase.ToolQuestionAnswered"/>), which keeps the
    /// dataset's gating law intact: reads are free, asking is free, giving is not.</para>
    /// <para>Because a read is re-callable, it is also how NPC memory survives compaction: state
    /// that lives in the world (has the player got the sword? did I already give mine away?) can be
    /// re-established with one call after the transcript is summarized away.</para>
    /// </summary>
    public interface INPCToolProvider
    {
        /// <summary>One JSON function schema per tool, in the same compact
        /// <c>{"type": "function", "function": {...}}</c> shape the SFT dataset uses — spliced into
        /// the prompt's &lt;tools&gt; block. Keep them SHORT: the NPC pays for every tool in context
        /// on every single turn.</summary>
        IEnumerable<string> ToolSchemas { get; }

        /// <summary>Answer a call this provider owns and return its result as JSON, or return null
        /// when <paramref name="toolName"/> is not ours (the next provider is offered the call).
        /// Runs synchronously on the main thread — a tool that must wait does not belong here.</summary>
        string TryHandleTool(string toolName, string argumentsJson);

    }

    /// <summary>
    /// Shared base for every demo dialogue NPC (3D souls castle, 3D forest fork, 2D farm) backed
    /// by a local full-GPU LLM and optional streaming TTS. Owns the whole conversation stack:
    /// model selection + lazy load (EnsureLlm), per-engine voice wiring (EnsureVoice), the
    /// latent-loading prefetch zone (slow prefetch on entry, GPU RESIDENCY while the player is
    /// inside, deload on exit; transparent green sphere/disc gizmo, auto 2D/3D), the dialogue
    /// state machine (open / ask / stream / close, Escape-interrupt), per-token TTS fan-out, the
    /// audio-driven talk-animation watch, and the three
    /// <see cref="HistoryMode"/> persistence behaviors. Subclasses keep only their
    /// presentation specifics: camera framing, chat window type, player controller type, trigger
    /// colliders and character animation.
    ///
    /// Several NPCs can share one chat window: every UI listener fires on all interactors, but
    /// only the one actually in interaction reacts (AskNPC guards on its state; CloseInteraction
    /// no-ops when Idle).
    /// </summary>
    public abstract class NPCChatBase : MonoBehaviour
    {
        public enum NPCState
        {
            Idle,
            PreparingForInteraction,
            WaitingInInteraction,
            TalkingInInteraction,
        }

        public enum TtsModel { Chatterbox, CosyVoice3, Kokoro, PocketTTS }
        /// <summary>How the NPC answers: text-only, or text + streaming speech.</summary>
        public enum ConversationMode { LlmOnly, LlmPlusTts }

        /// <summary>Granularity of the audio-synced text reveal (see syncedTextReveal).</summary>
        public enum RevealGranularity { CharByChar, WordByWord }

        /// <summary>
        /// What happens to the conversation HISTORY between two openings of the dialogue.
        /// (GPU residency is NOT decided here — the prefetch zone / talk trigger owns that.)
        /// <para><b>ResetEveryTime</b> — only the SYSTEM PROMPT is ever cached. Closing the dialogue
        /// wipes the transcript and marks the live KV dead, so the next opening is a fresh
        /// <c>InitializeChat(EffectiveSystemPrompt)</c> that hits the system-prompt KV disk cache and
        /// nothing else. Same after quitting the app. Exactly like restarting a Claude Code session:
        /// the persona is there, the conversation is not.</para>
        /// <para><b>ResumeFromCompact</b> — the history is PERMANENT: closing the dialogue, walking out
        /// of the residency zone (the model unloads) or quitting the app all keep every reply. Three
        /// tiers serve a reopen, in order: (a) our conversation KV is still live on the pooled GPU
        /// instance → just keep talking; (b) restore the whole conversation from the disk snapshot
        /// written on the last clean close (KV/SSM + sampler state + transcript + compact) — this is
        /// the tier that survives an app restart, and it needs <c>cacheKVCache</c>; (c) re-prefill
        /// <see cref="BuildResumePrompt"/> (system prompt + HISTORY block + the recorded turns). The
        /// window is repopulated with the stored turns either way, so the player sees the chat they
        /// left.</para>
        /// <para><b>What compaction does</b> (ResumeFromCompact only, at <c>maxContextLength</c>): the
        /// model summarizes the WHOLE conversation into one text; that text is appended to the system
        /// prompt as a <c>## MEMORY</c> block, the KV is recomputed from that prefix alone, and the
        /// individual replies are dropped (the window starts visually empty on the next open). So a
        /// compacted NPC is in the same shape as ResetEveryTime — except its system prompt now carries
        /// the compact. There is only ever ONE compact in the prompt: a later compaction REPLACES it,
        /// and it can do that losslessly because the old HISTORY block is still in the model's context
        /// when it is asked to compact, so it folds forward whatever it judges worth keeping (the
        /// dataset teaches exactly this). The compact is visible in the inspector as Compact Summary.</para>
        /// </summary>
        public enum HistoryMode
        {
            [Tooltip("Only the SYSTEM PROMPT is cached. The conversation ceases to exist the moment the chat CLOSES (transcript wiped, live KV marked dead), so every opening — and every app restart — starts from the bare persona, like restarting a Claude Code session.")]
            ResetEveryTime,
            // NOTE: the former middle value ContinueWhereLeftOff was removed 2026-07-15 — halting at
            // the limit was pointless (a full conversation is simply over); ResumeFromCompact keeps
            // talking instead. The enum is kept CONTIGUOUS (0,1) so (int)value == enumValueIndex ==
            // the serialized value; the builders' SetEnum uses enumValueIndex, so a gap would throw
            // "enum index is out of range". Scenes that used the old value were remapped to this one.
            [Tooltip("The history is PERMANENT — with every reply, across closing the dialogue, unloading the model and quitting the app (live KV while resident, else the disk snapshot, else a re-prefill of the recorded turns; the window is repopulated so you see the chat you left). When the conversation reaches Max Context Length the model COMPACTS itself: it summarizes the whole chat in one shot, that text is APPENDED TO THE SYSTEM PROMPT as a HISTORY block (visible below as Compact Summary), the KV is recomputed from that prefix and the individual replies are dropped — from there it is like ResetEveryTime, but with the compact baked into the prompt. Only ONE compact ever exists: the next compaction replaces it and folds forward whatever still matters (the old block is still in context when it compacts). The KV is allocated above the limit so the compact pass has room; the limit-hitting reply is always delivered IN FULL (decoded, typed and spoken to the end) and 'Compacting…' appears only after the voice finishes, with input blocked until it lands. Crash-recovery: compacts on the next open if one never landed. Never canceled once started.")]
            ResumeFromCompact,
        }

        [SerializeField, ViewOnly] protected NPCState state = NPCState.Idle;
        [SerializeField, UnityEngine.Serialization.FormerlySerializedAs("npc_name")] protected string NpcName = "Villager";
        // NOT called system_prompt (renamed 2026-07-25): "system prompt" is reserved, in this
        // project's vocabulary, for the WHOLE text that goes into Qwen's system message — tools,
        // rules, this description and the ## MEMORY block. This field is only the description-and-
        // rules part an author writes, so calling it the system prompt made people (rightly) expect
        // the inspector field to be everything the model reads. The Effective System Prompt foldout
        // below shows the real thing.
        [Tooltip("Who this NPC is: persona, world facts, its rules — its only source of truth. NOT the whole system prompt: the model receives the # Tools block first (while tools are on), then this text, then the Compact Summary below under a ## MEMORY heading. The 'Effective System Prompt' foldout shows the assembled result. Editing this invalidates the NPC's KV caches.")]
        [TextArea(4, 12)]
        [SerializeField, UnityEngine.Serialization.FormerlySerializedAs("system_prompt")]
        protected string descriptionAndRules =
            "You are a friendly villager. Stay in character at all times. " +
            "Keep your replies to one to three short sentences.";
        // Sits directly under the description on purpose (user 2026-07-25): it IS part of the prompt
        // the model sees — EffectiveSystemPrompt/BuildResumePrompt append it under ## MEMORY — so the
        // two read together in the inspector.
        [Tooltip("RUNTIME STATE (ResumeFromCompact): the model's own summary of everything before the last compaction, appended to the text above under a ## MEMORY heading. Only ever ONE — a new compaction replaces it, folding forward what still matters. Survives leaving play mode; Reset Conversation clears it. Edit it to hand-write the NPC's memory.")]
        [TextArea(2, 8)]
        [SerializeField] protected string compactSummary;

        // Lives in the BASE class (user 2026-07-25): every window derives from NPCDialogueWindow, which
        // implements the whole surface NPCChatBase drives, and the component cannot run a single
        // dialogue without one — so it is a base-class requirement, not a per-environment detail. The
        // field name is unchanged from the subclasses' own, so existing scene references still bind.
        [Tooltip("The dialogue window this NPC talks through — REQUIRED, nothing works without it. Any window deriving from NPCDialogueWindow (SoulsChatWindow in 3D, ChatWindow2D in 2D); one window is normally shared by every NPC in the scene.")]
        [SerializeField] protected NPCDialogueWindow chatWindow;
        // "Backend Tradeoff": drawn by NPCChatBaseEditor as its own labeled dial with the selected
        // row's numbers under it. Serialized PER NPC but engine-wide once applied
        // (BackendTradeoffTable.Level is static, as it must be — one GPU, one frame): the level of the
        // NPC you are talking to is the level in force. Replaced the continuous smoothVsSpeed float
        // 2026-07-26 (float→enum cannot be migrated by FormerlySerializedAs, so those scene values
        // were dropped); renamed from llmTradeoff 2026-07-27, which enum→enum DOES migrate.
        // SECOND field of the CONVERSATION group, not the tail of the LLM one (moved 2026-07-27): it
        // stopped being an LLM setting the moment the table started pacing the voice too, and the
        // machine you are on is the frame every other choice here is made inside — so it reads
        // directly under the window, before the modes.
        [Tooltip("How capable is this machine, as five fixed presets — the ONE dial the engine has. One pick sets EVERY per-frame backend budget at once: weight bytes fetched per frame while any model loads, prompt-prefill steps per frame, whole tokens decoded per frame, AND the voice's pacing (heavy TTS ticks per frame speaking and refilling, prebuffer seconds, decode chunk size, and how much banked audio makes it yield a frame to the LLM). Very Smooth = an old 2 GB card, Smooth = a GTX 1650 / 4 GB laptop (the reference machine), Balanced = a healthy mid-range desktop GPU, Fast / Very Fast = cards with headroom, where a loading hitch is cheaper than waiting. Note the TTS numbers run the OTHER WAY: a weak machine must spend MORE frames defending the audio, because a stutter in speech is worse than a stutter in framerate. Nothing self-tunes: this dial IS how the engine learns which machine it is on. Live — moving it mid-dialogue applies from the next frame, except the two fields pushed onto the voice component (prebuffer, chunk frames), which are written once in EnsureVoice like every other voice setting.")]
        // Smooth, not Balanced: this repo's reference machine is a GTX 1650 / 4 GB laptop, and Smooth's
        // 8 MB/frame fetch is EXACTLY the literal the engine shipped before this dial existed — so a
        // scene rebuilt after the refactor loads at the same rate it always did, and only prefill and
        // decode change (both because the old self-tuner was pinned at its floor, not by design).
        // A stronger machine raises this per NPC; it is the one number a new machine has to be told.
        [UnityEngine.Serialization.FormerlySerializedAs("llmTradeoff")]
        [SerializeField] protected BackendTradeoffLevel backendTradeoff = BackendTradeoffLevel.Smooth;
        [Tooltip("LlmOnly = text-only replies (talk animation follows the writing; voice fields hidden below). LlmPlusTts = replies are spoken: the talk animation follows the AUDIO, and the next sentence synthesizes while the current one plays.")]
        [SerializeField] protected ConversationMode conversationMode = ConversationMode.LlmOnly;
        protected bool speakReplies => conversationMode == ConversationMode.LlmPlusTts;
        [Tooltip("How the audio-synced reply types itself into the window while the NPC speaks (LlmPlusTts only). CharByChar = typewriter, letters land in step with the voice. WordByWord = whole words pop in on their spoken beat. Text-only replies stream per LLM token and ignore this.")]
        [SerializeField] protected RevealGranularity syncedTextReveal = RevealGranularity.CharByChar;
        [Tooltip("ResetEveryTime = the chat starts from the bare system prompt EVERY opening (only the system-prompt KV is cached). ResumeFromCompact = the history is permanent — it survives closing the dialogue AND quitting the app, with every reply — until it fills Max Context Length, at which point the model compacts the whole conversation into one text that is APPENDED TO THE SYSTEM PROMPT (the Compact Summary field below) and the replies are dropped. See the dropdown's own tooltips for the full lifecycle.")]
        [SerializeField] protected HistoryMode historyMode = HistoryMode.ResetEveryTime;
        [Tooltip("Runaway guard, NOT a feature limit. One exchange may legitimately run text → call → text → call → text: decoding stops at each call, the result lands in the context, and the NPC carries on from there, so a single player line can produce several calls and several spoken stretches. This caps only the INTERNAL reads (an un-finetuned small model handed a read will happily read the same thing forever); the two INTERACTIVE tools (AskUserQuestion, GiveTool) are not counted, because neither can loop — both wait for a human. Past the cap the read is REFUSED to the model as {\"error\": \"read_limit_reached\"}, which is what makes it stop and speak. 0 = no internal reads at all (every one is refused).")]
        [Min(0)] [SerializeField] protected int maxToolReadsPerTurn = 6;
        [Tooltip("Persist this NPC's KV cache to disk (persistentDataPath/DeepUnity): the system-prompt state in EVERY mode, plus — in ResumeFromCompact — the WHOLE conversation on a clean close (KV + sampler state + transcript + compact), so reopening after the model was released, the scene reloaded or the app restarted restores the chat from disk instead of re-prefilling. This is what makes ResumeFromCompact survive an app restart, so leave it ON for that mode. Qwen3.5 only for now; Gemma3 NPCs fall back to the re-prefill path.")]
        [SerializeField] protected bool cacheKVCache = true;

        [Tooltip("Which local LLM voices this NPC — the dropdown lists every model registered in LLMRegistry, so a freshly ported LLM appears here automatically. Sampling fields at -1 fall back to this model's Config presets.")]
        [SerializeField] protected string model = "Qwen3.5-0.8B";
        [Tooltip("Weight format. INT8 is ~lossless at half the VRAM — the recommended default. INT4 is lossy on models this small (Gemma int4 collapses outright).")]
        [SerializeField] protected LLMQuant quantization = LLMQuant.INT8;
        [Tooltip("Context window (tokens) — the conversation size the history mode acts on. ResumeFromCompact auto-compacts here (and allocates the KV +8192 above it for the compact pass). Sizes the KV cache (pre-allocated → more = more VRAM). 8192 default. Instances are pooled per (model, quant, KV, this length + headroom), so NPCs sharing a model should share this value.")]
        [Min(256)] [SerializeField] protected int maxContextLength = 8192;
        [Tooltip("Let a thinking-capable model (Qwen3.5) reason in <think> before answering. The reasoning is NEVER voiced and never shown as reply text (the window's ShowThinkingTokens debug toggle can render it dimmed); while the model thinks, the dialog pulses an animated 'Thinking…' placeholder until the final answer starts. Non-thinking models ignore this.")]
        [SerializeField] protected bool allowThinking = false;
        [Tooltip("The NPC's built-in interactive tool, ON by default. A # Tools block rides in the system prompt; a <tool_call> in the reply pulses 'Tool calling…' and then replaces the input row with the question + its clickable options (the JSON itself is never rendered, but the QUESTION is spoken like any other line), and the player's pick returns to the model as the <tool_response> result — the model then reacts to the choice in a fresh streamed turn. Every window deriving from NPCDialogueWindow renders it. Costs ~300 tokens of system prompt (Qwen3.5's own tools preamble — a shorter paraphrase measurably stops the model from ever calling): turn it off for an NPC with a very small Max Context Length, or one that should never offer choices.")]
        [SerializeField] protected bool enableAskUserQuestion = true;
        [Tooltip("The NPC's second interactive tool — handing the player an ITEM: a gift, promised gear, or a sale at a price the NPC names. A <tool_call> replaces the input row with the item (its quantity and price too, when he named them) and exactly two buttons, Accept and Decline, and the player's decision returns to the model as {\"accepted\": true} or {\"accepted\": false}. The game decides whether Accept is even offered (accept-gate: not enough money -> Accept is disabled, Decline always works) and performs the transfer itself when it lands, so the model can offer but never give. OFF by default: it costs ~130 tokens of system prompt and only an NPC that actually hands things over needs it. Every window deriving from NPCDialogueWindow renders it. These two are the ONLY interactive tools; anything else an NPC does is an internal read on an INPCToolProvider.")]
        [SerializeField] protected bool enableGiveTool = false;

        [Header("Sampling (-1 = model preset)")]
        [Min(0f)] [SerializeField] protected float temperature = 0.8f;
        [Tooltip("Reply length cap in tokens.")]
        [SerializeField] protected int maxNewTokens = 1024;
        [Tooltip("-1 = model preset. 0 disables top-k filtering.")]
        [SerializeField] protected int topK = -1;
        [Tooltip("-1 = model preset. 1 disables nucleus filtering.")]
        [SerializeField] protected float topP = -1f;
        [Tooltip("-1 = model preset. 0 disables min-p filtering.")]
        [SerializeField] protected float minP = -1f;
        [Tooltip("-1 = model preset. 0 disables the presence penalty.")]
        [SerializeField] protected float presencePenalty = -1f;
        [Tooltip("-1 = model preset. 1 disables the repetition penalty.")]
        [SerializeField] protected float repetitionPenalty = -1f;

        [Tooltip("PocketTTS = Kyutai 100M AR, RTF ~0.15 int8 (speaks in real time DURING generation, voice cloning — DEFAULT); Kokoro = 82M non-AR, RTF ~0.3; Chatterbox = clause-streamed (RTF~1.4); CosyVoice3 = streaming-native. 2D demo NPCs are Kokoro-only and ignore this.")]
        [SerializeField] protected TtsModel ttsModel = TtsModel.PocketTTS;
        [Tooltip("TTS weight format. PocketTTS: int8 = 116 MB vs 209 MB fp16, same speed, mel-gated parity (also picks which voice-clone cache dir is used). Chatterbox: int8 = T3 matmuls int8 (~300 MB less, parity-validated); s3gen stays fp16 either way. Kokoro/CosyVoice ignore this (their voice component's weightsPath decides).")]
        [SerializeField] protected LLMQuant ttsQuantization = LLMQuant.INT8;
        [Tooltip("Playback pitch for this NPC. 1 = natural (the voice's own timbre); <1 = deeper/slower.")]
        [SerializeField] protected float voicePitch = 1.0f;
        [Tooltip("Loudness of this NPC's voice. AudioSource.volume tops out at 1, so this multiplies the samples themselves — >1 = louder (peaks clamp at full scale).")]
        [Min(0f)] [SerializeField] protected float voiceVolume = 1.4f;
        [Tooltip("How loud the rest of the game stays while THIS NPC's dialogue is open — music, ambience, footsteps, everything outside the conversation. 1 = untouched, 0.5 = half, 0 = silence. Needs a ConversationAudioDucker in the scene, which eases there over ~3 s and back on close.")]
        [Range(0f, 1f)] [SerializeField] protected float worldAudioWhileTalking = 1f;
        [Tooltip("BAKED voice shipped inside the selected TTS engine's weights export (voices/<name> dirs for PocketTTS/CosyVoice3, voices/<name>.bin voicepacks for Kokoro) — the inspector dropdown lists what's on disk. Pick 'Clone (reference clip)' on PocketTTS to clone from an AudioClip instead; a non-null clip always overrides this name.")]
        [SerializeField] protected string ttsVoice = "jean";
        [Tooltip("PocketTTS only: reference clip to VOICE-CLONE for this NPC (overrides the baked ttsVoice). First runtime use encodes it once through the Mimi encoder and caches by content hash; press 'Precompute voice-clone cache' below to bake the embedding into the shared Resources/Cache so runtime (editor AND builds) is a pure load — no recompute, ever.")]
        [SerializeField] protected AudioClip clonedVoiceClip;
        [Tooltip("Sentences per spoken chunk. Smaller = faster response, lower quality (prosody resets each sentence); larger = higher quality (intonation flows across sentences), slower response.")]
        // 2, not 1 (user 2026-07-26): one sentence per chunk restarts prosody at every full stop and
        // pays a TTS round-trip per sentence, which on this 4 GB card is the shakier trade. This value
        // is pushed onto whichever voice component the NPC ends up with (see the ttsModel switch), so
        // it — not the voice component's own default — is what every NPC actually runs.
        [Range(1, 3)] [SerializeField] protected int clausesPerChunk = 2;
        // Collapsed from three knobs to one (user 2026-07-26): a cut only ever lands at an ender, and
        // every pause INSIDE a clause is the TTS model's own prosody — there was nothing for a
        // comma-specific value to control.
        [Tooltip("PocketTTS pacing: pause between spoken chunks, in seconds. Pauses inside a chunk are rendered by the model itself.")]
        [Min(0f)] [SerializeField] [UnityEngine.Serialization.FormerlySerializedAs("sentencePauseSeconds")]
        protected float clausePauseSeconds = 0.36f;
        [Tooltip("PocketTTS pacing: extra model-generated tail on the reply's last chunk, in seconds — lets the final word decay naturally instead of cutting ~0.16 s after it.")]
        [Min(0f)] [SerializeField] protected float replyTailSeconds = 0.32f;

        [Tooltip("ON: the big sphere (3D) / circle (2D — auto-detected) around the NPC is the model-RESIDENCY zone: entering slow-prefetches the LLM + TTS in the background, both stay on the GPU while the player is inside (closing the chat releases nothing), and leaving unloads both. OFF: the small talk trigger plays that role instead — load on contact, unload when the player walks off it.")]
        [SerializeField] protected bool usePrefetchZone = false;
        [Min(0f)]
        [Tooltip("Radius of the prefetch/residency zone (transparent green sphere/disc gizmo). " +
                 "BIGGER = models start loading earlier and further away, so they're fully resident " +
                 "(and instantly responsive) by the time the player reaches the NPC — at the cost of " +
                 "holding GPU memory while the player merely passes nearby, and more wasted " +
                 "load/unload churn if they wander in and out. SMALLER = VRAM is held only near the " +
                 "NPC, but a running player can beat the load and hit a not-ready NPC (the stream " +
                 "then boosts to full speed, which can cost a few heavy frames). Rule of thumb: " +
                 "radius ≥ player speed × slowPrefetchSeconds, larger on slow GPUs/disks.")]
        [SerializeField] protected float prefetchRadius = 10f;
        [Min(1f)]
        [Tooltip("The TTS weight stream is spread over ~this many seconds of walking-up time. " +
                 "BIGGER = gentler per-frame upload budget (imperceptible, zero frame drops) " +
                 "but the voice needs longer to become ready — pair with a larger prefetchRadius. " +
                 "SMALLER = ready sooner after zone entry, but each frame uploads more bytes and " +
                 "weak GPUs/disks may show hitches during the walk-up. The LLM does NOT use this " +
                 "window: inside the zone it streams at the tier's slow rate for as long as the " +
                 "player idles there, and BOOSTS to full speed only when the dialogue opens " +
                 "(or the model finishes / the zone releases it).")]
        [SerializeField] protected float slowPrefetchSeconds = 3f;

        // Sits below the zone knobs (user 2026-07-25): it is a scene reference the author wires once,
        // not something tuned while iterating, so it does not belong up among the dialogue settings.
        [Tooltip("The walk-up prompt (\"[I] Speak\" / \"Talk — [ E ]\") shown while the player is in the talk trigger. Its OWN component on its OWN GameObject (fade/bob/text knobs live there) — the NPC only calls Show/Hide on it.")]
        [SerializeField] protected NPCInteractPrompt interactPrompt;

        // ---------------------------------------------------------------- runtime state
        protected LLM llm;
        protected ChatterboxVoice voice;
        protected CosyVoiceModeling.CosyVoiceVoice cvVoice;
        protected KokoroVoice kkVoice;
        protected PocketTTSModeling.PocketTTSVoice pkVoice;
        protected Coroutine dialogueCoroutine;
        // Dialogue GENERATION counter: bumped on every open and close. Long-lived coroutines
        // (Talk, OpenConversation, the close/interrupt waiters) capture it at start and stand
        // down when it moved — a stale coroutine must never touch the handle/state of the
        // session that replaced it (audit #5/#6).
        protected int dialogueEpoch;
        protected Transform playerZoneT;      // player transform for the prefetch-zone distance check
        protected bool inPrefetchZone;

        // last value fed to OnTalkingChanged: audio-driven in Kokoro TTS mode, state-driven otherwise
        private bool talkAnimActive;
        /// <summary>True while the NPC's streaming speech ring is actually audible (Kokoro/PocketTTS
        /// LLM+TTS mode) — drives the talk animation from the AUDIO, not the token stream.</summary>
        protected bool IsVoiceAudible => (kkVoice != null && kkVoice.IsAudioPlaying)
                                      || (pkVoice != null && pkVoice.IsAudioPlaying);
        /// <summary>The last talk-animation value the watch fired (audio-driven with Kokoro, state-driven otherwise).</summary>
        protected bool TalkAnimActive => talkAnimActive;

        // Conversation persistence: the running transcript (recorded in the continue modes) and
        // whether the CURRENT llm instance holds a live, trustworthy conversation KV. chatLive
        // goes false whenever the model is released or a reply is interrupted mid-generation
        // (StopCoroutine can land between a forward pass's per-layer yields — half-written KV).
        // Serializable so the transcript can ride inside the on-disk conversation KV file
        // (SaveConversationKV userState) and come back after a scene reload.
        /// <summary>Read-only peeks for test harnesses (the E2E probe drives real dialogues).</summary>
        public NPCState State => state;
        public bool LlmLoaded => llm != null;
        public bool LlmReady => llm != null && llm.IsReady;
        /// <summary>True once the player has actually SPOKEN in this conversation — at least one
        /// recorded turn they typed. A <c>tool</c> turn does NOT count: its <c>user</c> text is a
        /// &lt;tool_response&gt; carrying an AskUserQuestion pick or a provider read, which the player
        /// never typed (see <see cref="Turn"/>). The inspector's Reset Conversation button is gated on
        /// this, so "there is nothing but the system prompt yet" reads as an unavailable button rather
        /// than a click that does nothing.</summary>
        public bool HasPlayerMessage
        {
            get
            {
                foreach (var t in transcript)
                    if (!t.tool && !string.IsNullOrWhiteSpace(t.user)) return true;
                return false;
            }
        }

        // `tool` marks a turn the PLAYER never typed: the <tool_response> that carried an
        // AskUserQuestion pick or an internal provider read. It still belongs in the history (the
        // NPC's reply only makes sense next to it) but it must never be replayed as a player line
        // on a resume, nor drawn as a "You" bubble when the window is repopulated.
        // `cut` marks a turn the player talked over: generation was cancelled mid-reply, so the text
        // ends wherever the token stream happened to stop — usually mid-word. It is presentation only
        // (the window appends CutMark) and deliberately NOT baked into `npc`: that string is the
        // model-facing record and rich-text tags have no business in it. JsonUtility defaults a
        // missing bool to false, so older on-disk transcripts deserialize as "not cut".
        // `npcShown` is what the window actually REVEALED of a cut turn, which is less than `npc`: the
        // reveal follows the voice, generation ran ahead of it, and cancelling stops generation later
        // than speech. Only `npc` is the model-facing record (it matches the KV); this one exists so a
        // reopened window shows what the player HEARD instead of paragraphs they never got.
        //   null => a cut path that did not capture it: fall back to `npc` rather than lose text
        //   ""   => captured, nothing was revealed: draw no bubble at all
        //   text => captured: draw it with CutMark
        [System.Serializable] private class Turn
        { public string user; public string npc; public bool tool; public bool cut; public string npcShown; }
        [System.Serializable] private class TranscriptState { public List<Turn> turns = new List<Turn>(); public string summary; }
        private readonly List<Turn> transcript = new List<Turn>();
        private bool chatLive;
        private Turn activeTurn;                 // turn currently being generated (for interrupt finalize)
        // The turn that has finished GENERATING but is still being spoken/revealed. Generation ends
        // seconds before the voice does (pocket synthesizes ~6x realtime, so late in a reply that
        // window is SECONDS long — the same reason AskNPC tests VoicesAudible() and not
        // dialogueCoroutine). activeTurn is null throughout it, so marking a cut off activeTurn alone
        // missed every interrupt and every Leave that happened while he was still talking, which is
        // the majority of them (found by audit 2026-07-28).
        private Turn drainTurn;
        private StringBuilder activeResponse;    // its streaming reply buffer
        // A background conversation-KV save is reading GPU state. Keyed PER LLM INSTANCE (audit
        // #9): pooled instances are shared across NPCs, so nobody may reset/forward THAT model
        // while a save still reads it — but a save on one model must not stall/skip an NPC on a
        // DIFFERENT one. Value = the NPC that latched the entry (OnDisable drops only its own).
        private static readonly Dictionary<LLM, NPCChatBase> kvSavesInFlight = new Dictionary<LLM, NPCChatBase>();
        private static bool KvSaveInFlightFor(LLM m) => m != null && kvSavesInFlight.ContainsKey(m);
        // ResumeFromCompact maintenance state: the in-flight compaction coroutine and — STATIC, same
        // pooled-model reasoning as kvSavesInFlight — which NPC is compacting, so a dialogue opening
        // on the shared instance WAITS for it before driving the model itself (user rule: a
        // compaction is never canceled once its Chat started — the window pulses "Compacting…" and
        // input stays blocked until the compact lands). The compact TEXT itself is the serialized
        // compactSummary field up in the Conversation section, so it is visible in the inspector.
        private Coroutine compactRoutine;
        private static NPCChatBase compactingNpc;
        // The in-flight manual reset (ResetConversation). Deliberately NOT paired with a static
        // "resettingNpc" the way compactRoutine is paired with compactingNpc: a reset only ever runs
        // when THIS NPC owns the conversation on the instance, and it re-establishes that same NPC's
        // system prompt — so a sibling on the same pooled model has nothing to wait for, whereas a
        // sibling's compaction genuinely drives the shared model.
        private Coroutine resetRoutine;

#if UNITY_EDITOR
        // Sampling GATES, not defaults (user 2026-07-25): stop nonsense values being typed in, without
        // touching what the fields mean. -1 stays the "use this model's preset" sentinel on every field
        // that has one (see how Chat() reads them: `topK >= 0 ? topK : Config.DefaultTopK`), so anything
        // negative snaps to exactly -1 rather than being clamped up into a real value — clamping topP to
        // 0 would silently turn "let the model decide" into "nucleus off", which is a different run.
        // topK keeps 0 as its own meaning (top-k filtering disabled), so it is not floored at 1.
        protected virtual void OnValidate()
        {
            if (temperature < 0f) temperature = 0f;
            if (topK < 0) topK = -1;
            topP = topP < 0f ? -1f : Mathf.Clamp01(topP);
            minP = minP < 0f ? -1f : Mathf.Clamp01(minP);
            if (presencePenalty < 0f) presencePenalty = -1f;
            if (repetitionPenalty < 0f) repetitionPenalty = -1f;
        }
#endif

        // ---------------------------------------------------------------- subclass surface

        /// <summary>The shared chat window, from the serialized field the base class now owns. The
        /// <c>!= null</c> is a Unity null check on purpose: an unassigned or destroyed component must
        /// read as real null through the interface, which a plain cast would not give.
        /// <para>Override only to source the window from somewhere other than the field.</para></summary>
        protected virtual INPCChatWindow Window => chatWindow != null ? chatWindow : null;
        /// <summary>Key that opens the dialogue while the player is in the talk trigger.</summary>
        protected abstract KeyCode InteractKey { get; }
        /// <summary>True when a player is in the trigger and free to start an interaction.</summary>
        protected abstract bool PlayerReady { get; }
        /// <summary>Seconds to wait (camera transition) before the chat window opens.</summary>
        protected abstract float DialogueOpenDelay { get; }
        /// <summary>Camera framing + player EnterInteractiveMode + facing; runs on interaction start.</summary>
        protected abstract void OnInteractionStarted();
        /// <summary>Camera back + player ExitInteractiveMode + prompt re-show; runs on interaction close.</summary>
        protected abstract void OnInteractionClosed(bool interrupted);
        /// <summary>Talk-animation hook: audio-driven in Kokoro TTS mode, state-driven otherwise.</summary>
        protected virtual void OnTalkingChanged(bool talking) { }
        /// <summary>Per-demo AudioSource setup (3D spatial vs 2D flat) after the voice component is built.</summary>
        protected virtual void ConfigureVoiceAudioSource(AudioSource src) { }
        /// <summary>The TTS model actually used. The 2D demo overrides this to Kokoro (Kokoro-only).</summary>
        protected virtual TtsModel EffectiveTtsModel => ttsModel;

        /// <summary>The prefetch zone auto-adapts to the NPC's own dimensionality: an NPC carrying
        /// any 2D piece (sprite / 2D collider anywhere under it) gets a circle in the XY plane
        /// (2D scenes layer depth on Z, which must not count), everything else a sphere.</summary>
        protected bool ZoneIs2D
        {
            get
            {
                if (zoneIs2DCached == null)
                    zoneIs2DCached = GetComponentInChildren<SpriteRenderer>(true) != null
                                  || GetComponentInChildren<Collider2D>(true) != null;
                return zoneIs2DCached.Value;
            }
        }
        private bool? zoneIs2DCached;

        /// <summary>Prefetch-zone membership test (planar for 2D NPCs, spherical for 3D).</summary>
        protected virtual bool IsPlayerInsideZone(Vector3 playerPos)
        {
            Vector3 d = playerPos - transform.position;
            if (ZoneIs2D) d.z = 0f;
            return d.sqrMagnitude <= prefetchRadius * prefetchRadius;
        }

        // ---------------------------------------------------------------- lifecycle

        protected virtual void Start()
        {
            // Scene-start prewarm: compiles the model's compute kernels (one per frame) and parses
            // the tokenizer in the background while the player walks around, so the dialogue
            // later opens without hitches. The static prewarm flag is per-model, so several NPCs
            // sharing a model make the extra calls cheap no-ops.
            var prewarm = LLMRegistry.Find(model)?.prewarm;
            if (prewarm != null) StartCoroutine(RunPrewarm(prewarm()));
            else prewarmDone = true;

            if (speakReplies)
                EnsureVoice();

            if (interactPrompt != null) interactPrompt.HideInstant();
            Window?.SetTitle(NpcName);

            var playerGO = GameObject.FindWithTag("Player");
            if (playerGO != null) playerZoneT = playerGO.transform;
        }

        // ---- session-wide kernel prewarm, INSIDE the scene-load frame -------------------------
        // Compiling a model's compute kernels is a one-time driver cost that must land somewhere:
        // spread one-per-frame it was ~2 s of ~30 fps at scene start (the old LLMBootHelper
        // object), skipped it hits as a hitch at the first chat open. Draining it whole in Awake
        // hides it in the scene-load blackout instead — and every NPC prewarms its OWN model via
        // LLMRegistry, so there is no helper object to remember. Once per model per session.
        static readonly HashSet<string> prewarmedModels = new HashSet<string>();

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetPrewarmedModels()   // domain-reload-off replays
        {
            prewarmedModels.Clear();
            kvSavesInFlight.Clear();   // stale gate entries (dead instances) must not survive either
        }

        protected virtual void Awake()
        {
            // Prewarm every distinct model present in the scene — INCLUDING NPCs sitting on
            // INACTIVE GameObjects, whose own Awake would otherwise fire only at activation and
            // drop the kernel-compile hitch mid-game right as they get switched on. Deduped per
            // session (prewarmedModels), so after the first NPC of the scene this whole block is
            // a no-op; NPCs spawned/loaded later still cover themselves via their own Awake.
            PrewarmModel(model);
            bool anyPocket = ttsModel == TtsModel.PocketTTS;
            foreach (var npc in FindObjectsOfType<NPCChatBase>(true))
            {
                PrewarmModel(npc.model);
                if (npc.ttsModel == TtsModel.PocketTTS) anyPocket = true;
            }
            // TTS kernels compile at frame 0 too (weights-free, same scene-load blackout as the LLM
            // prewarm above) — pocket-tts is the DEFAULT voice, so its shaders precompile here
            // instead of on the first zone entry / first clause. Its per-voice PrewarmKernels then
            // only warms real buffer/KV paths once weights are resident. Other TTS engines
            // (Kokoro/CosyVoice/Chatterbox) still prewarm on zone entry — a weights-free frame-0
            // pass for them is a follow-up (needs their own per-shader degenerate-dispatch list).
            if (anyPocket)
            {
                LLM.CurrentPhase = "kernel-prewarm";
                DrainNow(PocketTTSModeling.PocketTTS.PrewarmKernels());
                LLM.CurrentPhase = "idle";
            }
        }

        static void PrewarmModel(string id)
        {
            if (string.IsNullOrEmpty(id) || !prewarmedModels.Add(id)) return;
            var e = LLMRegistry.Find(id)?.prewarm?.Invoke();
            if (e == null) return;
            LLM.CurrentPhase = "kernel-prewarm";
            DrainNow(e);
            LLM.CurrentPhase = "idle";
        }

        // Runs a prewarm enumerator to completion within THIS frame. Recursive: coroutine-style
        // `yield return InnerEnumerator()` only executes the inner one under Unity's scheduler,
        // so a manual drain must descend into nested enumerators itself.
        static void DrainNow(IEnumerator e)
        {
            while (e.MoveNext())
                if (e.Current is IEnumerator nested) DrainNow(nested);
        }

        protected virtual void Update()
        {
            if (state == NPCState.Idle && PlayerReady && Input.GetKeyDown(InteractKey))
                StartInteraction();

            // context bar: live conversation fill vs maxContextLength (smoothed window-side);
            // after a compaction the token count drops, so the bar glides back by itself
            if (state != NPCState.Idle && maxContextLength > 0 && llm != null)
                Window?.SetContextFill(ContextTokensNow() / (float)maxContextLength);

            // per-NPC Backend Tradeoff level: pushed to the (global) table while THIS NPC's dialogue is
            // the active one, so opening a conversation adopts that NPC's level even if another NPC
            // left a different one behind. Comparing against the table needs no "applied" bookkeeping
            // of its own (the old float did, to trigger an auto-tuner re-probe that no longer exists),
            // and an inspector change mid-dialogue lands on the very next frame.
            if (state != NPCState.Idle && BackendTradeoffTable.Level != backendTradeoff)
                BackendTradeoffTable.Level = backendTradeoff;

            // Talk-animation watch. LLM+TTS with Kokoro: the gesture follows the AUDIO (ring
            // actually audible — the NPC keeps talking after the window closes too). Text-only:
            // it follows the token stream (state). Chatterbox/CosyVoice have no audio probe wired
            // yet, so (as before) they drive no gesture in TTS mode.
            bool audibleNow = speakReplies
                ? IsVoiceAudible
                : state == NPCState.TalkingInInteraction;
            // hysteresis: hold the talk gesture through short audio gaps (inter-chunk pauses, a
            // synth briefly behind playback) so the NPC doesn't flick to idle for a fraction of
            // a second between spoken chunks
            if (audibleNow) lastAudibleRealtime = Time.realtimeSinceStartup;
            bool talking = audibleNow || (speakReplies && talkAnimActive
                && Time.realtimeSinceStartup - lastAudibleRealtime < TALK_HOLD_SECONDS);
            if (talking != talkAnimActive)
            {
                talkAnimActive = talking;
                OnTalkingChanged(talking);
            }

            // model-RESIDENCY zone: slow-prefetch both models while the player walks up (both
            // weight streams spread over ~slowPrefetchSeconds), hold them on the GPU while the
            // player is inside (closing the chat releases nothing), unload on wander-off
            // (never mid-dialogue). Residency is the zone's job alone — history modes only
            // decide what happens to the CONVERSATION, never to the weights.
            // Gated on prewarmDone: entering the zone seconds after scene start must not pay the
            // COLD construction path (tokenizer file read + kernel lookups ≈ a 40 ms hitch) —
            // the zone simply arms a frame or two later, once the prewarm coroutine finished.
            if (usePrefetchZone && playerZoneT != null && prewarmDone)
            {
                bool inside = IsPlayerInsideZone(playerZoneT.position);
                if (inside && !inPrefetchZone)
                {
                    // Adopt THIS NPC's level before the streams start (fix 2026-07-28). The push below
                    // is gated on leaving Idle, but the whole point of the walk-up is that we are still
                    // Idle — so every prefetch ran on whatever level was left over, which on a fresh
                    // play is the static default (Balanced). Visible in the 2026-07-28 log on the
                    // reference 1650: `SLOW prefetch started — 1009 MB at 2.1 MB/frame` is Balanced's
                    // 16.8/8, then `BOOSTED to max budget — 8.4 MB/frame` is Smooth's — the same load
                    // changing budget mid-flight because the NPC woke up. A weak machine was streaming
                    // at twice its configured rate during precisely the seconds the dial exists to
                    // protect. Same last-writer-wins semantics as the conversation push: the NPC the
                    // player is walking toward is the one about to be talked to.
                    if (BackendTradeoffTable.Level != backendTradeoff)
                        BackendTradeoffTable.Level = backendTradeoff;
                    kkVoice?.SlowPrefetchNow(slowPrefetchSeconds);
                    kkVoice?.PrewarmKernels();   // shader compiles hide in the walk-up too
                    cvVoice?.SlowPrefetchNow(slowPrefetchSeconds);
                    pkVoice?.SlowPrefetchNow(slowPrefetchSeconds);
                    pkVoice?.PrewarmKernels();
                    BeginLlmSlowPrefetch();      // the LLM stream shares the walk-up window
                }
                else if (!inside && inPrefetchZone && state == NPCState.Idle)
                {
                    kkVoice?.DefetchNow();
                    cvVoice?.DefetchNow();
                    pkVoice?.DefetchNow();
                    // The LLM release defers past any in-flight background compaction and
                    // conversation-KV save (the compact must land, the save's GPU readbacks
                    // need the buffers alive); ResumeFromCompact keeps the transcript
                    // either way — reopen restores from disk or re-prefills.
                    StartCoroutine(ReleaseLlmAfterKvSave());
                }
                inPrefetchZone = inside;
            }

            if (state != NPCState.Idle && Input.GetKeyDown(KeyCode.Escape))
                CloseInteraction();
        }

        // Talk-gesture hysteresis: when the voice was last audible. Bridges short audio gaps
        // (inter-chunk pauses, a synth briefly behind playback) so the talk animation doesn't
        // flick to idle for a fraction of a second between spoken chunks.
        float lastAudibleRealtime = -999f;
        const float TALK_HOLD_SECONDS = 0.45f;

        // Scene-start prewarm completion flag — the prefetch zone waits for it (cold Acquire hitches).
        private bool prewarmDone;
        private IEnumerator RunPrewarm(IEnumerator job)
        {
            yield return job;
            prewarmDone = true;
        }

        // ---------------------------------------------------------------- voice / model setup

        /// <summary>
        /// Builds this NPC's voice component (per engine) on the NPC's GameObject — constructing
        /// it early starts/permits the TTS weight stream so the first spoken reply doesn't pay the
        /// load. Kokoro is load-on-approach (prefetch zone / talk trigger starts the 155 MB stream).
        /// </summary>
        protected void EnsureVoice()
        {
            switch (EffectiveTtsModel)
            {
                case TtsModel.PocketTTS:
                    pkVoice = GetComponent<PocketTTSModeling.PocketTTSVoice>();
                    if (pkVoice == null) pkVoice = gameObject.AddComponent<PocketTTSModeling.PocketTTSVoice>();
                    pkVoice.pitch = voicePitch;
                    pkVoice.volume = voiceVolume;
                    pkVoice.clausePauseSeconds = clausePauseSeconds;
                    pkVoice.replyTailSeconds = replyTailSeconds;
                    pkVoice.clausesPerChunk = clausesPerChunk;
                    // Tier-driven, not authored per NPC (2026-07-27): both are statements about the
                    // GPU, not about this character, and PocketTTSVoice used to learn them per device
                    // with an escalation ladder that cost an audible gap per rung. Pushed from the
                    // NPC's OWN row rather than BackendTradeoffTable.Current, because the static Level
                    // only becomes this NPC's while its dialogue is open — EnsureVoice runs earlier.
                    var tier = BackendTradeoffTable.At(backendTradeoff);
                    pkVoice.prebufferSeconds = tier.ttsPrebufferSeconds;
                    pkVoice.streamChunkFrames = tier.ttsStreamChunkFrames;
                    pkVoice.voiceName = ttsVoice;   // baked voices/<name> (default "jean")
                    if (pkVoice.clonedVoiceClip != clonedVoiceClip)
                        pkVoice.SetClonedVoice(clonedVoiceClip);   // clone-from-clip (cached) overrides baked
                    pkVoice.weightsPath = ttsQuantization == LLMQuant.FP16
                        ? PocketTTSModeling.PocketTTSConfig.WEIGHTS_DIR_FP16
                        : PocketTTSModeling.PocketTTSConfig.WEIGHTS_DIR_INT8;
                    pkVoice.loadOnStart = false;    // load-on-approach — the zone/trigger drives the stream
                    pkVoice.OnClauseSpoken -= OnClauseSpokenHandler;   // audio-synced text reveal (as Kokoro)
                    pkVoice.OnClauseSpoken += OnClauseSpokenHandler;
                    break;
                case TtsModel.Kokoro:
                    kkVoice = GetComponent<KokoroVoice>();
                    if (kkVoice == null) kkVoice = gameObject.AddComponent<KokoroVoice>();
                    kkVoice.streaming = true;
                    kkVoice.pitch = voicePitch;
                    kkVoice.volume = voiceVolume;
                    kkVoice.voiceName = ttsVoice;
                    kkVoice.clausesPerChunk = clausesPerChunk;
                    kkVoice.loadOnStart = false;   // load-on-approach, see Update / OnPlayerContact
                    kkVoice.OnClauseSpoken -= OnClauseSpokenHandler;   // audio-synced text reveal
                    kkVoice.OnClauseSpoken += OnClauseSpokenHandler;
                    break;
                case TtsModel.CosyVoice3:
                    cvVoice = GetComponent<CosyVoiceModeling.CosyVoiceVoice>();
                    if (cvVoice == null) cvVoice = gameObject.AddComponent<CosyVoiceModeling.CosyVoiceVoice>();
                    cvVoice.pitch = voicePitch;
                    cvVoice.volume = voiceVolume;
                    cvVoice.voiceName = ttsVoice;   // unknown names fall back inside CosyVoiceTTS (warned)
                    cvVoice.clausesPerChunk = clausesPerChunk;
                    cvVoice.loadOnStart = false;    // load-on-approach — the zone/trigger drives the stream
                    // streaming synth currently runs ~2.9x real-time on a 4060 (pre-int8) — a
                    // generous prebuffer turns that into one early gap instead of mid-word stutter
                    cvVoice.prebufferSeconds = 2.5f;
                    break;
                default:
                    voice = GetComponent<ChatterboxVoice>();
                    if (voice == null) voice = gameObject.AddComponent<ChatterboxVoice>();
                    voice.streaming = true;
                    voice.pitch = voicePitch;
                    voice.volume = voiceVolume;
                    voice.voiceName = ttsVoice;   // applied in ChatterboxVoice.Start (TTS is lazy-built)
                    voice.quantization = ttsQuantization;
                    voice.clausesPerChunk = clausesPerChunk;
                    break;
            }
            var src = GetComponent<AudioSource>();
            if (src != null) ConfigureVoiceAudioSource(src);
        }

        // Acquire is cheap; weights stream to the GPU over the next frames. The instance comes
        // from LLMPool, so NPCs sharing a model (same id/quant/kv) share ONE stream + ONE VRAM
        // copy — walking between two such NPCs never double-loads.
        // Standard KV pairing from the benchmark matrix: fp16 weights → fp16 KV, int8/int4 → int8 KV.
        // ResumeFromCompact allocates the KV LARGER than the trigger threshold so LLM.Compact's
        // own forward pass has room to run once the conversation reaches maxContextLength.
        const int COMPACT_HEADROOM = 8192;

        protected void EnsureLlm()
        {
            if (llm != null) return;
            KVQuant kv = quantization == LLMQuant.FP16 ? KVQuant.FP16 : KVQuant.INT8;
            int cap = historyMode == HistoryMode.ResumeFromCompact ? maxContextLength + COMPACT_HEADROOM : maxContextLength;
            llm = LLMPool.Acquire(model, quantization, kv, cap);
            llm.DiskKVCache = cacheKVCache;   // per-NPC toggle: system-prompt + conversation KV disk cache
            llm.CacheOwnerKey = ConversationKvKey();   // one cache file per NPC, overwritten when its prompt changes
            chatLive = false;   // we held no reference — whatever KV it carries isn't OUR conversation
        }

        /// <summary>Drops this NPC's pool reference; the GPU buffers actually free only when the
        /// last sharer lets go (the finalizer can't call Unity APIs safely).</summary>
        protected void ReleaseLlm(bool collectGarbage = false)
        {
            chatLive = false;
            if (llm == null) return;
            LLMPool.Release(llm);
            llm = null;
            if (collectGarbage)
                StartCoroutine(CollectGarbageIncremental());
        }

        // The legacy LLM loaders sample ONE global per-frame budget (LLM.UploadBudgetBytes) live
        // each frame — until WS-F gives them per-instance budgets, this governor holds that global
        // at the tier's SLOW rate for the whole time the player is merely Idle inside the zone,
        // and restores the full-speed budget on the exit edges: the dialogue opening (state
        // leaves Idle), the model finishing, or the zone releasing it. That restore IS the boost —
        // the zone itself never boosts (user 2026-07-30: "while in the zone it gets only slow
        // prefetched"; the previous version retargeted the budget to finish the WHOLE model inside
        // slowPrefetchSeconds — ~5.6 MB/frame for a 1 GB model in a 3 s window, nearly full rate —
        // and then boosted outright when the window elapsed, with the player just standing there).
        // BOTH ends of the rate come from the dial (BackendTradeoffTable): SlowFetchBytesPerFrame
        // while idle, FetchBytesPerFrame on the edges. This used to latch the full-speed baseline
        // into a static on first use, because with overlapping zones two governors run concurrently
        // and the second must not adopt the first one's SLOWED value as "full speed" — reading the
        // row instead makes that class of bug impossible (2026-07-26): the dial is the truth
        // regardless of how many governors are mid-flight, or which of them ran first.
        private Coroutine llmSlowJob;

        // Entry point for the zone: seeds a SLOW budget BEFORE the Acquire so the loader's own
        // "[GPU] ... SLOW prefetch started" line tells the truth from the first frame (the exact
        // remaining/window rate takes over on the next frames once totals are known).
        private void BeginLlmSlowPrefetch()
        {
            bool fresh = llm == null;
            if (fresh) LLM.UploadBudgetBytes = BackendTradeoffTable.SlowFetchBytesPerFrame;
            EnsureLlm();
            if (llmSlowJob != null) StopCoroutine(llmSlowJob);
            llmSlowJob = StartCoroutine(LlmSlowPrefetch());
        }

        private IEnumerator LlmSlowPrefetch()
        {
            bool announced = false;
            while (llm != null && !llm.IsReady && state == NPCState.Idle)
            {
                // Shared global budget: while ANY dialogue is open — not this NPC's (state would
                // not be Idle) but another one boosting the pooled instance — its boost owns the
                // budget; an idle zone must not fight it back down to the slow rate every frame.
                if (!AnyConversationOpen)
                {
                    long remaining = llm.TotalWeightBytes - llm.UploadedWeightBytes;
                    if (remaining > 0)
                    {
                        LLM.UploadBudgetBytes = BackendTradeoffTable.SlowFetchBytesPerFrame;
                        if (!announced)
                        {
                            ResidencyLog.Budget(llm.WeightsLabel, LLM.UploadBudgetBytes, remaining);
                            announced = true;
                        }
                    }
                }
                yield return null;
            }
            // boost/restore: dialogue opened, model finished, or the zone released it. Announce it
            // only when there is actually something left to stream — completion has its own
            // "resident" line.
            if (llm != null && !llm.IsReady)
                ResidencyLog.Budget(llm.WeightsLabel, BackendTradeoffTable.FetchBytesPerFrame,
                                    System.Math.Max(0, llm.TotalWeightBytes - llm.UploadedWeightBytes));
            LLM.UploadBudgetBytes = BackendTradeoffTable.FetchBytesPerFrame;
            llmSlowJob = null;
        }

        // ---------------------------------------------------------------- interaction flow

        // Which NPCs currently have their dialogue open. A SET rather than a counter so a double
        // open, a disable mid-conversation or a destroyed NPC can never leak the "someone is
        // talking" state — the thing <see cref="ConversationAudioDucker"/> hangs the world-audio
        // duck off, and a stuck true would leave the game permanently quiet.
        static readonly HashSet<NPCChatBase> conversing = new HashSet<NPCChatBase>();

        /// <summary>True while ANY NPC in the scene has its dialogue open — from the moment the
        /// window starts opening until it has fully closed, replies and silences alike. Environments
        /// use it to step out of the NPC's way while it speaks (see
        /// <see cref="ConversationAudioDucker"/>, which ducks every non-conversation sound).</summary>
        public static bool AnyConversationOpen => conversing.Count > 0;

        /// <summary>Where world audio should sit right now: the QUIETEST
        /// <c>worldAudioWhileTalking</c> among the NPCs currently in conversation, or 1 (untouched)
        /// when nobody is talking. The minimum — not the last one to open — so overlapping dialogues
        /// cannot let a permissive NPC undo a strict one's ducking.</summary>
        public static float WorldAudioTarget
        {
            get
            {
                float t = 1f;
                foreach (var npc in conversing)
                    if (npc != null && npc.worldAudioWhileTalking < t) t = npc.worldAudioWhileTalking;
                return t;
            }
        }

        public void StartInteraction()
        {
            dialogueEpoch++;
            state = NPCState.PreparingForInteraction;
            conversing.Add(this);
            // Prefetch policy (user 2026-07-30): entering the conversation IS the boost edge —
            // for EVERY stream, not just the LLM's. The LLM governor exits on the state flip
            // above but only on its next tick, so set the global budget NOW: the voice boosts
            // below copy it (ModelBase.BoostFetch) and any remaining LLM bytes run at the full
            // rate from this very frame. Level first — a shared-window sibling can reach here
            // before any zone/contact adoption ran for this NPC.
            if (BackendTradeoffTable.Level != backendTradeoff)
                BackendTradeoffTable.Level = backendTradeoff;
            LLM.UploadBudgetBytes = BackendTradeoffTable.FetchBytesPerFrame;
            kkVoice?.BoostPrefetchNow();
            cvVoice?.BoostPrefetchNow();
            pkVoice?.BoostPrefetchNow();
            if (interactPrompt != null) interactPrompt.Hide();
            OnInteractionStarted();
            dialogueCoroutine = StartCoroutine(OpenConversation());
        }

        /// <summary>
        /// Opens the chat window after the camera settles and brings the conversation up per
        /// <see cref="historyMode"/>:
        ///   ResetEveryTime          — fresh InitializeChat(system_prompt) every time (the system
        ///                             prompt itself still disk-caches inside InitializeChat).
        ///   ResumeFromCompact       — tier (a): the llm instance is alive with a live KV → reuse
        ///                             it untouched (no re-init, no history clear, instant);
        ///                             tier (b): the model was released → with cacheKVCache the
        ///                             WHOLE conversation state (KV + sampler state + transcript)
        ///                             is restored from disk (skips the prefill entirely); on a
        ///                             miss/mismatch it falls back to re-prefilling the recorded
        ///                             conversation (system prompt + all turns) through the
        ///                             normal chunked prefill. Correct, but pays the prefill.
        ///                             The state being resumed may be a background-COMPACTED one:
        ///                             when a chat reaches Max Context Length,
        ///                             CompactConversationRoutine collapses the history to
        ///                             [system prompt + ## MEMORY + compact], so every tier below
        ///                             stays short (live KV, disk KV and the re-prefill all carry
        ///                             the compacted prefix). Reopening mid-compaction WAITS on it
        ///                             behind a "Compacting…" pulse (input stays blocked).
        /// </summary>
        protected IEnumerator OpenConversation()
        {
            int epoch = dialogueEpoch;
            yield return new WaitForSeconds(DialogueOpenDelay);
            if (epoch != dialogueEpoch) yield break;   // closed while the camera settled

            var w = Window;
            w.Open();
            // several NPCs share the one chat window — stamp THIS NPC's name every interaction
            w.SetTitle(NpcName);
            w.Clear();   // a straggler bubble from a just-canceled reply must not survive into this session
            w.SetInfoText("");
            // model still loading: Send pulses dots and stays disabled, but the input field is
            // live so the first question can be typed while the weights stream in
            w.SetSendLoading(true);
            w.InputField.ActivateInputField();

            EnsureLlm();
            // an in-flight background compaction on THIS model instance is AWAITED, never killed
            // (user rule: once started, the compact must land — and two coroutines forwarding one
            // model would corrupt the KV). The player cannot talk meanwhile: state is still
            // PreparingForInteraction so AskNPC rejects sends, and the bubble that normally pulses
            // "Thinking…" pulses "Compacting…" instead until the routine finishes.
            if (compactingNpc != null && compactingNpc.llm == llm)
                yield return ShowCompactingUntilDone(w);
            // a manual reset is tearing this conversation down and re-initializing the model on the
            // bare system prompt — let it land before we init/restore onto the same instance. Its
            // InitializeChat holds the model's Busy guard, so ours would be REFUSED, and silently
            // (LLM.Guarded warns and yield-breaks). Bounded: the reset is itself bounded.
            while (resetRoutine != null) yield return null;
            llm.DiskKVCache = cacheKVCache;   // re-assert (a compaction/resume prefill clears it temporarily)
            llm.CacheOwnerKey = ConversationKvKey();   // pooled instance: claim the cache names for THIS NPC

            // a background conversation-KV save still reading THIS model's GPU state must finish
            // before anything resets/forwards it again (the SSM snapshot would tear mid-read) —
            // and before we try to restore the very file it is writing. Saves on other instances
            // don't block us (per-instance gate, audit #9).
            while (KvSaveInFlightFor(llm)) yield return null;
            if (epoch != dialogueEpoch) yield break;   // closed during the load/compact/save waits

            if (historyMode == HistoryMode.ResetEveryTime)
            {
                transcript.Clear();   // fresh persona every opening (also covers runtime mode switches)
                compactSummary = null;
            }

            if (historyMode != HistoryMode.ResetEveryTime && chatLive
                && LLMPool.OwnsConversation(llm, this))
            {
                // tier (a): OUR conversation KV is still on the shared GPU instance — just keep
                // talking. (Another NPC chatting on the pooled model in between steals the KV;
                // the ownership check routes us to the restore/re-prefill tiers below instead.)
            }
            else
            {
                bool restored = false;
                if (historyMode != HistoryMode.ResetEveryTime && cacheKVCache)
                {
                    // tier (b) fast path: pull the whole conversation state back from disk
                    // (KV/SSM + penalties + open-turn flag + transcript) instead of re-prefilling.
                    // AcceptRestoredTranscript vetoes the file when our in-memory transcript is
                    // fuller than the saved one (an interrupted reply landed after the last save)
                    // — what the player saw is ground truth. Gemma3 doesn't implement the API yet
                    // (base no-op reports false) so its NPCs always take the fallback below.
                    yield return llm.TryRestoreConversationKV(ConversationKvKey(), ok => restored = ok,
                        EffectiveSystemPrompt, AcceptRestoredTranscript);
                }
                if (!restored)
                {
                    // Waits for the weight stream, warms the kernels and prefills the prompt — all
                    // budgeted per frame, so the game keeps rendering smoothly behind the dialogue.
                    // tier (b) resume folds the transcript into the prompt; InitializeChat truncates
                    // the encode at 2048 tokens, so ancient history eventually falls off the front.
                    // A resume prompt is a one-shot prefix (it embeds the transcript), so DiskKVCache
                    // is dropped for the call — otherwise InitializeChat's system-prompt cache would
                    // write one orphan file per transcript. The conversation file saved on close
                    // covers persistence instead.
                    // a compacted conversation can have an EMPTY transcript (the HISTORY block is
                    // the whole history) — the compact alone still forces the resume prefix
                    bool resume = historyMode != HistoryMode.ResetEveryTime
                        && (transcript.Count > 0 || !string.IsNullOrEmpty(compactSummary));
                    if (resume) llm.DiskKVCache = false;
                    yield return llm.InitializeChat(system_prompt: resume ? BuildResumePrompt() : EffectiveSystemPrompt);
                    llm.DiskKVCache = cacheKVCache;
                }
                chatLive = true;
                LLMPool.ClaimConversation(llm, this);   // the shared KV now carries OUR conversation
            }
            if (epoch != dialogueEpoch) yield break;   // closed during restore/prefill

            // (No repopulate here. RepaintTranscript below is the ONE painter — see the note on it.
            // Until 2026-07-28 this line called RepopulateWindow() as well and every reopen drew the
            // whole history TWICE. Painting after the on-open compaction is also the correct order:
            // a compact that lands here changes what should be visible.)

            // Context-window state now that the conversation KV is live (ResumeFromCompact only):
            // a state restored ABOVE the trigger means a previous compaction never landed (game
            // stopped mid-compact) — compact it now, before the player talks, behind the
            // "Compacting…" pulse. Normal live triggering happens after each reply.
            WarnIfPrefixOverBudget();
            if (historyMode == HistoryMode.ResumeFromCompact && compactRoutine == null && ContextFull()
                && HasCompactableHistory())
            {
                compactRoutine = StartCoroutine(CompactConversationRoutine());
                yield return ShowCompactingUntilDone(w);
            }

            if (epoch != dialogueEpoch) yield break;   // closed during the on-open compaction
            RepaintTranscript(w);
            w.SetInfoText("");
            w.SetSendLoading(false);
            if (w.SendButton != null) w.SendButton.interactable = true;
            state = NPCState.WaitingInInteraction;
            if (!w.InputField.isFocused)
                w.InputField.ActivateInputField();
            dialogueCoroutine = null;
        }

        /// <summary>Called by the Send button / submitting the input field.</summary>
        public void AskNPC()
        {
            var w = Window;
            if (w == null || w.InputField == null || string.IsNullOrWhiteSpace(w.InputField.text))
                return;
            if (state != NPCState.WaitingInInteraction && state != NPCState.TalkingInInteraction)
                return;
            // compacting owns the model / interrupt already queued / a manual reset is demolishing the
            // conversation this line would have been part of
            if (compactRoutine != null || interruptPending || resetRoutine != null)
                return;
            // the model is mid-tool-call and owns the turn: either the choice panel is up, or the reply
            // that ended in a call is still being spoken and the panel is about to replace this very
            // input row. A user turn slipped in between would orphan the pending call.
            if (toolQuestionOpen || awaitingToolDispatch)
                return;

            string question = w.InputField.text;
            w.InputField.text = "";
            w.InputField.ActivateInputField();   // keep the caret in the field after every send

            // Sending while the previous reply is still GENERATING or still being SPOKEN:
            // cancel the generation at a token boundary (the KV keeps the truncated turn exactly
            // as if the model had stopped there), fade the voice to silence, THEN land the new
            // question as the next turn.
            if (dialogueCoroutine != null || VoicesAudible())
            {
                interruptPending = true;
                StartCoroutine(InterruptThenAsk(question, w));
                return;
            }

            PrepareForNextReply(w);   // settle a leftover bubble BEFORE the user line lands
            w.AddMessage("You", question);
            dialogueCoroutine = StartCoroutine(Talk(question));
        }

        /// <summary>Sends a prompt WITHOUT echoing it as a visible player line — for scripted
        /// events (handing items over, world triggers). The reply streams exactly like a typed
        /// question and the prompt still enters the conversation history/KV as a player turn.</summary>
        public void AskNPCSilent(string prompt)
        {
            var w = Window;
            if (w == null || string.IsNullOrWhiteSpace(prompt) || state != NPCState.WaitingInInteraction)
                return;
            // scripted events obey the same gates as the player — a Talk launched over a running
            // compaction/interrupt/reply would double-forward the model (audit #3); an open tool
            // question owns the turn the same way
            if (compactRoutine != null || interruptPending || dialogueCoroutine != null || VoicesAudible()
                || toolQuestionOpen || resetRoutine != null)
            {
                ConsoleMessage.Warning($"[NPC] {NpcName}: AskNPCSilent dropped — model busy (reply/compaction in flight).");
                return;
            }
            PrepareForNextReply(w);
            dialogueCoroutine = StartCoroutine(Talk(prompt));
        }

        // ---- mid-reply interruption (send-while-talking / leave-while-talking) ----------------
        // A reply is never StopCoroutine'd anymore: LLM.CancelChat() makes the decode loop exit
        // at the NEXT TOKEN BOUNDARY, so the KV holds the truncated turn EXACTLY as after a
        // natural stop token (the next Chat closes the turn with the template suffix as usual).
        bool replyCanceled;       // in-flight reply was canceled → skip voice flush + OnReplyFinished
        bool interruptPending;    // one interrupt-ask in flight at a time

        // IsSpeaking alone misses the drain window (synthesis done, ring/tail still audible —
        // pocket synthesizes ~6x realtime, so late in a reply this window is SECONDS long):
        // an ask there must still take the interrupt path, not talk over the playing audio.
        bool VoicesAudible() => (pkVoice != null && (pkVoice.IsSpeaking || pkVoice.IsAudioPlaying))
                             || (kkVoice != null && (kkVoice.IsSpeaking || kkVoice.IsAudioPlaying));

        // LLM-output sanitizer: glyphs outside the UI font (emoji, dingbats, CJK, symbols —
        // rendered as squares) also make the TTS produce strange sounds. Kept: ASCII, Latin-1 +
        // Latin Extended (covers Romanian and Western diacritics), and general punctuation
        // (curly quotes, dashes, ellipsis). Everything else — including emoji surrogates — drops.
        static bool RenderableChar(char c) =>
            c == '\n' || c == '\t'
            || (c >= 0x20 && c <= 0x7E)
            || (c >= 0xA0 && c <= 0x024F)
            || (c >= 0x2010 && c <= 0x2027);

        static string StripUnrenderable(string s)
        {
            if (string.IsNullOrEmpty(s)) return s ?? "";
            bool clean = true;
            for (int i = 0; i < s.Length; i++)
                if (!RenderableChar(s[i])) { clean = false; break; }
            if (clean) return s;   // the overwhelmingly common case — zero allocation
            var sb = new StringBuilder(s.Length);
            foreach (char c in s)
                if (RenderableChar(c)) sb.Append(c);
            return sb.ToString();
        }

        IEnumerator InterruptThenAsk(string question, INPCChatWindow w)
        {
            int epoch = dialogueEpoch;
            replyCanceled = true;
            // freeze the bubble at what was actually SPOKEN: kill the typewriter NOW and keep
            // every settle path (PrepareForNextReply / FinishSyncedReveal) from revealing the
            // rest — the cut reply stays partial on screen, exactly where the voice stopped.
            StopThinkingDots();
            StopRevealJob();
            bool wasSynced = revealActive;
            revealActive = false;
            // Mark NOW, before any wait: Talk clears activeTurn as it unwinds, and the unwind is what
            // we are about to wait on. bubbleLive is the frozen on-screen text, so a reopened window
            // replays what was heard, not what was generated.
            MarkReplyCutShort();
            llm?.CancelChat();
            FadeOutVoices();          // ~1 s smooth ramp to silence, never a hard cut
            float deadline = Time.unscaledTime + 10f;
            // wait for the reply coroutine to unwind at its token boundary (a post-reply
            // compaction extends the wait — compaction is never canceled), then for the fade
            while (dialogueCoroutine != null && epoch == dialogueEpoch
                   && (compactRoutine != null || Time.unscaledTime < deadline))
                yield return null;
            while (VoicesAudible() && epoch == dialogueEpoch && Time.unscaledTime < deadline + 3f)
                yield return null;
            // unconditional: also cancels a still-running leave-fade, whose terminal StopSpeaking
            // would otherwise land INSIDE the next reply and silently kill its first clause
            StopVoices();
            // a compaction may have started during the waits (limit-hitting reply) — it is never
            // canceled; the queued question lands right after it (audit #2)
            while (compactRoutine != null && epoch == dialogueEpoch) yield return null;
            interruptPending = false;
            if (dialogueCoroutine != null || state != NPCState.WaitingInInteraction || epoch != dialogueEpoch)
            {
                // This bail drops the player's question on the floor and, until 2026-07-28, said so
                // nowhere: the symptom is "I sent a message mid-reply and nothing ever happened", with
                // a silent log. Name the guard that fired and how long we waited, so the next report is
                // a diagnosis instead of three rounds of guessing.
                ConsoleMessage.Warning(
                    $"[NPC] {NpcName}: interrupt dropped the queued question — " +
                    $"dialogueCoroutine {(dialogueCoroutine != null ? "STILL RUNNING" : "null")}, " +
                    $"state {state} (need WaitingInInteraction), " +
                    $"epoch {epoch}->{dialogueEpoch}, " +
                    $"waited {Time.unscaledTime - (deadline - 10f):F1}s.");
                yield break;          // model stuck (fallback close handles it) or dialogue closed meanwhile
            }
            ConsoleMessage.Info($"[NPC] {NpcName}: interrupt handed over after " +
                                $"{Time.unscaledTime - (deadline - 10f):F1}s — asking the queued question.");
            // a reply cut before it SPOKE anything leaves its dots/Thinking bubble behind
            if (wasSynced && string.IsNullOrEmpty(spokenShown)) w.PopLastMessage();
            // ...and one that DID get some words out keeps them, now flagged as cut short. Drawn here,
            // after every wait above: the settle paths are all gated on revealActive (false since the
            // top of this method) so none of them can repaint over it from this point on.
            else if (!string.IsNullOrEmpty(bubbleLive))
            {
                w.PopLastMessage();
                w.AddMessage(NpcName, Bubble(WithCutMark(bubbleLive)));
                bubbleLive = null;
            }
            w.AddMessage("You", question);
            w.InputField.ActivateInputField();   // the fade window steals focus — hand it back
            dialogueCoroutine = StartCoroutine(Talk(question));
        }

        /// <summary>Runs when a reply finishes generating normally (never on an Escape
        /// interrupt) — scripted-event follow-ups hook here (e.g. coins after a thank-you).</summary>
        protected virtual void OnReplyFinished() { }

        // ---- audio-synced text reveal (LlmPlusTts + Kokoro): the window follows the VOICE, one
        // clause at a time (clauseRevealLead seconds early), instead of the raw token stream ----
        string spokenShown;          // what the window currently shows of the in-flight reply
        string pendingFullReply;     // full generated text (set once generation completes)
        bool revealActive;
        // The LIVE part (no exchangePrefix) of the reply bubble as last drawn — the text an interrupt
        // has to append CutMark to. Tracked separately from spokenShown because that one only exists
        // on the voice-paced path: without TTS the bubble follows the raw token stream instead, and an
        // interrupt there needs marking just the same.
        string bubbleLive;

        bool SyncedReveal => speakReplies && (kkVoice != null || pkVoice != null);

        // ---- thinking-dots placeholder + <think> stream filtering --------------------------
        Coroutine dotsJob;

        void StartThinkingDots(INPCChatWindow w, string label = null)
        {
            StopThinkingDots();
            dotsJob = StartCoroutine(ThinkingDots(w, label));
        }

        void StopThinkingDots()
        {
            if (dotsJob != null) { StopCoroutine(dotsJob); dotsJob = null; }
        }

        /// <summary>Redraw the surviving conversation into a freshly opened window.
        /// <para>The continue modes keep their transcript across a close, and with the disk cache
        /// across an app restart too — but the window was cleared unconditionally on open, so a
        /// resumed conversation LOOKED like a first meeting while the KV remembered everything the
        /// player could no longer see (user 2026-07-26). Turns already compacted away are
        /// deliberately absent: <c>transcript</c> is cleared when a compaction lands because the
        /// summary stands in for them, which is exactly "load the messages that were NOT compacted".
        /// ResetEveryTime is unaffected — it wipes the transcript on open, so there is nothing to
        /// draw, and that mode's whole point is to forget.</para>
        /// A <c>tool</c> turn draws only the NPC's side: its `user` text is a
        /// &lt;tool_response&gt; the player never typed (see <see cref="Turn"/>).</summary>
        void RepaintTranscript(INPCChatWindow w)
        {
            if (transcript.Count == 0 && string.IsNullOrEmpty(compactSummary)) return;
            // without this the conversation appears to begin mid-thought — the earlier turns exist,
            // they are just folded into the summary the model still reads
            if (!string.IsNullOrEmpty(compactSummary))
                w.AddMessage(NpcName, Bubble(StatusStyled("(earlier conversation summarized)")));
            foreach (var t in transcript)
            {
                if (!t.tool && !string.IsNullOrWhiteSpace(t.user)) w.AddMessage("You", t.user);
                string npcText = DisplayedNpcText(t);
                if (!string.IsNullOrWhiteSpace(npcText)) w.AddMessage(NpcName, Bubble(npcText));
            }
        }

        IEnumerator ThinkingDots(INPCChatWindow w, string labelOverride = null)
        {
            string[] frames = { "..", "...", "." };
            // with thinking enabled the model REALLY reasons behind these dots (until the final
            // </think>), so say so; plain models just get the typing pulse. A caller can name what
            // is actually happening instead — "Tool calling" while a <tool_call> streams.
            string label = labelOverride ?? (allowThinking ? "Thinking" : "");
            int i = 0;
            // self-terminates when the reply ends or the dialogue closes
            while (state == NPCState.TalkingInInteraction)
            {
                yield return new WaitForSecondsRealtime(0.4f);
                if (state != NPCState.TalkingInInteraction) break;
                w.PopLastMessage();
                w.AddMessage(NpcName, Bubble(StatusStyled(label + frames[i++ % 3])));
            }
            dotsJob = null;
        }

        /// <summary>What a rebuilt window should show for one transcript turn: the full reply normally,
        /// and for a turn the player cut short (send-while-talking or Leave-while-talking) only the part
        /// that was actually revealed, marked. Returns "" for a turn cut before it said anything — the
        /// player never heard it, so it gets no bubble, matching what the live interrupt does.
        /// <para>Never touches <c>Turn.npc</c>, which is the model's record and must stay whole.</para></summary>
        /// <summary>Record that the player cut this reply short — by sending over it or by leaving —
        /// so a rebuilt window shows what was HEARD plus the cut marker instead of paragraphs that were
        /// generated but never spoken. Covers the generating turn AND the one still draining through
        /// the voice.
        /// <para>Self-cancelling: if everything generated was also revealed, there is nothing to mark
        /// and this returns without touching the turn. That is what keeps it safe to call
        /// unconditionally on every close, including clean ones.</para></summary>
        void MarkReplyCutShort()
        {
            Turn t = activeTurn ?? drainTurn;
            if (t == null) return;
            string shown = bubbleLive ?? "";
            // t.npc is null while still generating (Talk sets it at the end) — then it is certainly cut.
            if (t.npc != null && shown.Length >= t.npc.Length) return;
            t.cut = true;
            t.npcShown = shown;
        }

        static string DisplayedNpcText(Turn t)
        {
            if (!t.cut) return t.npc;
            if (t.npcShown == null) return WithCutMark(t.npc);   // cut, but nothing captured
            return t.npcShown.Length == 0 ? "" : WithCutMark(t.npcShown);
        }

        static string ThinkStyled(string think)
            => $"<i><color=#9A9A9AB0>{think.Trim()}</color></i>\n";

        // status pulses (Thinking… / Compacting… / typing dots) are meta-text, not dialogue —
        // render them italic + slightly dimmed so they read as a different breed of text
        static string StatusStyled(string status)
            => $"<i><color=#CFCFCFC8>{status}</color></i>";

        // A call that was thrown away. Same italic meta-text breed as StatusStyled, but a DESATURATED
        // red so it reads as "this did not happen" at a glance without shouting like a real error would
        // (user 2026-07-28: "mai cu un rosu asa cu saturatie redusa"). Deliberately close in value to
        // the grey above so the two sit together in one bubble without one of them dominating.
        static string CanceledStyled(string status)
            => $"<i><color=#C68B8BC8>{status}</color></i>";

        // "the player talked over him here". Three ASCII periods, not U+2026: the demos use static
        // TMP font atlases and a missing glyph renders as a hollow box, which would read as a bug.
        // The blue is desaturated and slightly transparent on purpose — it has to be noticeable as a
        // different KIND of mark from dialogue without competing with the words it follows.
        const string CutMark = "<color=#7FA6D0C0>...</color>";

        /// <summary>Dialogue text plus the interrupted marker, unless it is already there (the freeze
        /// path can be reached twice — a queued ask during a fade, then the unwind).</summary>
        static string WithCutMark(string shown)
            => string.IsNullOrEmpty(shown) || shown.EndsWith(CutMark, System.StringComparison.Ordinal)
             ? shown : shown + CutMark;

        /// <summary>Split the accumulated reply into visible/think channels. Re-parses the FULL
        /// string every token (replies are short) so tags split across tokens just work; a
        /// trailing PARTIAL tag is held back from both channels until disambiguated.</summary>
        static void SplitThink(string full, out string visible, out string think)
        {
            var vis = new StringBuilder(full.Length);
            var thk = new StringBuilder();
            bool inThink = false;
            int i = 0;
            while (i < full.Length)
            {
                if (full[i] == '<')
                {
                    string tag = inThink ? Qwen3_5Modeling.Qwen3_5ChatTemplate.ThinkEndTag
                                        : Qwen3_5Modeling.Qwen3_5ChatTemplate.ThinkTag;
                    int remain = full.Length - i, match = 0;
                    while (match < tag.Length && match < remain && full[i + match] == tag[match]) match++;
                    if (match == tag.Length) { inThink = !inThink; i += tag.Length; continue; }
                    if (match == remain) break;   // trailing partial tag — hold it back
                }
                if (inThink) thk.Append(full[i]); else vis.Append(full[i]);
                i++;
            }
            visible = vis.ToString();
            think = thk.ToString();
        }

        // ---- Tools (enableAskUserQuestion + enableGiveTool + INPCToolProvider components) --------
        // TWO generic interactive tools cover every window interaction, and there are deliberately no
        // more: the model emits AskUserQuestion(question, options[]) — "decide something" — or
        // GiveTool(item, price?, quantity?) — "take this" — inside <tool_call> tags → the window shows
        // that tool's panel in place of the input row → the click returns as the <tool_response> result
        // → the model reacts in a fresh streamed turn. The JSON never reaches the screen or the TTS
        // (same channel-split treatment as <think>). An NPC's belt may grant either, both or neither.
        // INTERNAL tools come from INPCToolProvider components on the NPC (see the interface above):
        // same wire format, but the result goes straight back to the model with no window in
        // between, so an NPC can read world state mid-reply before deciding what to offer.

        // The block is Qwen3.5's OWN preamble — ~300 tokens for AskUserQuestion alone, ~400 with
        // Velmire's gear read, and every one of them is charged to the NPC's Max Context Length on
        // every turn. It stays verbatim rather than paraphrased because it is what the model was
        // trained on; a compact paraphrase measurably degraded tool calling.
        //
        // Qwen's text itself does NOT live here any more (moved 2026-07-28): it is owned by
        // Qwen3_5Modeling.Qwen3_5ChatTemplate, transcribed from the vendored chat_template.jinja and
        // guarded against drift by Qwen3_5ChatTemplateProbe. A gameplay behaviour class holding a
        // tokenizer-level contract as a hand-typed string was exactly how three prompt divergences
        // got into the training data before they were caught by eye on 2026-07-26. What DOES stay
        // here is the two bullets that are OURS (below), because they are ours — the finetune
        // depends on them and they must never be mistaken for something Qwen shipped.
        //
        // EVERYTHING that pushes the model to call lives HERE, in the prompt the author can read in the
        // inspector — that is a deliberate constraint (user 2026-07-25: "no hidden mechanisms; put it all
        // in the system prompt, the finetune fixes the rest"). An earlier engine-side rescue that quietly
        // asked the model to convert its own prose question into a call was removed for exactly that
        // reason, even though it worked.
        //
        // MEASURED, 2026-07-25, un-finetuned Qwen/Qwen3.5-0.8B, greedy, Velmire's persona, three
        // "I need a weapon" turns each: a compact JSON block called 1/3 times (and that one call copied
        // the block's own worked example verbatim, which is why no example lives here now), Qwen3.5's
        // official XML preamble 0/3, +"HARD RULE: never ask in prose" 0/3. So prompt-only elicitation on
        // a 0.8B is weak, and the two <IMPORTANT> lines below are the honest best effort until the v1.4
        // finetune lands — after which whatever format IT is trained on becomes the right thing to ship
        // here (the parser already accepts both).
        // Deliberately says NOTHING about how to PHRASE the question. This block is shared by every NPC
        // that gets the tool, so anything specific in here travels: the phrasing rule used to carry a
        // worked example naming one demo character, and Velmire's sword and shield ended up in Granny
        // Marla's and Anya's prompts (user 2026-07-26). Phrasing is a per-NPC concern — impersonal,
        // third-person question text matters for a VOICED NPC, where the spoken line and the on-screen
        // prompt are different channels, and much less for a text-only 2D demo — so it belongs in that
        // NPC's own descriptionAndRules, and in the finetune. Dropping the clause also returns ~45
        // tokens of every tool-bearing NPC's context.
        // The SHAPE matters as much as the words: this is byte-for-byte what `tool | tojson` emits
        // for this schema (compact one-line JSON, ", " and ": " separators). That is the only reason
        // the assembled block is byte-identical to the template's — a re-indented or differently
        // spaced schema would still be valid JSON and still be a divergence from the dataset.
        // PUBLIC like every other pinned wire string in this repo (see Qwen3_5ChatTemplate's consts):
        // the drift guard that compares it to dataset_creation/wire_format.py reads it by name.
        public const string AskUserQuestionSchema =
            "{\"type\": \"function\", \"function\": {\"name\": \"AskUserQuestion\", \"description\": " +
            "\"Put a choice to the player: shows the question with 2-4 clickable options and returns the one they pick. " +
            "Use it for any real decision - taking or refusing what you offer, picking a path, agreeing to a deal.\", " +
            "\"parameters\": {\"type\": \"object\", \"properties\": {" +
            "\"question\": {\"type\": \"string\"}, " +
            "\"options\": {\"type\": \"array\", \"items\": {\"type\": \"string\"}}}, " +
            "\"required\": [\"question\", \"options\"]}}}";

        // The SECOND (and last) interactive tool. Same pinning discipline as the schema above and for
        // the same reason: these bytes are what the model reads at inference, so they must be the bytes
        // the finetune was trained on. Mirrors dataset_creation/wire_format.py's GIVE_TOOL_SCHEMA
        // EXACTLY — that constant is the pin and this is the copy; NpcGiveToolProbe compares them
        // byte for byte and fails the build on a divergence, because a wrong prompt produces perfectly
        // plausible output and nothing else in Unity notices.
        // The 2026-07-31 corpus normalization folded the old catalog tools give_item (free handout,
        // {"status": "given"}) and sell_item (paid sale, item+quantity) into this ONE name and result
        // shape, corpus-wide: item required, price and quantity optional integers, and the result is
        // the player's own button press — exactly {"accepted": true} or {"accepted": false}.
        // Same one-line `tool | tojson` shape as AskUserQuestionSchema (", " and ": " separators):
        // re-indenting it would still be valid JSON and still be a divergence from the dataset.
        public const string GiveToolSchema =
            "{\"type\": \"function\", \"function\": {\"name\": \"GiveTool\", \"description\": " +
            "\"Hand the player an item - a gift, promised gear, or a sale at the price you name. " +
            "Shows the item (with the price, when there is one) with Accept and Decline buttons and " +
            "returns their decision. Call it to hand something over, or to table a final " +
            "take-it-or-leave-it offer.\", " +
            "\"parameters\": {\"type\": \"object\", \"properties\": {" +
            "\"item\": {\"type\": \"string\"}, " +
            "\"price\": {\"type\": \"integer\"}, " +
            "\"quantity\": {\"type\": \"integer\"}}, " +
            "\"required\": [\"item\"]}}}";

        // ---- the two <IMPORTANT> bullets that are OURS, not Qwen's -----------------------------
        // Everything else in the block — the header, the schema list, the call-format spec and the
        // four reminder bullets above these — is Qwen3.5's own preamble and lives in
        // Qwen3_5ChatTemplate, transcribed from the vendored chat_template.jinja. These two are
        // DeepUnity's, spliced in at the end of the reminder list (the one seam that moves none of
        // Qwen's bytes), and they are kept HERE precisely so the boundary stays visible: the v1.4
        // finetune is trained on them, so they must never be edited as if they were Qwen's, and
        // Qwen's must never be edited as if they were ours.
        // Measured cost of paraphrasing Qwen's part away: with a compact block that has no format
        // spec, the elicitation pass stops converting — the model just repeats its prose line
        // (2026-07-25). It costs ~66 tokens more than the compact block did; that is the price of
        // the offer actually reaching the player, so do not "optimize" it back out without
        // re-running the elicitation check.

        // OURS: the engine answers a call it cannot honour with {"error": …} (see RefuseToolCall)
        // and the model has to know what that means, or it retries the same call until the turn
        // dies. Says nothing about tools out loud — the player must never hear it. Unconditional:
        // any tool at all can come back refused.
        const string ErrorResultRule =
            "- If a result comes back as an error, do NOT call that function again: say what you can from what " +
            "you already know, in character, without mentioning the error or any function\n";

        // A rule of OUR OWN, appended to the reminder list only when the NPC actually has
        // AskUserQuestion: an NPC with nothing but provider reads would otherwise be ordered to call a
        // tool that is not in its <tools> block. The dataset follows the same rule (a sample's prompt
        // carries this line iff it declares the tool), so training and inference see one prefix.
        // It governs WHEN to call, never how to word the question — see AskUserQuestionSchema.
        const string AskUserQuestionRules =
            "- When your reply would put a decision to the player - taking or refusing something you offer, " +
            "choosing between paths, agreeing to a deal - you MUST call AskUserQuestion for that decision in the " +
            "same reply: say your line, then make the call. NEVER ask the player a question in prose and leave it " +
            "hanging, and NEVER answer on their behalf\n";

        INPCToolProvider[] toolProviders;
        /// <summary>Tool-provider components on this NPC, cached — <see cref="ToolsEnabled"/> is
        /// polled per streamed TOKEN, so this must never allocate.</summary>
        protected INPCToolProvider[] ToolProviders
        {
            get
            {
                if (toolProviders == null) toolProviders = GetComponents<INPCToolProvider>();
                return toolProviders;
            }
        }

        /// <summary>True when this NPC has ANY tool in its prompt. The streaming &lt;tool_call&gt;
        /// filter and the dispatch tail hang off THIS, not off <c>enableAskUserQuestion</c> alone —
        /// an NPC with only provider tools still emits calls that must not reach the screen.</summary>
        protected bool ToolsEnabled => enableAskUserQuestion || enableGiveTool || ToolProviders.Length > 0;

        /// <summary>The # Tools block this NPC's granted tools imply: Qwen3.5's canonical block for
        /// the schemas this NPC actually grants — asked for by name, not retyped here — with
        /// DeepUnity's own two reminder bullets appended inside its &lt;IMPORTANT&gt; list
        /// (<see cref="ErrorResultRule"/> always, <see cref="AskUserQuestionRules"/> only when that
        /// tool is granted, since an NPC with nothing but provider reads must not be ordered to call
        /// a tool that is missing from its own &lt;tools&gt;). Empty when the NPC has no tools.
        /// <para>This is a GENERATOR, not what runs: the text belongs in
        /// <c>descriptionAndRules</c>, written there once by the inspector button or the scene builder,
        /// so that everything the model reads is visible and editable in one field (user 2026-07-25).
        /// Nothing is appended behind the author's back at runtime.</para></summary>
        public string ComposeToolsBlock()
        {
            var schemas = new List<string>();
            // Interactive tools first, AskUserQuestion before GiveTool. That order is load-bearing the
            // same way the rule order below is: every one of the 279 corpus samples that declares BOTH
            // writes them in exactly this sequence, so it is not a free choice. A belt may carry
            // either, both or neither.
            if (enableAskUserQuestion) schemas.Add(AskUserQuestionSchema);
            if (enableGiveTool) schemas.Add(GiveToolSchema);
            foreach (var p in ToolProviders)
            {
                if (p.ToolSchemas == null) continue;
                foreach (string s in p.ToolSchemas)
                    if (!string.IsNullOrWhiteSpace(s)) schemas.Add(s.Trim());
            }
            if (schemas.Count == 0) return "";

            // Order is load-bearing: the error rule then the AskUserQuestion rule, both AFTER
            // Qwen's four bullets and before its </IMPORTANT>. That is the order all 300 finetuning
            // samples carry, so it is not a free choice.
            string ourRules = enableAskUserQuestion ? ErrorResultRule + AskUserQuestionRules : ErrorResultRule;
            return Qwen3_5Modeling.Qwen3_5ChatTemplate.RenderToolsBlock(schemas, ourRules);
        }

        /// <summary>Where a # Tools block ends inside an authored prompt: the first
        /// <c>&lt;/IMPORTANT&gt;</c> after a <c># Tools</c> heading. Used to REPLACE a stale block
        /// instead of stacking a second one. The template's own terminator — searching for anything
        /// else here would silently stop finding blocks the moment Qwen changed it.</summary>
        public const string ToolsBlockTerminator = Qwen3_5Modeling.Qwen3_5ChatTemplate.ReminderTerminator;

        /// <summary>Remove a previously written # Tools block from authored text, leaving the persona.
        /// Idempotent, and a no-op on text that never had one.</summary>
        public static string StripToolsBlock(string authored)
        {
            if (string.IsNullOrEmpty(authored)) return authored ?? "";
            int i = authored.IndexOf(Qwen3_5Modeling.Qwen3_5ChatTemplate.ToolsHeading, System.StringComparison.Ordinal);
            if (i < 0) return authored;
            int j = authored.IndexOf(ToolsBlockTerminator, i, System.StringComparison.Ordinal);
            if (j < 0) return authored;                     // half a block: leave it, don't guess
            return (authored.Substring(0, i) + authored.Substring(j + ToolsBlockTerminator.Length))
                   .TrimStart('\n', '\r', ' ');
        }

        /// <summary>The authored text with THIS NPC's current # Tools block joined to it, replacing any
        /// block already there. What the inspector button and the scene builders write into
        /// <c>descriptionAndRules</c> — the tools live in the field, visible, not injected at runtime.
        /// <para><paramref name="toolsFirst"/> defaults to TRUE because that is the CANONICAL order:
        /// Qwen3_5ChatTemplate's tools branch opens the system message, writes the block, and only
        /// then appends the persona after a blank line (vendored template L46-58), and all 300
        /// finetuning samples are written that way. Passing false puts the persona on top instead —
        /// ChatDemo3D does exactly that for Velmire, by request, and it is kept ON PURPOSE: it is
        /// more readable for a long persona. It is nonetheless a deliberate divergence from the
        /// canonical order and from the training data, not an alternative spelling of it, so flip the
        /// dataset too if it ever becomes the house style.</para></summary>
        public string WithToolsBlock(string authored, bool toolsFirst = true)
        {
            string persona = StripToolsBlock(authored).Trim('\n', '\r', ' ');
            string block = ComposeToolsBlock();
            if (block.Length == 0) return persona;
            // The blank line between them is the template's own (L57: '\n\n' + content), in both
            // orders — swapping which side is on top must not also change the separator.
            string sep = Qwen3_5Modeling.Qwen3_5ChatTemplate.SystemContentSeparator;
            return toolsFirst ? block + sep + persona : persona + sep + block;
        }

        /// <summary>What the model actually receives as its system prompt: the NAME heading composed
        /// from <see cref="NpcName"/>, then the authored text verbatim — tools, rules and persona
        /// alike, exactly as the inspector shows them. The compact rides underneath as ## MEMORY
        /// (see <see cref="BuildResumePrompt"/> / <see cref="EffectivePromptPreview"/>).
        /// <para>EVERY re-seed path goes through this (fresh init, resume prefix, compaction,
        /// conversation-KV key), so the prompt hash tracks edits to the field and correctly
        /// invalidates KV disk caches.</para></summary>
        protected string EffectiveSystemPrompt =>
            // WHO first, in a heading of its own (user 2026-07-25). It costs a handful of tokens and
            // duplicates a name the description usually repeats, but it puts the one fact the model
            // must never lose at the very top of the prompt — the position a compaction re-seed, a
            // long history and a truncated context all preserve longest.
            "## NAME\n" + NpcName + "\n\n" + descriptionAndRules;

        /// <summary>Editor-facing view of <see cref="EffectiveSystemPrompt"/> plus the compact, i.e. the
        /// EXACT text the model is seeded with. Exists because the inspector's System Prompt field is only
        /// the persona — roughly a third of what actually goes in — and there was no way to read the rest
        /// without stepping through code (user 2026-07-25). Rendered by NPCChatBaseEditor.</summary>
        public string EffectivePromptPreview
        {
            get
            {
                string p = EffectiveSystemPrompt;
                return string.IsNullOrEmpty(compactSummary)
                    ? p : p + "\n\n" + LLM.HISTORY_HEADING + "\n" + compactSummary;
            }
        }

        /// <summary>Split the (already think-stripped) reply into the visible channel and the
        /// machine &lt;tool_call&gt; channel. Same contract as <see cref="SplitThink"/>: re-parses the
        /// full string every token, holds a trailing partial tag back until disambiguated.</summary>
        static void SplitToolCall(string full, out string visible, out string tool)
        {
            var vis = new StringBuilder(full.Length);
            var tl = new StringBuilder();
            bool inCall = false;
            int i = 0;
            while (i < full.Length)
            {
                if (full[i] == '<')
                {
                    string tag = inCall ? Qwen3_5Modeling.Qwen3_5ChatTemplate.ToolCallEndTag
                                        : Qwen3_5Modeling.Qwen3_5ChatTemplate.ToolCallTag;
                    int remain = full.Length - i, match = 0;
                    while (match < tag.Length && match < remain && full[i + match] == tag[match]) match++;
                    if (match == tag.Length) { inCall = !inCall; i += tag.Length; continue; }
                    if (match == remain) break;   // trailing partial tag — hold it back
                }
                if (inCall) tl.Append(full[i]); else vis.Append(full[i]);
                i++;
            }
            visible = vis.ToString();
            tool = tl.ToString();
        }

        [System.Serializable] class ToolCallMsg { public string name; public ToolCallArgs arguments; }
        [System.Serializable] class ToolCallArgs { public string question; public string[] options; }
        [System.Serializable] class ToolPickResult { public string selected; }
        [System.Serializable] class StringArrayWrap { public string[] items; }

        /// <summary>A &lt;tool_call&gt; body parsed out of EITHER wire shape (see below), normalized:
        /// a name, a JSON arguments string for providers, the AskUserQuestion fields and the GiveTool
        /// ones.</summary>
        class ParsedToolCall
        {
            public string name;
            public string argsJson = "{}";
            public string question;
            public readonly List<string> options = new List<string>();
            // GiveTool. Read out of argsJson AFTER either branch built it, so both wire shapes are
            // served by one piece of code — the XML template renders every scalar as bare text inside
            // <parameter=…> while the JSON shape writes a real number, and neither is special-cased.
            public string item;
            public int? price;
            public int? quantity;
        }

        // TWO shapes are accepted, because the model and the finetune do not agree on one:
        //  (1) XML — what Qwen3.5's OWN chat template declares, and therefore what an un-finetuned
        //      Qwen3.5-0.8B actually emits (measured against Qwen/Qwen3.5-0.8B on 2026-07-25):
        //        <tool_call><function=NAME><parameter=KEY>\nvalue\n</parameter>…</function></tool_call>
        //      Array/object parameter values are rendered as JSON by that template.
        //  (2) JSON — the Qwen2.5/3-era Hermes style the SFT dataset (dataset_creation v1.3) uses:
        //        <tool_call>{"name": NAME, "arguments": {…}}</tool_call>
        // Accepting both means the demo works BEFORE the finetune lands and keeps working after,
        // whichever shape the trained model settles on — and a model that drifts between them mid
        // conversation never silently drops the player's choice.
        static readonly Regex FunctionTagRe = new Regex(@"<function\s*=\s*([^>\s]+)\s*>", RegexOptions.Compiled);
        static readonly Regex ParameterRe = new Regex(@"<parameter\s*=\s*([^>\s]+)\s*>(.*?)</parameter>",
                                                      RegexOptions.Compiled | RegexOptions.Singleline);

        static ParsedToolCall ParseToolCall(string body)
        {
            if (string.IsNullOrWhiteSpace(body)) return null;
            Match fn = FunctionTagRe.Match(body);
            if (fn.Success) return WithGiveArgs(ParseXmlToolCall(body, fn));

            string json = FirstJsonObject(body);
            if (json == null) return null;
            ToolCallMsg msg = null;
            try { msg = JsonUtility.FromJson<ToolCallMsg>(json); } catch { }
            if (msg == null || string.IsNullOrWhiteSpace(msg.name)) return null;
            var call = new ParsedToolCall { name = msg.name.Trim(), argsJson = ArgumentsJson(json) };
            if (msg.arguments != null)
            {
                call.question = msg.arguments.question;
                if (msg.arguments.options != null)
                    foreach (string o in msg.arguments.options) AddOption(call.options, o);
            }
            return WithGiveArgs(call);
        }

        /// <summary>GiveTool's three arguments, read off the normalized <c>argsJson</c> both wire
        /// shapes already produced. Done here rather than in either branch precisely so there is ONE
        /// reading of them: JsonUtility cannot tell an absent integer from a zero, and price 0 (a free
        /// handout the NPC still called a price) is a different offer from no price at all.</summary>
        static ParsedToolCall WithGiveArgs(ParsedToolCall call)
        {
            if (call == null) return null;
            call.item = ArgValue(call.argsJson, "item");
            call.price = ArgInt(call.argsJson, "price");
            call.quantity = ArgInt(call.argsJson, "quantity");
            return call;
        }

        /// <summary>Value of a TOP-LEVEL key in a flat arguments object as text, or null when the key
        /// is absent. Quoted and bare values both come back unquoted, because the XML template renders
        /// numbers as text and the JSON shape renders them as numbers.</summary>
        static string ArgValue(string argsJson, string key)
        {
            if (string.IsNullOrEmpty(argsJson)) return null;
            string needle = "\"" + key + "\"";
            int depth = 0, i = 0;
            while (i < argsJson.Length)
            {
                char c = argsJson[i];
                if (c == '"')
                {
                    if (depth == 1 && i + needle.Length <= argsJson.Length
                        && string.CompareOrdinal(argsJson, i, needle, 0, needle.Length) == 0)
                    {
                        int j = i + needle.Length;
                        while (j < argsJson.Length && char.IsWhiteSpace(argsJson[j])) j++;
                        if (j < argsJson.Length && argsJson[j] == ':') return ReadScalar(argsJson, j + 1);
                    }
                    // skip the whole literal, so a VALUE that happens to read "item" is never a key
                    i = SkipString(argsJson, i);
                    continue;
                }
                if (c == '{' || c == '[') depth++;
                else if (c == '}' || c == ']') depth--;
                i++;
            }
            return null;
        }

        /// <summary>Index just past the string literal opening at <paramref name="open"/>.</summary>
        static int SkipString(string s, int open)
        {
            for (int i = open + 1; i < s.Length; i++)
            {
                if (s[i] == '\\') { i++; continue; }
                if (s[i] == '"') return i + 1;
            }
            return s.Length;
        }

        /// <summary>One JSON scalar starting at <paramref name="at"/> (after the colon), unquoted and
        /// unescaped. Objects/arrays are not scalars and come back as their raw text — GiveTool has
        /// none, and a model that writes one gets it rejected by the int parse below.</summary>
        static string ReadScalar(string s, int at)
        {
            while (at < s.Length && char.IsWhiteSpace(s[at])) at++;
            if (at >= s.Length) return null;
            if (s[at] != '"')
            {
                int e = at;
                while (e < s.Length && s[e] != ',' && s[e] != '}' && s[e] != ']') e++;
                return s.Substring(at, e - at).Trim();
            }
            var sb = new StringBuilder();
            for (int i = at + 1; i < s.Length; i++)
            {
                char c = s[i];
                if (c == '"') break;
                if (c != '\\') { sb.Append(c); continue; }
                if (++i >= s.Length) break;
                char n = s[i];
                if (n == 'n') sb.Append('\n');
                else if (n == 't') sb.Append('\t');
                else if (n == 'r') { }
                else sb.Append(n);
            }
            return sb.ToString();
        }

        /// <summary>A whole-number argument, or null when absent/unreadable. Tolerant on purpose: a
        /// 0.8B writes "80 souls" or "80.0" often enough that failing the whole offer over it would
        /// cost the player a deal the NPC already agreed to, so the LEADING integer wins.</summary>
        static int? ArgInt(string argsJson, string key)
        {
            string raw = ArgValue(argsJson, key);
            if (string.IsNullOrWhiteSpace(raw)) return null;
            raw = raw.Trim();
            int i = 0;
            bool neg = raw[0] == '-';
            if (neg || raw[0] == '+') i = 1;
            int start = i;
            while (i < raw.Length && raw[i] >= '0' && raw[i] <= '9') i++;
            if (i == start) return null;
            if (!int.TryParse(raw.Substring(start, i - start), out int v)) return null;
            return neg ? -v : v;
        }

        static ParsedToolCall ParseXmlToolCall(string body, Match fn)
        {
            var call = new ParsedToolCall { name = fn.Groups[1].Value.Trim() };
            var args = new StringBuilder("{");
            bool first = true;
            foreach (Match p in ParameterRe.Matches(body))
            {
                string key = p.Groups[1].Value.Trim();
                string val = p.Groups[2].Value.Trim();
                if (key.Length == 0) continue;
                if (!first) args.Append(", ");
                first = false;
                args.Append('"').Append(JsonEscape(key)).Append("\": ");
                // a value the template already rendered as JSON (arrays, objects) rides through
                // untouched; anything else is a plain string
                if (val.StartsWith("[") || val.StartsWith("{")) args.Append(val);
                else args.Append('"').Append(JsonEscape(val)).Append('"');

                if (key.Equals("question", System.StringComparison.OrdinalIgnoreCase)) call.question = val;
                else if (key.Equals("options", System.StringComparison.OrdinalIgnoreCase))
                    AppendOptions(call.options, val);
            }
            call.argsJson = args.Append('}').ToString();
            return call;
        }

        /// <summary>Options as the model wrote them: the template's own JSON array, or — when it
        /// free-hands the parameter — one per line, or comma-separated.</summary>
        static void AppendOptions(List<string> into, string val)
        {
            val = val.Trim();
            if (val.StartsWith("["))
            {
                try   // JsonUtility cannot parse a bare array — wrap it
                {
                    var w = JsonUtility.FromJson<StringArrayWrap>("{\"items\":" + val + "}");
                    if (w?.items != null && w.items.Length > 0)
                    {
                        foreach (string s in w.items) AddOption(into, s);
                        return;
                    }
                }
                catch { }
                val = val.Trim('[', ']');
            }
            char sep = val.IndexOf('\n') >= 0 ? '\n' : ',';
            foreach (string part in val.Split(sep)) AddOption(into, part.Trim('"', '\'', ',', ' ', '\t', '\r'));
        }

        static void AddOption(List<string> into, string s)
        {
            s = s?.Trim();
            if (!string.IsNullOrEmpty(s) && into.Count < 4) into.Add(s);
        }

        static string JsonEscape(string s)
        {
            var sb = new StringBuilder(s.Length + 8);
            foreach (char c in s)
            {
                if (c == '"' || c == '\\') sb.Append('\\').Append(c);
                else if (c == '\n') sb.Append("\\n");
                else if (c == '\t') sb.Append("\\t");
                else if (c != '\r') sb.Append(c);
            }
            return sb.ToString();
        }

        // choice panel is up — typed sends are dropped until the pick lands (the model is mid-tool-call;
        // a user turn slipped in between would orphan the pending question)
        bool toolQuestionOpen;
        // the reply ended in a call and the NPC is still SAYING the line that came before it: the panel
        // must not appear over a talking NPC, and a send in between would orphan the call
        bool awaitingToolDispatch;

        /// <summary>Let the spoken part of a reply finish — voice AND the word-by-word reveal — and only
        /// then dispatch the call it ended with. This is the user-specified order: the NPC says its line
        /// to the end, and the question replaces the input row afterwards. Bounded, because a stalled
        /// voice must never park the dialogue.</summary>
        IEnumerator DispatchAfterSpeaking(string toolJson, int epoch)
        {
            awaitingToolDispatch = true;
            try
            {
                float deadline = Time.unscaledTime + 30f;
                while (epoch == dialogueEpoch && Time.unscaledTime < deadline
                       && (revealActive || revealJob != null || revealQueue.Count > 0 || VoicesAudible()))
                    yield return null;
            }
            finally { awaitingToolDispatch = false; }
            // a close, an interrupt-then-ask, or a compaction that started meanwhile owns the turn now
            if (epoch != dialogueEpoch || state != NPCState.WaitingInInteraction
                || dialogueCoroutine != null || compactRoutine != null)
                yield break;
            TryDispatchToolCall(toolJson, epoch);
        }

        // ---- one bubble per EXCHANGE -----------------------------------------------------------
        // A tool call splits what the player experiences as one reply into several decodes (speak → call →
        // result → speak → call → …). Each of those used to open its own bubble, so the NPC's name was
        // repeated above every fragment. Everything now renders into the SAME bubble: `exchangePrefix`
        // holds what the NPC has already committed to in this exchange (its earlier lines plus the
        // "Tool Called (…)" markers) and every render draws prefix + whatever is currently live.
        // Reset only when the PLAYER speaks — that is what starts a new exchange.
        string exchangePrefix = "";

        /// <summary>prefix + live, newline-joined — what the NPC's bubble should read right now.</summary>
        string Bubble(string live)
            => string.IsNullOrEmpty(exchangePrefix) ? live
             : string.IsNullOrEmpty(live) ? exchangePrefix
             : exchangePrefix + "\n" + live;

        /// <summary>The window's PERMANENT record that a call happened, appended INTO the current bubble
        /// (user spec: the call stays written, under the same NPC title as the line before it). Styled
        /// like the other meta lines, but unlike the pulses it is never popped.</summary>
        /// <param name="refused">The call was rejected before it ran (malformed, unknown tool, window
        /// without the popup). Say so in the line. Until 2026-07-28 a refused call rendered EXACTLY like
        /// an honoured one, so the player read "Tool Called (AskUserQuestion)" and waited for a choice
        /// panel that was never going to open — reported as "he made a tool call and I didn't see the
        /// option to choose, it was just called and that's it". The engine was recovering correctly
        /// underneath (the model gets an error result and answers in words); only the label lied.</param>
        void ShowToolCalledLine(INPCChatWindow w, string toolName, bool refused = false)
        {
            if (w == null) return;
            exchangePrefix = Bubble(refused ? CanceledStyled($"Tool Call Canceled ({toolName})")
                                            : StatusStyled($"Tool Called ({toolName})"));
            pendingFullReply = null;   // committed into the prefix; a later settle must not re-add it
            w.PopLastMessage();
            w.AddMessage(NpcName, exchangePrefix);
        }

        /// <summary>First balanced {...} object in <paramref name="s"/> (string-and-escape aware),
        /// or null — the model sometimes pads the tool JSON with newlines/prose.</summary>
        static string FirstJsonObject(string s)
        {
            int start = s.IndexOf('{');
            if (start < 0) return null;
            int depth = 0; bool inStr = false, esc = false;
            for (int i = start; i < s.Length; i++)
            {
                char c = s[i];
                if (esc) { esc = false; continue; }
                if (inStr) { if (c == '\\') esc = true; else if (c == '"') inStr = false; continue; }
                if (c == '"') inStr = true;
                else if (c == '{') depth++;
                else if (c == '}' && --depth == 0) return s.Substring(start, i - start + 1);
            }
            return null;
        }

        /// <summary>Raw JSON of a call's "arguments" sub-object — providers parse their own argument
        /// shape, and JsonUtility cannot hand back an untyped node. "{}" when the call has none.</summary>
        static string ArgumentsJson(string callJson)
        {
            int k = callJson.IndexOf("\"arguments\"", System.StringComparison.Ordinal);
            if (k < 0) return "{}";
            return FirstJsonObject(callJson.Substring(k)) ?? "{}";
        }

        /// <summary>Fires when the player picks an option, BEFORE the pick goes back to the model:
        /// <c>(question, pickedOption)</c>. This is where gated ACTIONS live — handing gear over,
        /// taking payment — so the value transfer is engine code reacting to a player choice, never
        /// something the model can do by itself. Subscribers must not block.</summary>
        public event System.Action<string, string> ToolQuestionAnswered;

        // ---- GiveTool's two extension hooks ----------------------------------------------------
        // GENERAL, not any one demo's: the base class owns the panel and the wire format, the host
        // game owns money and inventory. Same division of labour as ToolQuestionAnswered above — the
        // model proposes, the player disposes, and the ENGINE performs.

        /// <summary>Optional accept-gate: can the player take THIS offer at all? Returns false and the
        /// panel's Accept button is rendered DISABLED — that is the "not enough money" case, and the
        /// place to put "no room in the pack" or "already have one" too. Decline is never gated, so a
        /// refused offer still ends the exchange cleanly. Null (the default) = always acceptable.
        /// <para>Evaluated ONCE, when the panel opens. It is a PRESENTATION hint, not the transaction:
        /// the authoritative check belongs in the <see cref="ToolGiveAccepted"/> handler, which is
        /// where value actually changes hands — a gate that throws is reported and treated as "yes"
        /// for that reason.</para></summary>
        public System.Func<ToolGiveOffer, bool> ToolGiveAcceptGate { get; set; }

        /// <summary>Fires when the player ACCEPTS an offer, BEFORE the decision goes back to the model:
        /// take the payment, grant the item, update the HUD here. Declining fires nothing — the model
        /// simply reads {"accepted": false} and answers in character. Subscribers must not block.</summary>
        public event System.Action<ToolGiveOffer> ToolGiveAccepted;

        // The ONLY two results a GiveTool call can return, byte for byte what the corpus teaches
        // (dataset_creation validate.py: 'GiveTool result must be exactly {"accepted": true} or
        // {"accepted": false}'). Literals, not JsonUtility.ToJson — which writes {"accepted":true},
        // without the separator space every sample in the corpus has.
        const string ToolGiveAcceptedResult = "{\"accepted\": true}";
        const string ToolGiveDeclinedResult = "{\"accepted\": false}";

        /// <summary>The tool result a give decision sends back to the model. THE mapping — the window's
        /// Accept/Decline click goes nowhere else, and the headless probe asserts these bytes.</summary>
        public static string ToolGiveResult(bool accepted)
            => accepted ? ToolGiveAcceptedResult : ToolGiveDeclinedResult;

        /// <summary>Read a &lt;tool_call&gt; body as a GiveTool offer: true when it parses AND names an
        /// item (the schema's one required parameter), false when the model wrote something the panel
        /// cannot show. The seam <see cref="DispatchGiveTool"/> and the headless probe share, so what
        /// the probe exercises IS what the dialogue runs.</summary>
        public static bool TryReadGiveToolCall(string toolCallBody, out ToolGiveOffer offer)
        {
            offer = default;
            ParsedToolCall call = ParseToolCall(toolCallBody);
            if (call == null || string.IsNullOrWhiteSpace(call.item)) return false;
            offer = OfferOf(call);
            return true;
        }

        /// <summary>The offer a parsed call describes. ONE construction, shared by the dispatch and by
        /// <see cref="TryReadGiveToolCall"/>, so the probe cannot drift from the dialogue.</summary>
        static ToolGiveOffer OfferOf(ParsedToolCall call) => new ToolGiveOffer
        {
            item = call.item?.Trim(),
            price = call.price,
            quantity = call.quantity,
        };

        /// <summary>Internal (provider) reads already answered in the current player turn — see
        /// <c>maxToolReadsPerTurn</c> for why there is a cap at all and why the interactive tools are
        /// exempt.</summary>
        int internalToolCalls;

        /// <summary>An unhonourable call has already been refused in this exchange. ONE per exchange:
        /// the refusal is itself a &lt;tool_response&gt; the model answers, so a model that responds to
        /// "no" by calling again would ping-pong forever. The second failure ends the turn instead.</summary>
        bool toolRefusalSent;

        /// <summary>ANSWER a call the engine cannot honour instead of dropping it. Dropping it silently
        /// was the old behaviour and it dead-ends the exchange: decoding stopped at
        /// <c>&lt;/tool_call&gt;</c>, so the NPC has said its line, is waiting on a result that never
        /// comes, and the player is left staring at a half-finished thought — while the model, never
        /// told anything went wrong, cheerfully makes the same call next turn. So the failure goes back
        /// as a <c>{"error": …}</c> tool result: the model reads it and speaks in the SAME exchange,
        /// which is also the shape the finetune teaches (a refused read is answered from what the NPC
        /// already knows, and never retried).</summary>
        void RefuseToolCall(string toolName, string errorJson, string logMessage)
        {
            ConsoleMessage.Warning(logMessage);
            var w = Window;
            if (w == null) return;
            if (toolRefusalSent)
            {
                ConsoleMessage.Warning($"[NPC] {NpcName}: second unhonourable call in one exchange — " +
                                       "turn ended instead of refusing again (ping-pong guard).");
                StopThinkingDots();
                return;
            }
            toolRefusalSent = true;
            PrepareForNextReply(w);
            ShowToolCalledLine(w, toolName, refused: true);
            dialogueCoroutine = StartCoroutine(Talk(errorJson, asToolResult: true));
        }

        /// <summary>Parse + validate a completed &lt;tool_call&gt; body and either answer it in-engine
        /// (internal provider tool → result straight back to the model) or open an interactive panel:
        /// the choice popup (AskUserQuestion) or the offer popup (GiveTool). Malformed calls, unknown
        /// tool names and windows without the panel capability degrade to a console warning — the
        /// dialogue continues as if no call happened.</summary>
        void TryDispatchToolCall(string toolJson, int epoch)
        {
            ParsedToolCall call = ParseToolCall(toolJson);
            if (call == null || string.IsNullOrWhiteSpace(call.name))
            {
                // Also retryable: an unreadable call is a formatting slip, and the model gets one more
                // go (the ping-pong guard bounds it). Only tool_unavailable / unknown_tool /
                // read_limit_reached send it to words, because for those a retry cannot succeed.
                RefuseToolCall("tool", "{\"error\": \"malformed_call\", \"detail\": \"that tool call could not " +
                               "be read - write it again in the exact format from your instructions\"}",
                               $"[NPC] {NpcName}: malformed <tool_call>: {toolJson.Trim()}");
                return;
            }

            // The dialogue window handles EXACTLY TWO interactive tools — AskUserQuestion (a choice)
            // and GiveTool (an item). Everything else is an internal provider read.
            bool isAsk = "AskUserQuestion".Equals(call.name, System.StringComparison.OrdinalIgnoreCase);
            bool isGive = "GiveTool".Equals(call.name, System.StringComparison.OrdinalIgnoreCase);

            if (isGive)
            {
                DispatchGiveTool(call, toolJson, epoch);
                return;
            }

            // internal tools first: a world-state read the player never sees. The result feeds
            // straight into a fresh turn, so the model reads, then decides, in one breath.
            if (!isAsk)
            {
                string argsJson = call.argsJson;
                foreach (var p in ToolProviders)
                {
                    string result = p.TryHandleTool(call.name, argsJson);
                    if (result == null) continue;   // not this provider's tool
                    if (++internalToolCalls > maxToolReadsPerTurn)
                    {
                        RefuseToolCall(call.name,
                            "{\"error\": \"read_limit_reached\", \"detail\": \"no more lookups this turn - " +
                            "answer the player with what you already know\"}",
                            $"[NPC] {NpcName}: {call.name} called {internalToolCalls}x in one turn — " +
                            "refused (loop guard).");
                        return;
                    }
                    ConsoleMessage.Info($"[Tool] {NpcName}: {call.name} → {result}");
                    var wi = Window;
                    if (wi == null) return;
                    PrepareForNextReply(wi);          // settles any half-revealed line FIRST…
                    ShowToolCalledLine(wi, call.name);// …so this line is never popped off again
                    dialogueCoroutine = StartCoroutine(Talk(result, asToolResult: true));
                    return;
                }
                RefuseToolCall(call.name,
                    $"{{\"error\": \"unknown_tool\", \"name\": \"{JsonEscape(call.name)}\", \"detail\": " +
                    "\"not one of your tools - answer the player without it\"}",
                    $"[NPC] {NpcName}: unknown tool '{call.name}'. Add an INPCToolProvider that handles it, " +
                    "or drop it from the prompt.");
                return;
            }

            // The tool is not available at all — a retry cannot succeed, so send it to words.
            if (!enableAskUserQuestion)
            {
                RefuseToolCall(call.name, "{\"error\": \"tool_unavailable\", \"detail\": \"AskUserQuestion is not " +
                               "one of your tools here - answer the player in words instead\"}",
                               $"[NPC] {NpcName}: AskUserQuestion called but it is disabled on this NPC.");
                return;
            }
            // SHAPE errors below are RETRYABLE (user 2026-07-28). They used to say "ask the player in
            // words instead", which tells the model to give up on a call it was right to want — the
            // player saw the offer vanish into prose. Tell it what was wrong and to call again instead.
            // Bounded by the same `toolRefusalSent` ping-pong guard that already existed: exactly ONE
            // retry per exchange, then the turn ends. So "call it again" cannot become a loop.
            if (string.IsNullOrWhiteSpace(call.question))
            {
                RefuseToolCall(call.name, "{\"error\": \"malformed_call\", \"detail\": \"AskUserQuestion was " +
                               "called with no question - call it again with a question and 2-4 options\"}",
                               $"[NPC] {NpcName}: AskUserQuestion called with no question: {toolJson.Trim()}");
                return;
            }
            var opts = call.options;
            if (opts.Count < 2)
            {
                string detail = $"AskUserQuestion does not work with {opts.Count} " +
                                (opts.Count == 1 ? "option" : "options") +
                                " - the player needs something to choose between. " +
                                "Call it again with 2-4 options.";
                RefuseToolCall(call.name,
                    "{\"error\": \"malformed_call\", \"detail\": \"" + JsonEscape(detail) + "\"}",
                    $"[NPC] {NpcName}: AskUserQuestion needs 2-4 options, got {opts.Count} — asking it to retry.");
                return;
            }
            if (!(Window is INPCToolQuestionWindow tw))
            {
                ConsoleMessage.Warning($"[NPC] {NpcName}: AskUserQuestion fired but this chat window cannot show " +
                                       "the choice popup — call dropped. Derive the window from NPCDialogueWindow " +
                                       "(it implements INPCToolQuestionWindow for every environment).");
                return;
            }
            StopThinkingDots();                        // the "Tool calling" pulse has served its purpose
            ShowToolCalledLine(Window, call.name);     // and this stays under the spoken line for good
            StartCoroutine(AskUserQuestionRoutine(tw, call.question.Trim(), opts, epoch));
        }

        /// <summary>DEBUG: fire a synthetic AskUserQuestion as if the model had emitted it, so the
        /// popup + pick + &lt;tool_response&gt; resume can be verified independently of whether the
        /// (un-finetuned) model actually calls the tool. Right-click the component in play mode,
        /// with the dialogue open.</summary>
        [ContextMenu("Debug/Fire a test AskUserQuestion")]
        public void DebugFireTestToolCall()
        {
            if (!Application.isPlaying || state != NPCState.WaitingInInteraction || dialogueCoroutine != null)
            {
                ConsoleMessage.Warning($"[NPC] {NpcName}: test tool call needs an OPEN dialogue, idle between replies " +
                                       "(enter play mode, talk to the NPC, then fire it).");
                return;
            }
            TryDispatchToolCall("{\"name\": \"AskUserQuestion\", \"arguments\": {\"question\": " +
                                "\"Will you walk beneath the golden mist, or turn back while you still may?\", " +
                                "\"options\": [\"I will walk through\", \"I turn back\"]}}", dialogueEpoch);
        }

        IEnumerator AskUserQuestionRoutine(INPCToolQuestionWindow tw, string question, List<string> options, int epoch)
        {
            var w = Window;
            toolQuestionOpen = true;
            if (w.SendButton != null) w.SendButton.interactable = false;
            string picked = null;

            // The question is NOT spoken (user spec 2026-07-25). That is WHY a voiced NPC should word it
            // impersonally — "Take <name>'s offer?", not "Do you want to take mine?" — so it reads as the
            // game asking the player rather than as unspoken NPC dialogue. Nothing here enforces that: it
            // is asked for in the NPC's own descriptionAndRules (Velmire does) and learned from the
            // finetune, because a text-only 2D demo has no such split and does not need the rule (user
            // 2026-07-26). The NPC's own line, the one before the call, was already spoken in full:
            // DispatchAfterSpeaking waited for it before this panel opened.
            // The reveal job is stood down defensively: a leftover clause callback would otherwise drip
            // text into the last bubble while the panel is up.
            StopRevealJob();
            revealActive = false;

            tw.ShowToolQuestion(NpcName, question, options, opt => picked = opt);
            while (picked == null && epoch == dialogueEpoch && state == NPCState.WaitingInInteraction)
                yield return null;
            toolQuestionOpen = false;
            if (picked == null || epoch != dialogueEpoch || state != NPCState.WaitingInInteraction
                || dialogueCoroutine != null || compactRoutine != null)
            {
                // dialogue closed / a new session took over under the popup — tear down, no result
                tw.HideToolQuestion();
                if (epoch == dialogueEpoch && w.SendButton != null) w.SendButton.interactable = true;
                yield break;
            }
            // GATED ACTION: engine logic reacts to the pick FIRST (gear changes hands, payment is
            // taken), so the world is already updated when the model's reaction streams — and so a
            // value transfer is never something the model can do on its own, only something the
            // player's own choice triggers.
            try { ToolQuestionAnswered?.Invoke(question, picked); }
            catch (System.Exception e) { ConsoleMessage.Warning($"[NPC] {NpcName}: tool-pick handler threw: {e.Message}"); }

            // the pick goes back as the tool result; the model reads it and reacts in a fresh turn
            PrepareForNextReply(w);
            if (w.SendButton != null) w.SendButton.interactable = true;
            dialogueCoroutine = StartCoroutine(Talk(JsonUtility.ToJson(new ToolPickResult { selected = picked }),
                                                    asToolResult: true));
        }

        // ---- GiveTool: the same shape as AskUserQuestion, one step shorter ----------------------
        // The model names an item (and may name a price and a quantity), the window shows it with
        // Accept and Decline, and the player's button press IS the result. There is no wording to
        // interpret and nothing to guess: the old gear beat had to read the NPC's own option text to
        // tell "Take them" from "Keep your steel", and that guesswork is exactly what this tool
        // removes from the engine.

        /// <summary>Validate a GiveTool call and open the offer panel. Same three failure modes as the
        /// choice path: the tool is not on this NPC's belt (fatal — send it to words), the call names no
        /// item (retryable — the schema requires one), or the window cannot show the panel (dropped with
        /// a warning).</summary>
        void DispatchGiveTool(ParsedToolCall call, string toolJson, int epoch)
        {
            if (!enableGiveTool)
            {
                RefuseToolCall(call.name, "{\"error\": \"tool_unavailable\", \"detail\": \"GiveTool is not one " +
                               "of your tools here - answer the player in words instead\"}",
                               $"[NPC] {NpcName}: GiveTool called but it is disabled on this NPC.");
                return;
            }
            if (string.IsNullOrWhiteSpace(call.item))
            {
                // RETRYABLE, like AskUserQuestion's shape errors: it wanted to hand something over and
                // was right to, so tell it what was missing instead of sending the offer to prose.
                RefuseToolCall(call.name, "{\"error\": \"malformed_call\", \"detail\": \"GiveTool was called " +
                               "with no item - call it again and name the item you are handing over\"}",
                               $"[NPC] {NpcName}: GiveTool called with no item: {toolJson.Trim()}");
                return;
            }
            if (!(Window is INPCToolGiveWindow gw))
            {
                ConsoleMessage.Warning($"[NPC] {NpcName}: GiveTool fired but this chat window cannot show the " +
                                       "offer panel — call dropped. Derive the window from NPCDialogueWindow " +
                                       "(it implements INPCToolGiveWindow for every environment).");
                return;
            }
            ToolGiveOffer offer = OfferOf(call);
            StopThinkingDots();                        // the "Tool calling" pulse has served its purpose
            ShowToolCalledLine(Window, call.name);     // and this stays under the spoken line for good
            StartCoroutine(GiveToolRoutine(gw, offer, epoch));
        }

        IEnumerator GiveToolRoutine(INPCToolGiveWindow gw, ToolGiveOffer offer, int epoch)
        {
            var w = Window;
            // the SAME flag the choice panel raises: it means "a panel owns the turn", so a typed send
            // cannot slip in under this one either and orphan the pending call
            toolQuestionOpen = true;
            if (w.SendButton != null) w.SendButton.interactable = false;
            bool? accepted = null;

            // stand the reveal down for the same reason the choice panel does — a leftover clause
            // callback would drip text into the last bubble while the panel is up
            StopRevealJob();
            revealActive = false;

            bool canAccept = true;
            if (ToolGiveAcceptGate != null)
            {
                // A throwing gate is a bug in the host game, not a reason to refuse the player: it is
                // reported and read as "yes", because the transaction itself lives in the accepted
                // handler and can still decline there. See ToolGiveAcceptGate.
                try { canAccept = ToolGiveAcceptGate(offer); }
                catch (System.Exception e)
                {
                    ConsoleMessage.Warning($"[NPC] {NpcName}: GiveTool accept-gate threw: {e.Message} — " +
                                           "Accept left enabled.");
                    canAccept = true;
                }
            }
            ConsoleMessage.Info($"[Tool] {NpcName}: GiveTool → {offer.item}"
                              + (offer.quantity.HasValue ? $" x{offer.quantity.Value}" : "")
                              + (offer.price.HasValue ? $" @ {offer.price.Value}" : " (no price)")
                              + (canAccept ? "" : " — Accept gated off"));

            gw.ShowToolGive(NpcName, offer, canAccept, ok => accepted = ok);
            while (accepted == null && epoch == dialogueEpoch && state == NPCState.WaitingInInteraction)
                yield return null;
            toolQuestionOpen = false;
            if (accepted == null || epoch != dialogueEpoch || state != NPCState.WaitingInInteraction
                || dialogueCoroutine != null || compactRoutine != null)
            {
                // dialogue closed / a new session took over under the panel — tear down, no result
                gw.HideToolGive();
                if (epoch == dialogueEpoch && w.SendButton != null) w.SendButton.interactable = true;
                yield break;
            }
            // GATED ACTION, exactly as above: the world is updated BEFORE the model's reaction streams,
            // and only ever by the player's own click. Declining does nothing at all — the model just
            // reads the "no" and answers in character.
            if (accepted == true)
            {
                try { ToolGiveAccepted?.Invoke(offer); }
                catch (System.Exception e)
                {
                    ConsoleMessage.Warning($"[NPC] {NpcName}: GiveTool accepted handler threw: {e.Message}");
                }
            }

            PrepareForNextReply(w);
            if (w.SendButton != null) w.SendButton.interactable = true;
            dialogueCoroutine = StartCoroutine(Talk(ToolGiveResult(accepted == true), asToolResult: true));
        }

        /// <summary>DEBUG: fire a synthetic GiveTool as if the model had emitted it — the XML wire shape
        /// its own chat template declares — so the offer panel, the accept-gate, the hand-over and the
        /// &lt;tool_response&gt; resume can be verified without waiting for the model to call it.
        /// Right-click the component in play mode, with the dialogue open.</summary>
        [ContextMenu("Debug/Fire a test GiveTool")]
        public void DebugFireTestGiveTool()
        {
            if (!Application.isPlaying || state != NPCState.WaitingInInteraction || dialogueCoroutine != null)
            {
                ConsoleMessage.Warning($"[NPC] {NpcName}: test tool call needs an OPEN dialogue, idle between replies " +
                                       "(enter play mode, talk to the NPC, then fire it).");
                return;
            }
            TryDispatchToolCall("<function=GiveTool>\n<parameter=item>\nlongsword\n</parameter>\n" +
                                "<parameter=price>\n80\n</parameter>\n</function>", dialogueEpoch);
        }

        // Audio-synced reveal: each clause event carries its spoken DURATION, and a single
        // pacing coroutine drips the clause into the bubble across ~that window (char-by-char
        // or whole words — syncedTextReveal), finishing slightly early — the text "types
        // itself" in step with the voice instead of whole sentences popping in.
        readonly Queue<(string clause, float dur)> revealQueue = new Queue<(string, float)>();
        Coroutine revealJob;
        // Verbatim accumulation of everything fed to the voices this reply — the ONLY place
        // the whitespace BETWEEN clauses still exists (the clause cutter trims it off every
        // chunk), and therefore the only honest source for the separator the reveal joins
        // clauses with. See the alignment note in RevealWordsJob.
        readonly System.Text.StringBuilder revealSource = new System.Text.StringBuilder();
        int revealSrcPos;   // chars of revealSource already consumed by revealed clauses

        void OnClauseSpokenHandler(string clause, float duration)
        {
            if (!revealActive || Window == null || state == NPCState.Idle) return;
            StopThinkingDots();   // first spoken clause takes the bubble over from the dots
            revealQueue.Enqueue((clause, duration));
            if (revealJob == null) revealJob = StartCoroutine(RevealWordsJob());
        }

        IEnumerator RevealWordsJob()
        {
            while (revealQueue.Count > 0)
            {
                (string clause, float dur) = revealQueue.Dequeue();
                var w = Window;
                if (!revealActive || w == null || state == NPCState.Idle) break;
                // Reveal a growing PREFIX OF THE REAL STREAMED TEXT, never a re-joined copy. Two
                // regressions taught this shape (both looked the same: whitespace missing until
                // FinishSyncedReveal settled the bubble on the real text and it re-flowed "urat
                // fix cand abia se termina"):
                //  - 2026-07-26: re-joining split WORDS with single spaces flattened newlines
                //    INSIDE a clause -> fixed by typing a prefix of the clause string.
                //  - 2026-07-30: the clause CUTTER trims the ender run off every chunk
                //    (TtsClauseCut cuts at the end of "...\n\n", then Trim() eats it), so a
                //    paragraph break BETWEEN clauses survived in no clause at all and the
                //    single-space join drew one running paragraph again.
                // So each clause is ALIGNED against revealSource — the verbatim accumulation of
                // what was fed to the voices — and the separator between clauses is copied from
                // there. The alignment is exact (a chunk is a Trim of a substring of the feed);
                // if it ever fails (foreign clause), the old one-space join is the fallback.
                string basis = spokenShown ?? "";
                string src = revealSource.ToString();
                int start = revealSrcPos;
                while (start < src.Length && char.IsWhiteSpace(src[start])) start++;
                if (start + clause.Length <= src.Length &&
                    string.CompareOrdinal(src, start, clause, 0, clause.Length) == 0)
                {
                    // the real separator — where any "\n\n" lives. Dropped before the reply's
                    // FIRST clause, where it would render as a leading blank line.
                    if (basis.Length > 0) basis += src.Substring(revealSrcPos, start - revealSrcPos);
                    revealSrcPos = start + clause.Length;
                }
                else if (basis.Length > 0 && !char.IsWhiteSpace(basis[basis.Length - 1])
                         && clause.Length > 0 && !char.IsWhiteSpace(clause[0]))
                    basis += " ";                     // clause boundary the model did not punctuate
                int chars = Mathf.Max(1, clause.Length);
                float window = Mathf.Max(0.05f, dur * 0.98f);   // finish slightly early, as before
                float t0 = Time.realtimeSinceStartup;
                int shown = 0;
                while (shown < clause.Length)
                {
                    if (!revealActive || state == NPCState.Idle) break;
                    // Elapsed-based pacing (2026-07-30): the target index tracks the wall clock, so
                    // a slow frame CATCHES UP instead of drifting behind the voice — the old
                    // fixed-wait-per-word form accumulated its 0.02 s floors and its dropped
                    // frames, and every clause ended a little later than its audio.
                    float frac = Mathf.Clamp01((Time.realtimeSinceStartup - t0) / window);
                    int target = Mathf.Clamp(Mathf.CeilToInt(frac * chars), shown, clause.Length);
                    if (syncedTextReveal == RevealGranularity.WordByWord && target > shown)
                    {
                        // words land whole: extend to the end of every word the clock has entered
                        int e = shown;
                        while (e < target)
                        { int sp = clause.IndexOf(' ', e); e = sp < 0 ? clause.Length : sp + 1; }
                        target = e;
                    }
                    if (target > shown)
                    {
                        shown = target;
                        spokenShown = bubbleLive = basis + clause.Substring(0, shown);
                        w.PopLastMessage();
                        w.AddMessage(NpcName, Bubble(spokenShown));
                    }
                    if (shown < clause.Length) yield return null;
                }
            }
            revealJob = null;
        }

        void StopRevealJob()
        {
            if (revealJob != null) { StopCoroutine(revealJob); revealJob = null; }
            revealQueue.Clear();
        }

        // asking again while the previous reply is still being voiced = talking over him:
        // cut the audio and settle the window on the previous FULL text
        void PrepareForNextReply(INPCChatWindow w)
        {
            StopThinkingDots();
            if (!revealActive) return;
            StopRevealJob();
            if (kkVoice != null && (kkVoice.IsSpeaking || kkVoice.IsAudioPlaying)) kkVoice.StopSpeaking();
            if (pkVoice != null && (pkVoice.IsSpeaking || pkVoice.IsAudioPlaying)) pkVoice.StopSpeaking();
            if (pendingFullReply != null && spokenShown != pendingFullReply)
            {
                w.PopLastMessage();
                w.AddMessage(NpcName, Bubble(pendingFullReply));
                bubbleLive = pendingFullReply;   // settled to the full text: no longer a cut reply
            }
            revealActive = false;
        }

        // A starved voice (a GPU too slow to synthesize pocket-tts in real time) can leave IsSpeaking
        // stuck / never cleanly drain. The synced reveal must NOT wait on it forever, or the bubble
        // stays blank AND everything gated on the reveal (the dialogue settling, the pause menu)
        // wedges. If nothing at all advances for this long, we stop waiting and force-render the
        // full reply. IMPORTANT (bug fix, user 2026-07-22): "progress" MUST include the voice's
        // synthesis counter (SamplesPushed), not just the visible text — on a slow GPU a long
        // reply pauses for several seconds BETWEEN clauses while the next chunk's prebuffer
        // synthesizes (nothing audible, no words dripping), and the old text-only watchdog
        // executed a perfectly healthy voice mid-reply: audio died, full text dumped at once.
        const float RevealStallTimeout = 6f;

        // total samples the synced-reveal voices have synthesized — advancing = the voice is alive
        long SynthProgress() => (pkVoice != null ? pkVoice.SamplesPushed : 0)
                              + (kkVoice != null ? kkVoice.SamplesPushed : 0);

        IEnumerator FinishSyncedReveal(string full)
        {
            // wait for the voice AND for the word-pacing to drain its queued clauses — but bail on a
            // genuine stall (no progress for RevealStallTimeout), never hang the game on a slow GPU
            float lastProgress = Time.unscaledTime;
            string lastShown = spokenShown;
            int lastQueued = revealQueue.Count;
            long lastSynth = SynthProgress();
            // HasPendingSpeech, NOT IsSpeaking: this coroutine starts the same frame the reply's
            // tail was flushed into the voice, and IsSpeaking only latches on the voice's NEXT
            // Update. With clausesPerChunk>1 a short reply queues ALL its speech at flush time —
            // gating on IsSpeaking judged the voice "done" one frame before it ever spoke and
            // dumped the full text instantly (user bug, 2026-07-22).
            while (revealActive && ((kkVoice != null && kkVoice.HasPendingSpeech)
                                    || (pkVoice != null && pkVoice.HasPendingSpeech)
                                    || revealJob != null || revealQueue.Count > 0))
            {
                long synth = SynthProgress();
                if (spokenShown != lastShown || revealQueue.Count != lastQueued || synth != lastSynth)
                {
                    lastShown = spokenShown; lastQueued = revealQueue.Count; lastSynth = synth;
                    lastProgress = Time.unscaledTime;   // real progress — keep waiting
                }
                else if (Time.unscaledTime - lastProgress > RevealStallTimeout)
                {
                    // voice/reveal stalled: stop the drip job + hush the stuck voice so it can't keep
                    // VoicesAudible() true (which would block the next ask / the clean close), then
                    // settle the full text below.
                    StopRevealJob();
                    StopVoices();
                    break;
                }
                yield return null;
            }
            var w = Window;
            if (!revealActive || w == null || state == NPCState.Idle) yield break;
            if (spokenShown != full)   // voice done (or stalled) but tail text never got audio — settle it
            {
                w.PopLastMessage();
                w.AddMessage(NpcName, Bubble(full));
                spokenShown = bubbleLive = full;   // bubbleLive too: it is what MarkReplyCutShort compares
            }
            revealActive = false;
        }

        private IEnumerator Talk(string question, bool asToolResult = false)
        {
            state = NPCState.TalkingInInteraction;
            int epoch = dialogueEpoch;
            replyCanceled = false;
            // a real player line opens a fresh budget of internal reads; a tool-result turn is a
            // continuation of the same one (that is exactly what the loop guard must count)
            if (!asToolResult) { internalToolCalls = 0; toolRefusalSent = false; exchangePrefix = ""; }
            var w = Window;
            // Send stays interactable: sending mid-reply cancels this reply at a token boundary
            // and asks anew (InterruptThenAsk) — the state machine gates it, not the button.

            Turn turn = null;
            if (historyMode != HistoryMode.ResetEveryTime)
            {
                // recorded up-front so an Escape mid-reply still keeps the (partial) exchange
                turn = new Turn { user = question, npc = "", tool = asToolResult };
                transcript.Add(turn);
                activeTurn = turn;
            }

            StringBuilder response = new StringBuilder();
            activeResponse = response;
            bool synced = SyncedReveal;
            if (synced)
            {
                spokenShown = null; pendingFullReply = null; revealActive = true; StopRevealJob();
                revealSource.Clear(); revealSrcPos = 0;   // fresh reply, fresh alignment source
            }
            // unconditional (both paths draw into the bubble): a stale value here would mark the NEXT
            // reply's opening bubble with the previous reply's interruption. drainTurn goes with it —
            // once a new reply starts, the previous one is no longer the thing being cut short.
            bubbleLive = null;
            if (!asToolResult) drainTurn = null;
            bool showThink = w.ShowThinkingTokens;
            string visibleFull = "", thinkFull = "", toolCallFull = "";
            int voicedLen = 0;          // visible chars already handed to the voice
            bool contentShown = false;  // the animated dots own the bubble until real content
            bool toolPulseShown = false;// the dots were relabelled "Tool calling" for this reply
            bool toolCallClosed = false;// </tool_call> seen — decoding was stopped there
            bool toolSendLocked = false;// Speak was taken away because a call is mid-flight

            // thinking placeholder: ". / .. / ..." pulses until the first real content lands
            if (asToolResult) w.PopLastMessage();   // keep writing in THIS exchange's bubble
            w.AddMessage(NpcName, Bubble(StatusStyled(".")));
            StartThinkingDots(w);

            // A background conversation-KV save still reading this model's GPU state holds the Busy
            // guard (SaveConversationKV is Guarded) — most often the snapshot a just-finished
            // compaction kicked. Driving Chat now would be REFUSED by the guard and the turn lost
            // (audit #13). The gate always clears (its coroutine's finally drops it), so wait it out
            // behind the "Thinking…" dots; a close mid-wait bails via the epoch. This defers the
            // send instead of dropping it.
            while (KvSaveInFlightFor(llm) && epoch == dialogueEpoch) yield return null;
            if (epoch != dialogueEpoch) { StopThinkingDots(); yield break; }

            System.Action<string> onTok = (token) =>
                {
                    // emoji/symbols the UI font can't render (squares) also drive the TTS into
                    // garbage sounds — strip them HERE, before anything consumes the token
                    // (window, voices, transcript all flow from `response`)
                    token = StripUnrenderable(token);
                    if (token.Length == 0) return;
                    response.Append(token);
                    string raw = response.ToString();
                    SplitThink(raw, out visibleFull, out thinkFull);
                    // the <tool_call> body is a machine channel: split it out of the visible text
                    // (never rendered, never voiced — same treatment as <think>); a completed call
                    // dispatches AFTER the turn ends, from Talk's tail
                    if (ToolsEnabled)
                    {
                        SplitToolCall(visibleFull, out visibleFull, out toolCallFull);
                        // the call body is invisible by design, so without this the window would sit
                        // on anonymous dots while the model works a tool — name it, the same way
                        // "Thinking…" and "Compacting…" name their phases
                        if (!toolPulseShown && !contentShown && toolCallFull.Length > 0 && visibleFull.Length == 0)
                        {
                            toolPulseShown = true;
                            StartThinkingDots(w, "Tool calling");
                        }
                        // Once a call has STARTED streaming, take Speak away until the turn resolves
                        // (user 2026-07-26). A send landing mid-call cancels the decode at the next
                        // token boundary, so </tool_call> never arrives, the tool never runs, and the
                        // player's own message is what killed it — with nothing on screen saying so.
                        // The normal reply tail (state → WaitingInInteraction) re-enables it, as it
                        // does after any turn; this only closes the window where the call is in flight.
                        if (!toolSendLocked && toolCallFull.Length > 0)
                        {
                            toolSendLocked = true;
                            if (w.SendButton != null) w.SendButton.interactable = false;
                        }
                        // THE TURN ENDS AT THE CALL. Stop decoding the moment </tool_call> lands: the
                        // NPC must not run on past it (user spec 2026-07-25 — "the model has to wait for
                        // the answer, it cannot continue"), and anything it wrote after the call used to
                        // be rendered AND voiced. Cancelling here leaves the KV holding the turn exactly
                        // as far as the call, which is what a trained model would have emitted anyway,
                        // and the next turn closes it like any truncated turn. replyCanceled stays
                        // FALSE — this reply finished on purpose, it was not interrupted.
                        if (!toolCallClosed && raw.IndexOf(Qwen3_5Modeling.Qwen3_5ChatTemplate.ToolCallEndTag,
                                                           System.StringComparison.Ordinal) >= 0)
                        {
                            toolCallClosed = true;
                            llm.CancelChat();
                        }
                    }
                    // reasoning NEVER reaches the TTS — only newly-VISIBLE text is fed
                    if (speakReplies && !replyCanceled && visibleFull.Length > voicedLen)
                    {
                        FeedVoiceText(visibleFull.Substring(voicedLen));
                        voicedLen = visibleFull.Length;
                    }
                    if (synced)
                    {
                        // the voice-paced reveal owns the bubble; while he is still thinking
                        // (and the window allows it) stream the reasoning into it, dimmed
                        if (showThink && thinkFull.Length > 0 && spokenShown == null)
                        {
                            StopThinkingDots();
                            w.PopLastMessage();
                            w.AddMessage(NpcName, Bubble(ThinkStyled(thinkFull)));
                        }
                        return;
                    }
                    string display = showThink && thinkFull.Length > 0
                        ? ThinkStyled(thinkFull) + visibleFull
                        : visibleFull;
                    if (display.Length == 0) return;   // still inside <think> — dots keep pulsing
                    StopThinkingDots();
                    w.PopLastMessage();
                    w.AddMessage(NpcName, Bubble(display));
                    bubbleLive = display;
                    contentShown = true;
                };
            // -1 inspector values fall back to the selected model's recommended Config preset.
            // A tool-result turn renders as <tool_response> (the model's tool template) instead
            // of a plain user turn — same streaming contract either way.
            yield return asToolResult
                ? llm.ChatToolResult(question, max_new_tokens: maxNewTokens, temperature: temperature,
                    top_k: topK >= 0 ? topK : llm.Config.DefaultTopK,
                    top_p: topP >= 0f ? topP : llm.Config.DefaultTopP,
                    min_p: minP >= 0f ? minP : llm.Config.DefaultMinP,
                    presence_penalty: presencePenalty >= 0f ? presencePenalty : llm.Config.DefaultPresencePenalty,
                    repetition_penalty: repetitionPenalty >= 0f ? repetitionPenalty : llm.Config.DefaultRepetitionPenalty,
                    enable_thinking: allowThinking,
                    onTokenGenerated: onTok)
                : llm.Chat(question, max_new_tokens: maxNewTokens, temperature: temperature,
                    top_k: topK >= 0 ? topK : llm.Config.DefaultTopK,
                    top_p: topP >= 0f ? topP : llm.Config.DefaultTopP,
                    min_p: minP >= 0f ? minP : llm.Config.DefaultMinP,
                    presence_penalty: presencePenalty >= 0f ? presencePenalty : llm.Config.DefaultPresencePenalty,
                    repetition_penalty: repetitionPenalty >= 0f ? repetitionPenalty : llm.Config.DefaultRepetitionPenalty,
                    enable_thinking: allowThinking,
                    onTokenGenerated: onTok);
            if (speakReplies && !replyCanceled)
            {
                if (visibleFull.Length > voicedLen) FeedVoiceText(visibleFull.Substring(voicedLen));
                FlushVoiceText();   // speak the trailing clause
            }
            StopThinkingDots();
            // surface the model's hidden reasoning in the console so its behavior can be verified
            // (it is never voiced and only rendered in-window behind the ShowThinkingTokens toggle)
            if (thinkFull.Length > 0)
                ConsoleMessage.Info($"[Think] {NpcName}: <i>{thinkFull.Trim()}</i>");
            bool stillOpen = state == NPCState.TalkingInInteraction && epoch == dialogueEpoch;   // close mid-reply drops state/epoch
            // transcripts/window always carry the VISIBLE text (raw kept only if nothing parsed)
            string finalVisible = visibleFull.Length > 0 ? visibleFull
                                : thinkFull.Length > 0 ? "" : response.ToString();
            if (synced && stillOpen)   // after a close this state is junk — never carry it over
            {
                pendingFullReply = finalVisible;
                StartCoroutine(FinishSyncedReveal(pendingFullReply));
            }
            else if (!contentShown && stillOpen)   // reply was pure <think> with display off — settle the bubble
            {
                w.PopLastMessage();
                if (finalVisible.Length > 0) w.AddMessage(NpcName, Bubble(finalVisible));
                // a reply that is ONLY a tool call has no dialogue to settle: the permanent
                // "Tool Called (…)" line the dispatch leaves behind stands in for it, so don't
                // park an empty "..." bubble above it
                else if (string.IsNullOrWhiteSpace(toolCallFull)) w.AddMessage(NpcName, Bubble("..."));
            }

            if (turn != null) turn.npc = finalVisible;
            drainTurn = turn;      // still being spoken — keep it markable, see the field's note
            activeTurn = null;
            activeResponse = null;
            if (stillOpen) state = NPCState.WaitingInInteraction;
            // close+reopen (epoch +2) means the handle already belongs to a NEWER session —
            // never clobber it; a plain close (+1) still expects us to release it cleanly
            if (dialogueEpoch <= epoch + 1) dialogueCoroutine = null;

            // Context-window handling now that this reply is on the KV (ResumeFromCompact only):
            // at the limit, compact NOW (auto) behind the "Compacting…" pulse, then talking resumes
            // on the short compacted prefix. The KV headroom above the threshold gives the compact
            // pass room to run.
            if (stillOpen && historyMode == HistoryMode.ResumeFromCompact && compactRoutine == null
                && ContextFull() && HasCompactableHistory())
            {
                // STANDARD (user spec): the reply that hit the limit is delivered IN FULL —
                // decoded, typed AND spoken to the end (the KV headroom above the limit
                // absorbs the overshoot). "Compacting…" may only appear once the NPC
                // finished talking.
                while (state == NPCState.WaitingInInteraction && epoch == dialogueEpoch
                       && dialogueCoroutine == null && !interruptPending
                       && (VoicesAudible() || revealJob != null || revealQueue.Count > 0))
                    yield return null;
                // stand down if a new/queued ask took over meanwhile (its own tail
                // re-triggers compaction after THAT reply — audit #2) or the dialogue
                // closed (the next open's crash-recovery compacts behind the same pulse)
                if (state == NPCState.WaitingInInteraction && epoch == dialogueEpoch
                    && dialogueCoroutine == null && !interruptPending
                    && compactRoutine == null)
                {
                    w.SendButton.interactable = false;
                    compactRoutine = StartCoroutine(CompactConversationRoutine());
                    yield return ShowCompactingUntilDone(w);
                    // the pulse pops itself; the window intentionally KEEPS the whole
                    // conversation visible — only the next open collapses it to the compact
                }
            }

            if (state != NPCState.WaitingInInteraction || dialogueCoroutine != null || epoch != dialogueEpoch)
                yield break;   // dialogue closed or a new ask took over during the waits above
            w.SendButton.interactable = true;
            w.InputField.ActivateInputField();
            if (!replyCanceled) OnReplyFinished();

            // A completed <tool_call> is dispatched once the NPC has finished SAYING the line that came
            // before it (user spec: text, spoken to the end, then the question) — DispatchAfterSpeaking
            // owns that wait. An internal provider read then answers itself and re-enters Talk with the
            // result; an INTERACTIVE tool opens its panel — the choice popup (AskUserQuestion) or the
            // offer popup (GiveTool) — whose click comes back as the <tool_response>. All of them
            // continue THIS exchange: dialogueCoroutine was cleared just above, so the new turn owns
            // the handle.
            if (!replyCanceled && ToolsEnabled && !string.IsNullOrWhiteSpace(toolCallFull))
                StartCoroutine(DispatchAfterSpeaking(toolCallFull, epoch));
            // No call: the reply stands as spoken. There is deliberately NO engine-side rescue here —
            // a hidden second pass that asked the model to convert its own prose question into a call was
            // tried and REMOVED (user 2026-07-25): it injected an instruction turn the author could not
            // see into the NPC's own context. Everything that pushes the model toward calling now lives
            // in the system prompt, in plain sight, and the v1.4 finetune is what makes it reliable.
        }

        /// <summary>Closes the dialogue from any state — Escape, the Leave button, or scripted.</summary>
        public void CloseInteraction()
        {
            // shared-window UI: the Leave button fires on EVERY interactor — an idle sibling
            // must not run a close (it would fade its own voice and, worse, snapshot the ACTIVE
            // NPC's KV under its own key on a shared instance). Audit #1.
            if (state == NPCState.Idle) return;
            dialogueEpoch++;
            bool interrupted = dialogueCoroutine != null;
            if (interrupted)
            {
                // cooperative cancel: the reply unwinds at its next token boundary with the KV
                // holding the truncated turn as if the model had stopped there — never
                // StopCoroutine (that can land mid-forward and half-write the KV). A running
                // compaction is NEVER canceled — neither OURS nor a SIBLING's on the same pooled
                // instance (CancelChat would land on ITS in-flight Chat). Audit #4.
                replyCanceled = true;
                if (compactingNpc == null || compactingNpc.llm != llm) llm?.CancelChat();
                StartCoroutine(CloseConversationWhenReplyUnwinds());
            }

            // OUTSIDE the `interrupted` branch on purpose. Leaving mid-reply IS an interrupt (user
            // 2026-07-28), but `interrupted` only means "still GENERATING" — leave while he is merely
            // still TALKING and it is false, which is the common case and was the whole bug: the turn
            // kept only its full decoded text and a reopen replayed paragraphs he never said out loud
            // ("i see the entire text in the conversation that was unspoken part"). Safe here because
            // MarkReplyCutShort no-ops when everything generated was also revealed, so a clean close
            // marks nothing. Must run BEFORE bubbleLive is wiped a few lines down.
            MarkReplyCutShort();

            state = NPCState.Idle;
            conversing.Remove(this);   // world audio comes back up from here (ConversationAudioDucker)
            // Prefetch policy (user 2026-07-30): the conversation-open BOOST dies with the
            // conversation. Still inside the zone with weights left to stream -> back to the
            // zone's SLOW rate (the governor exited the moment the dialogue opened and left the
            // budget at full speed; nothing else would ever lower it again while the player
            // idles here). Zone exit keeps its own full-defetch edge in Update.
            if (usePrefetchZone && inPrefetchZone && llm != null && !llm.IsReady)
                BeginLlmSlowPrefetch();
            // reveal machinery dies WITH the dialogue — a leaked revealActive/pendingFullReply
            // otherwise resurrects the previous reply's text at the NEXT dialogue's first send
            // (PrepareForNextReply settles stale state), even on ResetEveryTime
            StopThinkingDots();
            StopRevealJob();
            revealActive = false;
            pendingFullReply = null;
            spokenShown = null;
            bubbleLive = null;
            // Reset the exchange prefix too (fix 2026-07-28). It was written only by ShowToolCalledLine
            // and cleared only by the next PLAYER turn in Talk, so it outlived the dialogue: reopening
            // after a tool exchange made RepaintTranscript prepend the old spoken line + "Tool Called
            // (…)" to EVERY repainted bubble, including the "(earlier conversation summarized)" banner.
            exchangePrefix = "";
            FadeOutVoices();   // Leave: speech fades to silence (~1 s) instead of cutting or
                               // talking on; the NPC settles to idle as IsAudioPlaying drops.
            if (!interrupted) CloseConversation(false);

            var w = Window;
            if (w != null)
            {
                w.SetSendLoading(false);   // Escape mid-load: stop the pulse, restore the label
                w.Clear();
                w.Close();
                w.SendButton.interactable = true;
            }

            OnInteractionClosed(interrupted);
        }

        // Leave mid-reply: wait for the canceled reply to unwind (token boundary; a post-reply
        // compaction extends the wait), then run the normal CLEAN close — the KV is valid, so
        // the conversation save/resume stay eligible. The hard StopCoroutine survives only as a
        // dead-model fallback after the deadline.
        IEnumerator CloseConversationWhenReplyUnwinds()
        {
            int epoch = dialogueEpoch;
            float deadline = Time.unscaledTime + 10f;
            while (dialogueCoroutine != null && epoch == dialogueEpoch
                   && (compactRoutine != null || Time.unscaledTime < deadline))
                yield return null;
            if (epoch != dialogueEpoch) yield break;   // a NEW session owns the state now (reopen)
            if (dialogueCoroutine != null)
            {
                StopCoroutine(dialogueCoroutine);
                dialogueCoroutine = null;
                // StopCoroutine skips the Busy-guard finally, so the abandoned Chat would leave the
                // pooled instance latched (every later Chat refuses). Release it explicitly — the KV
                // is marked dead just below, so the next open fully re-initializes anyway.
                llm?.AbandonGuardedOperation();
                CloseConversation(interrupted: true);
                yield break;
            }
            CloseConversation(interrupted: false);
        }

        /// <summary>
        /// Conversation-persistence half of closing, per <see cref="historyMode"/>. GPU residency
        /// is NOT decided here: the prefetch zone (or, without one, the talk trigger) owns it —
        /// closing the chat never releases the model while the player is still inside; walking
        /// out does (Update's zone-exit branch / OnPlayerLeft), except KeepAliveInBackground.
        ///   Interrupt — reached only through the dead-model FALLBACK in
        ///     CloseConversationWhenReplyUnwinds (an externally stopped coroutine can half-write
        ///     the KV) so the KV is marked dead (chatLive=false). The normal leave-mid-reply path
        ///     cancels cooperatively at a token boundary and closes CLEAN (interrupted=false).
        ///   Clean close in the continue modes — with cacheKVCache the WHOLE conversation state
        ///     snapshots to disk in the background (any residency release due later waits for the
        ///     snapshot's GPU readbacks via the per-instance save gate).
        /// </summary>
        protected void CloseConversation(bool interrupted)
        {
            if (interrupted)
            {
                // reply was cut mid-generation — cut the speech too
                StopVoices();
                // keep what the NPC managed to say, so a resumed conversation stays consistent
                if (activeTurn != null && activeResponse != null)
                    activeTurn.npc = activeResponse.ToString();
                chatLive = false;
            }
            activeTurn = null;
            activeResponse = null;
            // else: clean close — CloseInteraction already started the ~1 s voice fade, so the
            // speech ramps to silence instead of hard-cutting mid-word (it never talks on)

            // ResetEveryTime: the conversation ceases to exist the moment the chat CLOSES (same
            // session) — wipe it NOW instead of lazily at the next open, so nothing of it survives
            // in any state the modes below might touch. The continue mode is the fully-persistent
            // one (chat-to-chat and session-to-session).
            if (historyMode == HistoryMode.ResetEveryTime)
            {
                transcript.Clear();
                compactSummary = null;
                chatLive = false;
            }

            // WS-G SnapshotConversation — implemented: on a clean close in the continue modes the
            // whole conversation state (KV/SSM prefix + token-seen penalty counts + open-turn flag
            // + this transcript) is persisted via LLM.SaveConversationKV (Qwen3_5Cache v2 format —
            // FP32/FP16/INT8 KV incl. the INT8 scale/zp planes), and OpenConversation restores it
            // through TryRestoreConversationKV before falling back to the transcript re-prefill.
            // An interrupted KV is never saved (a stopped coroutine can leave it half-written
            // mid-forward — untrustworthy, same reason chatLive drops above).
            // TODO(Gemma3): mirror the v2 persistence in Gemma3Cache/Gemma3ForCausalLM; until
            // then Gemma NPCs no-op the save and always miss the restore (base-class defaults).
            bool saveConversation = cacheKVCache && !interrupted && chatLive && !KvSaveInFlightFor(llm)
                && compactRoutine == null   // a running compaction is FORWARDING the model — it re-saves itself when it lands
                && llm != null && LLMPool.OwnsConversation(llm, this)   // the GPU state must be OURS, not a sibling's (audit #1)
                && historyMode != HistoryMode.ResetEveryTime
                // a JUST-compacted conversation has an empty transcript but is far from empty — its
                // whole past is the compact, and without this it would never reach disk on a close
                // right after a compaction (so an app restart would lose it)
                && HasCompactableHistory();
            if (saveConversation)
                StartCoroutine(SaveConversationKvRoutine());

            // ResumeFromCompact compaction is NOT triggered on close anymore — it fires the moment
            // the conversation reaches maxContextLength (after the offending reply, or on the next
            // open as crash-recovery). The full-state save above is the pre-compact fallback that
            // on-open recovery re-detects and compacts if a mid-session compact never landed.
        }

        // Background conversation-KV snapshot. Any residency release that becomes due while it
        // runs (player walks out of the zone/trigger) waits on the per-instance save gate — the
        // save reads the model's GPU buffers, so releasing mid-save would tear it.
        private IEnumerator SaveConversationKvRoutine()
        {
            var saving = llm;
            if (saving == null) yield break;
            kvSavesInFlight[saving] = this;
            try
            {
                yield return saving.SaveConversationKV(ConversationKvKey(), SerializeTranscript(), EffectiveSystemPrompt);
            }
            finally
            {
                // an exception must never leave the gate latched (audit #12); drop only OUR entry
                if (kvSavesInFlight.TryGetValue(saving, out var owner) && owner == this)
                    kvSavesInFlight.Remove(saving);
            }
        }

        // Zone-exit release, deferred past any in-flight compaction AND conversation save (user
        // rule: the model NEVER leaves the GPU until the compact + its KV snapshot land — walking
        // out of the zone only delays the unload; releasing mid-save would error the readbacks
        // and lose the snapshot). Skipped if a dialogue started meanwhile, or if the player
        // walked back into the zone during the (possibly long) compaction wait.
        private IEnumerator ReleaseLlmAfterKvSave()
        {
            // also outlast an unwinding canceled reply — releasing mid-forward NREs the Talk
            // coroutine and tears the KV (audit #8) — and a manual reset, which is re-prefilling the
            // system prompt on this very instance
            while (compactRoutine != null || KvSaveInFlightFor(llm) || dialogueCoroutine != null
                   || resetRoutine != null) yield return null;
            if (state == NPCState.Idle && !(usePrefetchZone && inPrefetchZone))
                ReleaseLlm(collectGarbage: true);
        }

        // Coroutines die with the component; never leave the save gate latched for a later
        // re-enable — OpenConversation spins on it. Only OUR gate entries are dropped: a
        // sibling's in-flight save (even on the same shared instance) stays latched.
        protected virtual void OnDisable()
        {
            conversing.Remove(this);   // disabled mid-dialogue: never leave the world ducked forever
            foreach (var m in new List<LLM>(kvSavesInFlight.Keys))
                if (kvSavesInFlight[m] == this) kvSavesInFlight.Remove(m);
            compactRoutine = null;                     // its coroutine died with the component
            resetRoutine = null;                       // ...so did a manual reset; never leave the gate latched
            if (compactingNpc == this) compactingNpc = null;
            if (llm != null) llm.DiskKVCache = cacheKVCache;   // a dying compaction dropped it for its re-seed
            interruptPending = false;                  // interrupt-ask coroutine died with the component
        }

        // Destroy ≠ disable: a destroyed NPC must still drop its pool refcount or the shared
        // model stays pinned on the GPU for the rest of the session (F5). OnDisable (Unity
        // orders it first) already tore down this NPC's own save/compaction gates, and a
        // SIBLING's in-flight save keeps the instance alive through its own pool ref. A
        // synchronous OnDestroy can't wait like ReleaseLlmAfterKvSave — if OUR work somehow
        // still reads the GPU, release anyway: scene teardown tolerates it (that snapshot's
        // readbacks just error out and the old file on disk survives).
        protected virtual void OnDestroy()
        {
            if (llm == null) return;
            if ((kvSavesInFlight.TryGetValue(llm, out var saver) && saver == this) || compactingNpc == this)
                ConsoleMessage.Warning($"[NPC] {NpcName}: destroyed with its conversation-KV save/compaction " +
                                       "still reading the model — releasing the pool ref anyway.");
            ReleaseLlm();
        }

        // ---------------------------------------------------------------- helpers

        /// <summary>Per-token TTS fan-out to whichever voice component this NPC runs.</summary>
        protected void FeedVoiceText(string token)
        {
            // verbatim copy of the feed — the reveal aligns clauses against it (the clause
            // cutter destroys inter-clause whitespace, see RevealWordsJob)
            if (revealActive) revealSource.Append(token);
            voice?.FeedText(token);
            cvVoice?.FeedText(token);
            kkVoice?.FeedText(token);
            pkVoice?.FeedText(token);
        }

        /// <summary>Flushes the trailing (sentence-incomplete) text into speech.</summary>
        protected void FlushVoiceText()
        {
            voice?.FlushText();
            cvVoice?.FlushText();
            kkVoice?.FlushText();
            pkVoice?.FlushText();
        }

        /// <summary>Hard-stops all speech (used when a reply is interrupted mid-generation).</summary>
        protected void StopVoices()
        {
            voice?.StopSpeaking();
            cvVoice?.StopSpeaking();
            kkVoice?.StopSpeaking();
            pkVoice?.StopSpeaking();
        }

        /// <summary>Dialogue-close speech stop: Kokoro/pocket-tts fade smoothly to silence over
        /// ~<paramref name="seconds"/> (never a mid-word cut); the legacy engines hard-stop.</summary>
        protected void FadeOutVoices(float seconds = 1f)
        {
            voice?.StopSpeaking();
            cvVoice?.StopSpeaking();
            kkVoice?.FadeOutAndStop(seconds);
            pkVoice?.FadeOutAndStop(seconds);
        }

        /// <summary>Trigger-contact hook for subclasses (called from their OnTriggerEnter/2D).</summary>
        protected void OnPlayerContact()
        {
            if (interactPrompt != null && state == NPCState.Idle)
                interactPrompt.Show();

            // contact loading (no prefetch zone): start streaming the TTS + LLM weights the
            // moment the player reaches the interaction trigger
            if (!usePrefetchZone)
            {
                // same reason as the zone path in Update: these streams start while state is Idle, so
                // without this they run on the leftover/default level rather than this NPC's
                if (BackendTradeoffTable.Level != backendTradeoff)
                    BackendTradeoffTable.Level = backendTradeoff;
                kkVoice?.PrefetchNow();
                kkVoice?.PrewarmKernels();
                cvVoice?.PrefetchNow();
                pkVoice?.PrefetchNow();
                pkVoice?.PrewarmKernels();
                EnsureLlm();
            }
        }

        /// <summary>Trigger-exit hook for subclasses (called from their OnTriggerExit/2D).</summary>
        protected void OnPlayerLeft()
        {
            if (interactPrompt != null)
                interactPrompt.Hide();

            // contact-loading mode: the talk trigger IS the residency zone — walking off it
            // unloads what contact loaded (the prefetch zone owns residency otherwise, and its
            // exit is handled in Update). Same rules as the zone: never mid-dialogue, the LLM
            // release defers past an in-flight conversation-KV save.
            if (!usePrefetchZone && state == NPCState.Idle)
            {
                kkVoice?.DefetchNow();
                cvVoice?.DefetchNow();
                pkVoice?.DefetchNow();
                StartCoroutine(ReleaseLlmAfterKvSave());
            }
        }

        // tier (b) resume: the recorded conversation rides in as prompt prefix, so the history
        // lands back in the KV cache through the normal chunked prefill (the same seeding trick
        // LLM.Compact uses for its HISTORY block). The transcript only ever holds the turns
        // SINCE the last compaction — the HISTORY block stands in for everything before it,
        // mirroring exactly what CompactConversationRoutine seeded into the live KV.
        private string BuildResumePrompt()
        {
            var sb = new StringBuilder(EffectiveSystemPrompt);
            if (string.IsNullOrEmpty(compactSummary) && transcript.Count == 0) return sb.ToString();
            // exactly [system prompt] + ## MEMORY + the history, nothing else: same shape
            // the compaction re-seed produces, so the model only ever learns one layout
            sb.Append("\n\n").Append(LLM.HISTORY_HEADING).Append('\n');
            if (!string.IsNullOrEmpty(compactSummary)) sb.Append(compactSummary);
            if (transcript.Count > 0)
            {
                if (!string.IsNullOrEmpty(compactSummary)) sb.Append("\n\n");
                sb.Append(BuildRecentTurnsBlock(transcript.Count));
            }
            return sb.ToString();
        }

        // The last <paramref name="lastN"/> turns formatted as a resume block (the full resume
        // prompt above and the post-summary context of a compaction share this shape).
        private string BuildRecentTurnsBlock(int lastN)
        {
            // No framing sentence: the heading above already says what this is, and every extra word here
            // is a line the model sees at the start of every resumed conversation (user spec 2026-07-25).
            var sb = new StringBuilder();
            for (int i = Mathf.Max(0, transcript.Count - lastN); i < transcript.Count; i++)
            {
                var t = transcript[i];
                // a tool turn was never spoken by the player — replay it as what it was, the result
                // that came back from a tool, so the resumed model reads the exchange correctly
                sb.Append(t.tool ? "\n[Tool result: " : "\nPlayer: ").Append(t.user).Append(t.tool ? "]" : "");
                if (!string.IsNullOrEmpty(t.npc))
                    sb.Append('\n').Append(NpcName).Append(": ").Append(t.npc);
            }
            return sb.ToString();
        }

        // ---------------------------------------------------------------- background compaction

        // The "Compacting…" pulse: shown while a background compaction owns THIS NPC's model,
        // blocking input until it lands. Used both when REOPENING onto an in-flight compact and
        // when compaction is triggered LIVE at the context limit (mid-session or on open).
        private IEnumerator ShowCompactingUntilDone(INPCChatWindow w)
        {
            w.AddMessage(NpcName, StatusStyled("Compacting.."));
            string[] frames = { "..", "...", "." };
            int fi = 0;
            float next = Time.unscaledTime + 0.4f;
            // the pulse dies with the dialogue (compaction itself continues in the background) —
            // a closed window must not keep receiving bubbles
            while (compactingNpc != null && compactingNpc.llm == llm && state != NPCState.Idle)
            {
                if (Time.unscaledTime >= next)
                {
                    w.PopLastMessage();
                    w.AddMessage(NpcName, StatusStyled("Compacting" + frames[fi++ % 3]));
                    next = Time.unscaledTime + 0.4f;
                }
                yield return null;
            }
            w.PopLastMessage();
        }

        // ---------------------------------------------------------------- manual reset
        //
        // STATE 0 (user spec 2026-07-28): "reset conversation efectiv trimite modelul in starea 0,
        // doar cu sys prompt si fara memorie" — the system prompt, nothing else, no memory. The end
        // state is IDENTICAL in both history modes; ResumeFromCompact differs only in having had a
        // ## MEMORY block to throw away, which makes its post-reset prompt SHORTER than the prefix it
        // was running on. There is deliberately no `historyMode` branch anywhere below.
        //
        // HOW, and why it is not clever: the text is cleared and the model is RE-INITIALIZED on the
        // bare system prompt. That is a full re-establish of the whole cache — see the warning on
        // ResetConversationRoutine before reaching for anything cheaper-looking.
        //
        // Same 10 s budget the other cooperative waiters use (CloseConversationWhenReplyUnwinds,
        // InterruptThenAsk): a canceled reply lands at its next TOKEN boundary, so this is a
        // stuck-model allowance, not an expected wait.
        private const float RESET_SETTLE_SECONDS = 10f;

        /// <summary>
        /// Send this NPC back to STATE 0 — its system prompt and nothing else — right-click the
        /// component → "Reset Conversation", or call it from an in-window button.
        /// <para>Everything a conversation consists of goes: the recorded turns, the compact summary
        /// (the <c>## MEMORY</c> block, so a ResumeFromCompact NPC forgets what it remembered too),
        /// the on-disk snapshot, AND the LIVE KV cache — which is what used to be missing. Until
        /// 2026-07-28 this method changed bookkeeping only: the window cleared and the context bar
        /// did not move, because the bar reads <c>llm.CurrentContextTokens</c> and the model was
        /// still holding the entire conversation. Every reply for the rest of that session was still
        /// answered with the full pre-reset context (and got recorded into the now-"empty"
        /// transcript, so a later reopen resumed from it) — a reset that reset the UI and nothing
        /// else.</para>
        /// <para>IMMEDIATE, dialogue open or not: when it lands, the window, the context bar and the
        /// model's KV all agree, because the bar's number is genuinely the system prompt's length.</para>
        /// <para>Two shapes, because the button has two very different callers. <b>With a live
        /// conversation of OURS on a model</b> it runs as a coroutine — in-flight work has to be
        /// settled before the model may be touched (see <see cref="ResetConversationRoutine"/>).
        /// <b>Without one</b> — the inspector button in EDIT mode, where there is no <c>llm</c> at
        /// all; a released model; or a POOLED instance whose KV currently belongs to a sibling NPC —
        /// it is synchronous and never touches a model.</para>
        /// </summary>
        [ContextMenu("Reset Conversation")]
        public void ResetConversation()
        {
            // FIRST, on every path: the live KV is no longer a record of anything worth keeping.
            // Dropping the flag here rather than at the end is what makes a close landing mid-reset
            // harmless — CloseConversation gates its disk snapshot on chatLive, so it cannot
            // re-persist the conversation we are in the middle of deleting.
            chatLive = false;

            // `Application.isPlaying` is load-bearing, not defensive: StartCoroutine throws in edit
            // mode, and the inspector button in edit mode is a path that already worked and must keep
            // working. OwnsConversation is the pooled-instance guard (audit #1): if the shared KV
            // carries a SIBLING's conversation, re-initializing the model would delete THEIR chat, so
            // we clear only what is ours and leave the GPU state alone.
            bool oursOnTheGpu = Application.isPlaying && llm != null && LLMPool.OwnsConversation(llm, this);
            if (!oursOnTheGpu)
            {
                ForgetConversation();
                ShowResetInWindow();
                ConsoleMessage.Info($"[Reset] {NpcName}: conversation wiped (manual reset) — " +
                                    (llm == null
                                        ? "no model resident, so there was no live KV to clear."
                                        : "the shared model's KV belongs to another NPC and was left untouched."));
                return;
            }
            if (resetRoutine != null) return;   // already resetting; a second click is not a second reset
            resetRoutine = StartCoroutine(ResetConversationRoutine());
        }

        /// <summary>
        /// The live half of <see cref="ResetConversation"/>: settle, forget, re-initialize.
        ///
        /// <para><b>Do not "optimise" the re-initialize into a cache rewind.</b>
        /// <c>Qwen3_5Cache.CachedTokenCount</c> has a setter and the K/V layout is token-major, so
        /// walking the cursor back to the prompt length looks like the same thing for free. It is
        /// silently wrong on this model: Qwen3.5 is HYBRID — 18 Gated DeltaNet layers next to 6
        /// full-attention ones — and the DeltaNet layers hold <c>conv_state</c>/<c>recurrent_state</c>,
        /// running state that is not indexed by token position and cannot be truncated. A cursor
        /// rewind forgets the conversation in the attention layers and keeps it in the other
        /// eighteen: no error, no exception, and nothing a smoke test would notice. A full
        /// re-initialize rebuilds K/V AND the SSM states from zero, so that failure mode cannot
        /// arise at all — and it is not the expensive option it looks like, because
        /// <c>InitializeChat</c> hits the system-prompt KV disk cache
        /// (<c>qwen35_prompt_&lt;owner&gt;.kv</c>) and pays a frame-budgeted upload instead of a
        /// prefill. Qwen3_5ResetProbe gate 2 is the regression guard: it reads the DeltaNet buffers
        /// back and fails on any implementation that leaves them holding the old conversation.</para>
        ///
        /// <para>Ordering is the entire content of this method, so it is spelled out:
        /// <list type="number">
        /// <item>let an OPENING dialogue finish — it is the thing establishing the prefix we are
        ///       about to replace, so resetting under it would race its own init/restore;</item>
        /// <item>bump <see cref="dialogueEpoch"/>, this file's one signal for "everything in flight,
        ///       stand down". It retires the coroutines a reset must not let run: an open
        ///       AskUserQuestion panel (its pick would come back as a &lt;tool_response&gt; to a call
        ///       that is no longer in the context), a pending DispatchAfterSpeaking, a queued
        ///       InterruptThenAsk. It also retires OUR OWN window tail, which is why every UI touch
        ///       at the bottom is re-checked against it;</item>
        /// <item>cancel a generating reply COOPERATIVELY and wait for it to unwind. StopCoroutine is
        ///       the fallback after the deadline, never the first move — and a reset is the one
        ///       caller for which that fallback is genuinely safe, because a half-written KV is
        ///       about to be overwritten wholesale; only the Busy guard has to be released by hand;</item>
        /// <item>wait out a COMPACTION — ours, or a sibling's on the same pooled instance. Never
        ///       canceled once its Chat started (house rule, enforced everywhere in this file);</item>
        /// <item>wait out an in-flight conversation-KV save on this instance. It holds the model's
        ///       Busy guard, and it is writing the very file step 6 deletes — deleting first would
        ///       leave the snapshot behind, which is precisely the class of bug this change is
        ///       about;</item>
        /// <item>forget the conversation (fields + sidecar + disk), then</item>
        /// <item>re-initialize the model on the bare system prompt, and re-claim the shared KV only
        ///       if that actually happened.</item>
        /// </list></para>
        /// </summary>
        private IEnumerator ResetConversationRoutine()
        {
            // The gate MUST always drop: an exception in here would otherwise brick the NPC forever
            // (AskNPC refuses while resetting and OpenConversation waits on it) — audit #12's rule.
            try
            {
                // 1. an opening dialogue owns the prefix — let it land.
                float openDeadline = Time.unscaledTime + RESET_SETTLE_SECONDS;
                while (state == NPCState.PreparingForInteraction && Time.unscaledTime < openDeadline)
                    yield return null;

                // 2. retire everything in flight.
                int epoch = ++dialogueEpoch;

                // 3. stop the reply at its next token boundary and wait for it to unwind. No
                //    MarkReplyCutShort: the transcript this would annotate is about to be deleted.
                replyCanceled = true;   // its tail must not flush the voice or fire OnReplyFinished
                llm?.CancelChat();
                StopVoices();           // he is not finishing a line about a conversation that no longer exists
                StopThinkingDots();
                StopRevealJob();
                float replyDeadline = Time.unscaledTime + RESET_SETTLE_SECONDS;
                while (dialogueCoroutine != null
                       && (compactRoutine != null || Time.unscaledTime < replyDeadline))
                    yield return null;
                if (dialogueCoroutine != null)
                {
                    ConsoleMessage.Warning($"[Reset] {NpcName}: the in-flight reply did not unwind in " +
                                           $"{RESET_SETTLE_SECONDS:0}s — abandoning it. Safe here, and only here: " +
                                           "the reset rebuilds the whole KV, so a half-written one is discarded anyway.");
                    StopCoroutine(dialogueCoroutine);
                    dialogueCoroutine = null;
                    // StopCoroutine skips Guarded's finally, so the abandoned Chat would leave the
                    // (pooled) instance latched and every later Chat/InitializeChat would refuse forever.
                    llm?.AbandonGuardedOperation();
                }

                // 4. a compaction is never canceled — not ours, not a sibling's on this instance.
                while (compactRoutine != null || (compactingNpc != null && compactingNpc.llm == llm))
                    yield return null;
                // 5. ...and a conversation-KV save holds the Busy guard AND writes the file step 6 deletes.
                while (KvSaveInFlightFor(llm)) yield return null;

                // The presentation state a discarded reply leaves behind — the same block
                // CloseInteraction clears, for the same reason: a leaked revealActive or
                // exchangePrefix resurrects the deleted reply's text in the next bubble.
                revealActive = false;
                pendingFullReply = null;
                spokenShown = null;
                bubbleLive = null;
                drainTurn = null;
                exchangePrefix = "";
                internalToolCalls = 0;
                toolRefusalSent = false;
                interruptPending = false;
                // An open interactive panel (choice OR offer) is orphaned by the reset — the call it
                // belongs to is gone from the context — so tear it down here instead of waiting for the
                // routine to notice the epoch moved, and the panel never outlives its conversation. Its
                // own teardown may run too; both Hide calls are idempotent.
                toolQuestionOpen = false;
                awaitingToolDispatch = false;
                (Window as INPCToolQuestionWindow)?.HideToolQuestion();
                (Window as INPCToolGiveWindow)?.HideToolGive();

                // 6. everything outside the model.
                ForgetConversation();

                // 7. ...and the model itself. Re-assert both cache settings first: on a POOLED
                //    instance a sibling's open stamped its own CacheOwnerKey, and a compaction/resume
                //    prefill borrows DiskKVCache — writing OUR prompt state under THEIR file name
                //    would thrash their cache (the header hash keeps it correct, just wasteful).
                bool reinitialized = false;
                if (llm != null)
                {
                    llm.DiskKVCache = cacheKVCache;
                    llm.CacheOwnerKey = ConversationKvKey();
                    // Busy is checked BEFORE the call on purpose: InitializeChat is Guarded, and
                    // Guarded declines a busy instance with a warning plus a bare yield break —
                    // indistinguishable from success to the caller (LLM.cs:~368). A reset that
                    // silently did nothing is the bug being fixed, so it is never assumed to have
                    // happened. Nothing can grab the guard between this check and the first
                    // MoveNext: coroutines are single-threaded and steps 3-5 already settled every
                    // operation that could hold it.
                    if (llm.Busy)
                        ConsoleMessage.Warning($"[Reset] {NpcName}: the model is still busy after the settle " +
                                               "waits — skipping the live re-initialize (it would be refused).");
                    else
                    {
                        yield return llm.InitializeChat(system_prompt: EffectiveSystemPrompt);
                        reinitialized = true;
                    }
                }
                if (reinitialized)
                {
                    // The KV now holds OUR conversation again: an empty one, on the bare prompt.
                    chatLive = true;
                    LLMPool.ClaimConversation(llm, this);
                    ConsoleMessage.Info($"[Reset] {NpcName}: state 0 — system prompt only, no memory " +
                                        $"({ContextTokensNow()} of {maxContextLength} context tokens).");
                }
                else
                {
                    // chatLive stays false, so the next open re-initializes from scratch; the
                    // transcript, the memory and the disk snapshot are already gone, so there is
                    // nothing left that could bring the conversation back either way.
                    ConsoleMessage.Warning($"[Reset] {NpcName}: the conversation is gone (transcript, memory and " +
                                           "disk snapshot), but the LIVE KV was not re-initialized — it will be " +
                                           "rebuilt on the next open.");
                }

                if (epoch != dialogueEpoch) yield break;   // a close+reopen owns the window now — never touch its state
                ShowResetInWindow();
                var w = Window;
                if (w != null && state != NPCState.Idle)
                {
                    // Step 2 retired the reply's own tail, so the "ready for the next line" state it
                    // would have restored is restored here instead.
                    w.SetSendLoading(false);
                    state = NPCState.WaitingInInteraction;
                    w.InputField?.ActivateInputField();
                }
            }
            finally { resetRoutine = null; }
        }

        /// <summary>Everything a conversation consists of OUTSIDE the model: the recorded turns, the
        /// compact (and its sidecar, so the inspector stops showing a memory that no longer exists),
        /// our claim on the shared KV, and the on-disk snapshot. No <c>historyMode</c> branch — state
        /// 0 is state 0 in both modes.</summary>
        private void ForgetConversation()
        {
            transcript.Clear();
            compactSummary = null;
            SaveCompactSidecar();   // an empty summary DELETES the sidecar — a reset must leave no memory behind
            activeTurn = null;      // a reply that was mid-generation is never going to be finished
            activeResponse = null;
            if (llm != null)
            {
                if (LLMPool.OwnsConversation(llm, this))
                    LLMPool.ClaimConversation(llm, null);   // we no longer vouch for what is in the shared KV
                if (cacheKVCache) llm.DeleteConversationKV(ConversationKvKey());
            }
            // ...and delete the snapshot WITHOUT a model too (fix 2026-07-28). The branch above is
            // unreachable from the inspector button in edit mode — there is no llm then — so the
            // button cleared the in-memory transcript (which is not serialized anyway) and left the
            // on-disk conversation untouched. Next play restored it verbatim: "conversation restored
            // from disk (1011 tokens)", i.e. a reset that reset nothing. Deliberately NOT gated on
            // cacheKVCache either: if a file is there, a reset removes it, whatever the toggle says
            // now. Family-agnostic pattern rather than the Qwen-specific name, so this does not have
            // to learn about every model that persists a conversation.
            DeleteConversationSnapshots();
        }

        /// <summary>The window half of a reset: an emptied transcript view, the notice, and the
        /// context bar pushed to what the model is NOW holding. The bar is written here as well as
        /// per-frame in <see cref="Update"/> so it agrees with the cleared window in the SAME frame —
        /// that disagreement (empty chat, full bar) was the visible tell of the old reset.
        /// No-ops without an open dialogue: the inspector button in edit mode has no window, and a
        /// closed one is repainted from scratch on the next open anyway.</summary>
        private void ShowResetInWindow()
        {
            var w = Window;
            if (w == null || state == NPCState.Idle) return;
            w.Clear();
            w.SetInfoText("— conversation reset —");
            w.SetContextFill(ContextTokensNow() / (float)Mathf.Max(1, maxContextLength));
            if (w.SendButton != null) w.SendButton.interactable = true;
        }

        /// <summary>Delete this NPC's on-disk conversation snapshots, with or without a live model.
        /// Matches <c>&lt;family&gt;_conv_&lt;key&gt;.kv</c> and the legacy
        /// <c>&lt;family&gt;_conv_&lt;key&gt;_&lt;hash&gt;.kv</c> in the shared DeepUnity cache directory —
        /// the same two shapes <c>Qwen3_5.DeleteConversationKV</c> knows about, matched by convention so
        /// this stays model-agnostic like the rest of NPCChatBase.</summary>
        private void DeleteConversationSnapshots()
        {
            try
            {
                string dir = System.IO.Path.Combine(Application.persistentDataPath, "DeepUnity");
                if (!System.IO.Directory.Exists(dir)) return;
                string key = ConversationKvKey();
                int n = 0;
                foreach (var f in System.IO.Directory.GetFiles(dir, $"*_conv_{key}*.kv"))
                {
                    // The glob has to allow the legacy `<key>_<contexthash>` suffix, which means it also
                    // matches a SIBLING whose name merely starts with this key: key "Anya" would glob
                    // "..._conv_Anya_Two.kv" and a reset on Anya would wipe Anya Two's conversation.
                    // So accept only the exact name or the key followed by a pure hex hash.
                    string name = System.IO.Path.GetFileNameWithoutExtension(f);
                    int at = name.IndexOf("_conv_", System.StringComparison.Ordinal);
                    if (at < 0) continue;
                    string tail = name.Substring(at + 6);
                    if (tail != key)
                    {
                        if (!tail.StartsWith(key + "_", System.StringComparison.Ordinal)) continue;
                        string suffix = tail.Substring(key.Length + 1);
                        if (suffix.Length == 0) continue;
                        bool hex = true;
                        foreach (char c in suffix)
                            if (!System.Uri.IsHexDigit(c)) { hex = false; break; }
                        if (!hex) continue;
                    }
                    System.IO.File.Delete(f); n++;
                }
                if (n > 0) ConsoleMessage.Info($"[Reset] {NpcName}: deleted {n} on-disk conversation snapshot(s).");
            }
            catch (System.Exception e)
            { ConsoleMessage.Warning($"[Reset] {NpcName}: could not delete the conversation snapshot — {e.Message}"); }
        }

        // ---------------------------------------------------------------- compact sidecar
        // Unity throws away play-mode changes to serialized fields when you press Stop, so a compact
        // that happened while playing would vanish from the inspector the moment the session ended.
        // It is written next to the KV caches instead, and read back when the component loads in the
        // EDITOR — so a compact from the last play session is still on the inspector afterwards, and
        // across editor restarts (user spec 2026-07-25: "it is persistent, the script reads and sees
        // whether a cache exists"). Plain assignment, no SetDirty: the scene is NOT marked dirty, so
        // this never sneaks a stale memory into a committed scene — save it yourself if you want that.
        private string CompactSidecarPath()
            => System.IO.Path.Combine(Application.persistentDataPath, "DeepUnity",
                                      $"npc_compact_{ConversationKvKey()}.txt");

        private void SaveCompactSidecar()
        {
            try
            {
                string path = CompactSidecarPath();
                System.IO.Directory.CreateDirectory(System.IO.Path.GetDirectoryName(path));
                if (string.IsNullOrEmpty(compactSummary)) System.IO.File.Delete(path);
                else System.IO.File.WriteAllText(path, compactSummary);
            }
            catch (System.Exception e)
            {
                ConsoleMessage.Warning($"[Compact] {NpcName}: could not persist the compact: {e.Message}");
            }
        }

        private void LoadCompactSidecar()
        {
            try
            {
                string path = CompactSidecarPath();
                if (System.IO.File.Exists(path)) compactSummary = System.IO.File.ReadAllText(path);
            }
            catch { }   // unreadable sidecar: the inspector just shows nothing
        }

#if UNITY_EDITOR
        // edit mode only — at runtime the field is owned by the live conversation
        protected virtual void OnEnable()
        {
            if (!Application.isPlaying && string.IsNullOrEmpty(compactSummary)) LoadCompactSidecar();
        }
#endif

        // ~4 chars/token: a model-agnostic estimate of what the conversation costs in context
        // (the HISTORY compact counts too once one exists — compaction re-triggers when the
        // compacted state has grown long again).
        private int EstimatedTranscriptTokens()
        {
            int chars = string.IsNullOrEmpty(compactSummary) ? 0 : compactSummary.Length;
            foreach (var t in transcript)
                chars += (t.user?.Length ?? 0) + (t.npc?.Length ?? 0);
            return chars / 4;
        }

        // Live KV occupancy when the model reports it (accurate), else the chars/4 transcript
        // estimate — the measure ResumeFromCompact (compact) tests against maxContextLength.
        private int ContextTokensNow()
            => llm != null && llm.CurrentContextTokens >= 0 ? llm.CurrentContextTokens : EstimatedTranscriptTokens();

        // The conversation has reached the context window and this mode must act on it.
        private bool ContextFull() => ContextTokensNow() >= maxContextLength;

        /// <summary>There is actual CONVERSATION to compact. The system prefix (persona + any
        /// # Tools block) is charged to the same budget but compaction cannot shrink it, so a
        /// prefix that alone fills the window would otherwise trigger an endless compact loop —
        /// on every open and after every reply.</summary>
        private bool HasCompactableHistory() => transcript.Count > 0 || !string.IsNullOrEmpty(compactSummary);

        /// <summary>Report a system prefix that eats its own context budget. This has bitten once
        /// for real (enabling the AskUserQuestion tools block pushed a 400-token NPC over its
        /// limit, so it compacted on every open forever); the symptom reads like a model or engine
        /// bug, so name the cause and the fix explicitly. Called once per opened dialogue.</summary>
        private void WarnIfPrefixOverBudget()
        {
            int prefix = ContextTokensNow();
            if (prefix * 4 < maxContextLength * 3) return;   // under 75% — plenty of room to talk
            int suggested = prefix * 4 / 3 + 64;
            string fix = enableAskUserQuestion || enableGiveTool
                ? $"raise Max Context Length to >= {suggested}, shorten the description, or turn an " +
                  "interactive tool off (AskUserQuestion's schema plus the shared call-format block is " +
                  "~520 tokens, measured; GiveTool's own schema is ~130 on top of that block)"
                : $"raise Max Context Length to >= {suggested} or shorten the system prompt";
            string msg = $"[NPC] {NpcName}: the system prompt alone is ~{prefix} tokens of a " +
                         $"{maxContextLength}-token budget — {fix}.";
            if (prefix >= maxContextLength)
                ConsoleMessage.Error(msg + " It does not fit AT ALL: the NPC cannot hold a conversation " +
                                     "and (in ResumeFromCompact) would try to compact forever.");
            else
                ConsoleMessage.Warning(msg);
        }

        /// <summary>The EXACT request the model is asked when compacting (ResumeFromCompact hitting
        /// Max Context Length). It is one bare USER turn, continuing the tracked conversation —
        /// greedy (temperature 0), capped at 256 tokens. The model's reply IS the compact, which is
        /// then re-seeded as the KV prefix:
        /// <code>[this NPC's description and rules]\n\n## MEMORY\n[the model's reply]</code>
        /// so the NPC "remembers" everything through the HISTORY block on a short prefix.
        /// This mirrors the engine constant (single source of truth); change the wording there.</summary>
        public const string CompactRequestPrompt = LLM.COMPACT_PROMPT;   // == "Compact the conversation."

        // THE COMPACTION STANDARD (user spec 2026-07-15):
        //   WHEN — the conversation reaches maxContextLength, but NEVER before the limit-hitting
        //   reply is fully delivered: decoded, typed AND spoken to the end (+8192 KV headroom
        //   absorbs the overshoot). "Compacting…" appears only after the voice goes quiet; input
        //   is blocked until it lands. WINDOW — keeps the entire visible conversation through
        //   the compaction; only the NEXT open collapses it to the dimmed compact block
        //   (RepopulateWindow). RECOVERY — if a compact never landed (player left during the
        //   speech wait / game closed mid-compact), the next open compacts behind the same pulse.
        // Steps, on the resident model:
        //   1. wait out the full-state snapshot CloseConversation kicked (it reads the same GPU
        //      buffers this will re-write),
        //   2. LLM.Compact — the model answers "Compact the conversation." with a one-shot
        //      compact of the whole history, then the chat recomputes as [system + ## MEMORY +
        //      compact] (a short KV prefix),
        //   3. reset the transcript (the HISTORY block now IS the history) and keep the compact,
        //   4. snapshot the compacted state to disk (overwrites the pre-compact save).
        // NEVER canceled once its Chat started (user rule): a dialogue reopen WAITS on it behind
        // a "Compacting…" pulse, and residency release waits too — the model stays on GPU until
        // the compact + its KV snapshot land. The only bail-out is the guard below, BEFORE the
        // model is touched (e.g. the player re-engaged while we still waited on the KV save).
        private IEnumerator CompactConversationRoutine()
        {
            compactingNpc = this;
            while (KvSaveInFlightFor(llm)) yield return null;
            // Fired at the context limit: on open (PreparingForInteraction, crash recovery), right
            // after a reply (WaitingInInteraction), or on a resident close (Idle). Bail only if the
            // conversation is gone, a reply is actively generating, or another NPC owns the model.
            if (!chatLive || state == NPCState.TalkingInInteraction || llm == null
                || !LLMPool.OwnsConversation(llm, this))
            {
                compactRoutine = null;
                compactingNpc = null;
                yield break;
            }
            ConsoleMessage.Info($"[Compact] {NpcName}: compaction started at the context limit " +
                                $"(~{ContextTokensNow()} tokens ≥ {maxContextLength})");

            // the re-seeded prefix is one-shot (same rule as the resume prefill) — don't let
            // InitializeChat's system-prompt cache write an orphan file for it
            llm.DiskKVCache = false;
            string summary = null;
            try
            {
                var compact = llm.Compact(EffectiveSystemPrompt, s => summary = s);
                while (compact.MoveNext()) yield return compact.Current;
            }
            finally
            {
                // an exception mid-compact must never brick the NPC (permanent pulse, blocked
                // sends, pinned model) — gates always drop, cache flag always restored (audit #12)
                if (llm != null) llm.DiskKVCache = cacheKVCache;
                compactRoutine = null;
                if (compactingNpc == this) compactingNpc = null;
            }
            if (string.IsNullOrEmpty(summary))
            {
                ConsoleMessage.Warning($"[Compact] {NpcName}: model returned an empty compact — " +
                                       "keeping the full state, will retry at the next clean close");
                yield break;
            }

            compactSummary = summary;
            SaveCompactSidecar();   // so the inspector still shows it after play mode ends
            transcript.Clear();   // the HISTORY block stands in for every turn so far
            LLMPool.ClaimConversation(llm, this);   // the compacted prefix carries OUR conversation
            ConsoleMessage.Info($"[Compact] {NpcName}: compaction done — history → " +
                                $"{summary.Length}-char HISTORY block, KV recomputed");
            // the compact text itself, so its quality can be inspected in the console
            ConsoleMessage.Info($"[Compact] {NpcName}: <i>{summary.Trim()}</i>");
            if (cacheKVCache && chatLive && !KvSaveInFlightFor(llm))
                StartCoroutine(SaveConversationKvRoutine());
        }

        // RepopulateWindow was deleted 2026-07-28. It was the older, thinner painter (no compact
        // banner, drew a tool turn's <tool_response> as a player line) and RepaintTranscript
        // superseded it — but its call site was left behind, so both ran on every open and drew the
        // history twice. One painter, one call site; do not reintroduce a second one.

        // ---------------------------------------------------------------- conversation KV disk cache

        // Stable per-NPC key for the on-disk conversation snapshot (the LLM adds the
        // model/quant/kv-quant/system-prompt hash to the file name itself).
        private string ConversationKvKey()
        {
            if (string.IsNullOrEmpty(NpcName)) return "npc";
            var sb = new StringBuilder(NpcName.Length);
            foreach (char c in NpcName)
                sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');
            return sb.ToString();
        }

        // The transcript rides inside the conversation KV file as an opaque JSON string, so a
        // restore after a scene reload brings the visible history (and the re-prefill fallback
        // source) back together with the KV.
        private string SerializeTranscript()
        {
            var st = new TranscriptState();
            st.turns.AddRange(transcript);
            st.summary = compactSummary;   // a compacted state is meaningless without its briefing
            return JsonUtility.ToJson(st);
        }

        // TryRestoreConversationKV validator, called with the saved transcript BEFORE any KV
        // upload: adopt it when we hold none (fresh instance / scene reload), accept when it
        // matches ours exactly, veto otherwise — a mismatch means our in-memory transcript is
        // fuller (an interrupted reply landed after the last snapshot) and what the player saw
        // is ground truth, so the caller falls back to re-prefilling it.
        private bool AcceptRestoredTranscript(string savedTranscript)
        {
            if (transcript.Count == 0)
            {
                AdoptTranscript(savedTranscript);
                return true;
            }
            return savedTranscript == SerializeTranscript();
        }

        private void AdoptTranscript(string json)
        {
            transcript.Clear();
            compactSummary = null;
            if (string.IsNullOrEmpty(json)) return;
            try
            {
                var st = JsonUtility.FromJson<TranscriptState>(json);
                if (st?.turns != null) transcript.AddRange(st.turns);
                if (!string.IsNullOrEmpty(st?.summary)) compactSummary = st.summary;
            }
            catch { }   // unreadable history — the KV still restores; the window just starts blank
        }

        // Spreads the post-conversation cleanup over ~2 ms slices per frame instead of one
        // blocking GC.Collect (~400 ms). Incremental GC is enabled in Project Settings; if it
        // were disabled, CollectIncremental no-ops and the next natural collection handles it.
        private IEnumerator CollectGarbageIncremental()
        {
            while (UnityEngine.Scripting.GarbageCollector.CollectIncremental(2_000_000UL))
                yield return null;
        }

        // the residency zone: FILLED very-transparent green shape PLUS its wire outline —
        // a solid disc + circle for 2D NPCs, a solid sphere + wire sphere for 3D ones
        // (visible with Gizmos enabled)
        private void OnDrawGizmos()
        {
            if (!usePrefetchZone) return;
            var fill = new Color(0f, 1f, 0.35f, 0.14f);
            var wire = new Color(0f, 1f, 0.35f, 0.35f);
            if (ZoneIs2D)
            {
#if UNITY_EDITOR
                UnityEditor.Handles.color = fill;
                UnityEditor.Handles.DrawSolidDisc(transform.position, Vector3.forward, prefetchRadius);
                UnityEditor.Handles.color = wire;
                UnityEditor.Handles.DrawWireDisc(transform.position, Vector3.forward, prefetchRadius);
#endif
            }
            else
            {
                Gizmos.color = fill;
                Gizmos.DrawSphere(transform.position, prefetchRadius);
                Gizmos.color = wire;
                Gizmos.DrawWireSphere(transform.position, prefetchRadius);
            }
        }
    }
}
