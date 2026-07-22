using System.Collections;
using System.Collections.Generic;
using System.Text;
using TMPro;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity
{
    /// <summary>
    /// The chat-window surface NPCChatBase drives. SoulsChatWindow (3D) and ChatWindow2D (2D)
    /// already share this exact member set — the interface just names it so the base class can
    /// talk to either without knowing the concrete type.
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

        /// <summary>What happens to the conversation HISTORY between two openings of the dialogue.
        /// (GPU residency is NOT decided here — the prefetch zone / talk trigger owns that.)</summary>
        public enum HistoryMode
        {
            [Tooltip("The conversation ceases to exist the moment the chat CLOSES (same session): transcript wiped + live KV marked dead on close, fresh InitializeChat on the next open (the system-prompt KV disk cache still applies).")]
            ResetEveryTime,
            // NOTE: the former middle value ContinueWhereLeftOff was removed 2026-07-15 — halting at
            // the limit was pointless (a full conversation is simply over); ResumeFromCompact keeps
            // talking instead. The enum is kept CONTIGUOUS (0,1) so (int)value == enumValueIndex ==
            // the serialized value; the builders' SetEnum uses enumValueIndex, so a gap would throw
            // "enum index is out of range". Scenes that used the old value were remapped to this one.
            [Tooltip("Reopening resumes the SAME conversation (live KV while the model is resident, else restored from disk / re-prefilled). When it reaches Max Context Length the model COMPACTS itself: it summarizes the whole chat in one shot and the result rides in the system prompt as a HISTORY block, so talking continues on a short prefix forever. The KV is allocated larger than the limit so the compact pass has room. The limit-hitting reply is always delivered IN FULL (decoded, typed and spoken to the end) — 'Compacting…' appears only after the voice finishes (input blocked until it lands); the window keeps the whole conversation until the dialogue closes, and reopening starts visually empty (the compact lives only in the system prompt). Crash-recovery: compacts on the next open if one never landed. Never canceled once started.")]
            ResumeFromCompact,
        }

        [SerializeField, ViewOnly] protected NPCState state = NPCState.Idle;
        [SerializeField, UnityEngine.Serialization.FormerlySerializedAs("npc_name")] protected string NpcName = "Villager";
        [TextArea(4, 12)]
        [SerializeField] protected string system_prompt =
            "You are a friendly villager. Stay in character at all times. " +
            "Keep your replies to one to three short sentences.";

        [Header("Conversation")]
        [Tooltip("LlmOnly = text-only replies (talk animation follows the writing; voice fields hidden below). LlmPlusTts = replies are spoken: the talk animation follows the AUDIO, and the next sentence synthesizes while the current one plays.")]
        [SerializeField] protected ConversationMode conversationMode = ConversationMode.LlmOnly;
        protected bool speakReplies => conversationMode == ConversationMode.LlmPlusTts;
        [Tooltip("ResetEveryTime = wiped when the chat closes (same session). ResumeFromCompact = fully persistent (live KV while resident, restored from disk / re-prefilled after a release); once the conversation fills Max Context Length the model auto-compacts the history into a HISTORY block and keeps going (see the mode tooltip). The limit is driven by Max Context Length below.")]
        [SerializeField] protected HistoryMode historyMode = HistoryMode.ResetEveryTime;
        [Tooltip("Persist this NPC's KV cache to disk (persistentDataPath/DeepUnity): the system-prompt state in EVERY mode, plus — in the continue modes — the WHOLE conversation on a clean close (KV + sampler state + transcript), so reopening after the model was released (or the scene reloaded) restores the chat from disk instead of re-prefilling. Qwen3.5 only for now; Gemma3 NPCs fall back to the re-prefill path.")]
        [SerializeField] protected bool cacheKVCache = true;

        [Header("Text (LLM)")]
        [Tooltip("Which local LLM voices this NPC — the dropdown lists every model registered in LLMRegistry, so a freshly ported LLM appears here automatically. Sampling fields at -1 fall back to this model's Config presets.")]
        [SerializeField] protected string model = "Qwen3.5-0.8B";
        [Tooltip("Weight format. INT8 is ~lossless at half the VRAM — the recommended default. INT4 is lossy on models this small (Gemma int4 collapses outright).")]
        [SerializeField] protected LLMQuant quantization = LLMQuant.INT8;
        [Tooltip("Context window (tokens) — the conversation size the history mode acts on. ResumeFromCompact auto-compacts here (and allocates the KV +8192 above it for the compact pass). Sizes the KV cache (pre-allocated → more = more VRAM). 8192 default. Instances are pooled per (model, quant, KV, this length + headroom), so NPCs sharing a model should share this value.")]
        [SerializeField] protected int maxContextLength = 8192;
        [Tooltip("Let a thinking-capable model (Qwen3.5) reason in <think> before answering. The reasoning is NEVER voiced and never shown as reply text (the window's ShowThinkingTokens debug toggle can render it dimmed); while the model thinks, the dialog pulses an animated 'Thinking…' placeholder until the final answer starts. Non-thinking models ignore this.")]
        [SerializeField] protected bool allowThinking = false;

        [Header("Sampling (-1 = model preset)")]
        [SerializeField] protected float temperature = 0.8f;
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
        // "LLM Processing" (smooth <-> speed): drawn by NPCChatBaseEditor as its own labeled
        // slider at the end of this Sampling section.
        [Tooltip("The auto-detection ALWAYS computes reply pacing for a stable 60+ fps; this only biases around its result while you talk to THIS NPC. 0.5 = pure auto. Offsets multiply the measured budgets (toward Speed the frames may carry more decode → faster text; toward Smooth they carry less). The hard ends force the implementation limits: full Smooth = async decode + 1 layer/frame prefill, full Speed = sync decode + bulk prefill. Live: moving it mid-dialogue re-probes on the next reply.")]
        [Range(0f, 1f)] [SerializeField] protected float smoothVsSpeed = 0.5f;
        float appliedSmoothVsSpeed = -1f;   // last value pushed into InferencePerf (per dialogue)

        [Header("Voice (TTS)")]
        [Tooltip("PocketTTS = Kyutai 100M AR, RTF ~0.15 int8 (speaks in real time DURING generation, voice cloning — DEFAULT); Kokoro = 82M non-AR, RTF ~0.3; Chatterbox = clause-streamed (RTF~1.4); CosyVoice3 = streaming-native. 2D demo NPCs are Kokoro-only and ignore this.")]
        [SerializeField] protected TtsModel ttsModel = TtsModel.PocketTTS;
        [Tooltip("TTS weight format. PocketTTS: int8 = 116 MB vs 209 MB fp16, same speed, mel-gated parity (also picks which voice-clone cache dir is used). Chatterbox: int8 = T3 matmuls int8 (~300 MB less, parity-validated); s3gen stays fp16 either way. Kokoro/CosyVoice ignore this (their voice component's weightsPath decides).")]
        [SerializeField] protected LLMQuant ttsQuantization = LLMQuant.INT8;
        [Tooltip("Playback pitch for this NPC. 1 = natural (the voice's own timbre); <1 = deeper/slower.")]
        [SerializeField] protected float voicePitch = 1.0f;
        [Tooltip("Loudness of this NPC's voice. AudioSource.volume tops out at 1, so this multiplies the samples themselves — >1 = louder (peaks clamp at full scale).")]
        [Min(0f)] [SerializeField] protected float voiceVolume = 1.4f;
        [Tooltip("BAKED voice shipped inside the selected TTS engine's weights export (voices/<name> dirs for PocketTTS/CosyVoice3, voices/<name>.bin voicepacks for Kokoro) — the inspector dropdown lists what's on disk. Pick 'Clone (reference clip)' on PocketTTS to clone from an AudioClip instead; a non-null clip always overrides this name.")]
        [SerializeField] protected string ttsVoice = "jean";
        [Tooltip("PocketTTS only: reference clip to VOICE-CLONE for this NPC (overrides the baked ttsVoice). First runtime use encodes it once through the Mimi encoder and caches by content hash; press 'Precompute voice-clone cache' below to bake the embedding into the shared Resources/Cache so runtime (editor AND builds) is a pure load — no recompute, ever.")]
        [SerializeField] protected AudioClip clonedVoiceClip;
        [Tooltip("Sentences per spoken chunk. Smaller = faster response, lower quality (prosody resets each sentence); larger = higher quality (intonation flows across sentences), slower response.")]
        [Range(1, 3)] [SerializeField] protected int clausesPerChunk = 1;
        [Tooltip("PocketTTS pacing: pause between spoken chunks that ended at a sentence ender (. ! ?), in seconds. Inside a batched chunk the model renders its own natural pauses.")]
        [Min(0f)] [SerializeField] protected float sentencePauseSeconds = 0.36f;
        [Tooltip("PocketTTS pacing: pause after a chunk cut at a semicolon, in seconds.")]
        [Min(0f)] [SerializeField] protected float semicolonPauseSeconds = 0.2f;
        [Tooltip("PocketTTS pacing: pause after an emergency comma cut (very long run-on sentences), in seconds.")]
        [Min(0f)] [SerializeField] protected float commaPauseSeconds = 0.15f;
        [Tooltip("PocketTTS pacing: extra model-generated tail on the reply's last chunk, in seconds — lets the final word decay naturally instead of cutting ~0.16 s after it.")]
        [Min(0f)] [SerializeField] protected float replyTailSeconds = 0.32f;

        [Tooltip("The walk-up prompt (\"[I] Speak\" / \"Talk — [ E ]\") shown while the player is in the talk trigger. Its OWN component on its OWN GameObject (fade/bob/text knobs live there) — the NPC only calls Show/Hide on it.")]
        [SerializeField] protected NPCInteractPrompt interactPrompt;

        [Header("Prefetch Zone (A/B test)")]
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
        [Tooltip("BOTH weight streams (LLM + TTS) are spread over ~this many seconds of walking-up " +
                 "time. BIGGER = gentler per-frame upload budget (imperceptible, zero frame drops) " +
                 "but the models need longer to become ready — pair with a larger prefetchRadius. " +
                 "SMALLER = ready sooner after zone entry, but each frame uploads more bytes and " +
                 "weak GPUs/disks may show hitches during the walk-up. Opening the dialogue BOOSTS " +
                 "the stream to full speed regardless, so this only shapes the background portion. " +
                 "3s suits walking approaches; raise toward 5-10s for large zones or low-end hardware.")]
        [SerializeField] protected float slowPrefetchSeconds = 3f;

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

        [System.Serializable] private class Turn { public string user; public string npc; }
        [System.Serializable] private class TranscriptState { public List<Turn> turns = new List<Turn>(); public string summary; }
        private readonly List<Turn> transcript = new List<Turn>();
        private bool chatLive;
        private Turn activeTurn;                 // turn currently being generated (for interrupt finalize)
        private StringBuilder activeResponse;    // its streaming reply buffer
        // A background conversation-KV save is reading GPU state. Keyed PER LLM INSTANCE (audit
        // #9): pooled instances are shared across NPCs, so nobody may reset/forward THAT model
        // while a save still reads it — but a save on one model must not stall/skip an NPC on a
        // DIFFERENT one. Value = the NPC that latched the entry (OnDisable drops only its own).
        private static readonly Dictionary<LLM, NPCChatBase> kvSavesInFlight = new Dictionary<LLM, NPCChatBase>();
        private static bool KvSaveInFlightFor(LLM m) => m != null && kvSavesInFlight.ContainsKey(m);
        // ResumeFromCompact maintenance state: the compact standing in for every turn before it
        // (rides in the transcript JSON on disk), the in-flight compaction coroutine, and —
        // STATIC, same pooled-model reasoning as kvSavesInFlight — which NPC is compacting, so a
        // dialogue opening on the shared instance WAITS for it before driving the model itself
        // (user rule: a compaction is never canceled once its Chat started — the window pulses
        // "Compacting…" and input stays blocked until the compact lands).
        private string compactSummary;
        private Coroutine compactRoutine;
        private static NPCChatBase compactingNpc;

        // ---------------------------------------------------------------- subclass surface

        /// <summary>The shared chat window. IMPORTANT: implement with a Unity null check
        /// (<c>chatWindow != null ? chatWindow : null</c>) so an unassigned serialized field
        /// reads as real null through the interface.</summary>
        protected abstract INPCChatWindow Window { get; }
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

            // per-NPC Smooth⇄Speed preference: pushed to the (global) pacing runtime while THIS
            // NPC's dialogue is the active one; live slider moves re-probe on the next reply
            if (state != NPCState.Idle && !Mathf.Approximately(appliedSmoothVsSpeed, smoothVsSpeed))
            {
                bool reprobe = appliedSmoothVsSpeed >= 0f
                               || !Mathf.Approximately(InferencePerf.SmoothVsSpeed, smoothVsSpeed);
                InferencePerf.SmoothVsSpeed = smoothVsSpeed;
                if (reprobe) InferencePerf.ResetAutoTune();
                appliedSmoothVsSpeed = smoothVsSpeed;
            }
            else if (state == NPCState.Idle)
                appliedSmoothVsSpeed = -1f;   // re-arm for the next dialogue (another NPC may have set it)

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
                    pkVoice.sentencePauseSeconds = sentencePauseSeconds;
                    pkVoice.semicolonPauseSeconds = semicolonPauseSeconds;
                    pkVoice.commaPauseSeconds = commaPauseSeconds;
                    pkVoice.replyTailSeconds = replyTailSeconds;
                    pkVoice.clausesPerChunk = clausesPerChunk;
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

        // slowPrefetchSeconds applies to the LLM stream too. The legacy LLM loaders sample ONE
        // global per-frame budget (LLM.UploadBudgetBytes) live each frame — until WS-F gives them
        // per-instance budgets, this governor retargets that global every frame so the REMAINING
        // bytes land in roughly the walk-up window, then restores the full-speed budget. Ends
        // early (and boosts) when the dialogue opens, the model finishes, or the zone releases it.
        // The full-speed baseline is captured ONCE per session (static) — with overlapping zones
        // two governors run concurrently, and the second must not adopt the first one's slowed
        // value as "full speed".
        private Coroutine llmSlowJob;
        private static int llmFullBudget = -1;

        // Entry point for the zone: seeds a SLOW budget BEFORE the Acquire so the loader's own
        // "[GPU] ... SLOW prefetch started" line tells the truth from the first frame (the exact
        // remaining/window rate takes over on the next frames once totals are known).
        private void BeginLlmSlowPrefetch()
        {
            if (llmFullBudget < 0) llmFullBudget = LLM.UploadBudgetBytes;
            bool fresh = llm == null;
            if (fresh) LLM.UploadBudgetBytes = System.Math.Max(64 * 1024, llmFullBudget / 8);
            EnsureLlm();
            if (llmSlowJob != null) StopCoroutine(llmSlowJob);
            llmSlowJob = StartCoroutine(LlmSlowPrefetch());
        }

        private IEnumerator LlmSlowPrefetch()
        {
            float deadline = Time.unscaledTime + slowPrefetchSeconds;
            bool announced = false;
            while (llm != null && !llm.IsReady && state == NPCState.Idle
                   && Time.unscaledTime < deadline)
            {
                long remaining = llm.TotalWeightBytes - llm.UploadedWeightBytes;
                if (remaining > 0)
                {
                    long frames = (long)Mathf.Max(1f, (deadline - Time.unscaledTime) * 60f);
                    LLM.UploadBudgetBytes = (int)System.Math.Min(llmFullBudget,
                        System.Math.Max(64 * 1024, remaining / frames));
                    if (!announced)
                    {
                        // one line with the REAL retargeted rate + ETA for the walk-up window
                        ResidencyLog.Budget(llm.WeightsLabel, LLM.UploadBudgetBytes, remaining);
                        announced = true;
                    }
                }
                yield return null;
            }
            // boost: dialogue opened / window elapsed / released. Announce it only when there is
            // actually something left to stream — completion has its own "resident" line.
            if (llm != null && !llm.IsReady)
                ResidencyLog.Budget(llm.WeightsLabel, llmFullBudget,
                                    System.Math.Max(0, llm.TotalWeightBytes - llm.UploadedWeightBytes));
            LLM.UploadBudgetBytes = llmFullBudget;
            llmSlowJob = null;
        }

        // ---------------------------------------------------------------- interaction flow

        public void StartInteraction()
        {
            dialogueEpoch++;
            state = NPCState.PreparingForInteraction;
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
        ///                             [system prompt + HISTORY: compact], so every tier below
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
            llm.DiskKVCache = cacheKVCache;   // re-assert (a compaction/resume prefill clears it temporarily)

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
                        system_prompt, AcceptRestoredTranscript);
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
                    yield return llm.InitializeChat(system_prompt: resume ? BuildResumePrompt() : system_prompt);
                    llm.DiskKVCache = cacheKVCache;
                }
                chatLive = true;
                LLMPool.ClaimConversation(llm, this);   // the shared KV now carries OUR conversation
            }
            if (epoch != dialogueEpoch) yield break;   // closed during restore/prefill

            if (historyMode != HistoryMode.ResetEveryTime && transcript.Count > 0)
                RepopulateWindow();   // restore the turns since the last compaction (the compact itself stays invisible)

            // Context-window state now that the conversation KV is live (ResumeFromCompact only):
            // a state restored ABOVE the trigger means a previous compaction never landed (game
            // stopped mid-compact) — compact it now, before the player talks, behind the
            // "Compacting…" pulse. Normal live triggering happens after each reply.
            if (historyMode == HistoryMode.ResumeFromCompact && compactRoutine == null && ContextFull())
            {
                compactRoutine = StartCoroutine(CompactConversationRoutine());
                yield return ShowCompactingUntilDone(w);
            }

            if (epoch != dialogueEpoch) yield break;   // closed during the on-open compaction
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
            if (compactRoutine != null || interruptPending)   // compacting owns the model / interrupt already queued
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
            // compaction/interrupt/reply would double-forward the model (audit #3)
            if (compactRoutine != null || interruptPending || dialogueCoroutine != null || VoicesAudible())
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
                yield break;          // model stuck (fallback close handles it) or dialogue closed meanwhile
            // a reply cut before it SPOKE anything leaves its dots/Thinking bubble behind
            if (wasSynced && string.IsNullOrEmpty(spokenShown)) w.PopLastMessage();
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

        bool SyncedReveal => speakReplies && (kkVoice != null || pkVoice != null);

        // ---- thinking-dots placeholder + <think> stream filtering --------------------------
        Coroutine dotsJob;

        void StartThinkingDots(INPCChatWindow w)
        {
            StopThinkingDots();
            dotsJob = StartCoroutine(ThinkingDots(w));
        }

        void StopThinkingDots()
        {
            if (dotsJob != null) { StopCoroutine(dotsJob); dotsJob = null; }
        }

        IEnumerator ThinkingDots(INPCChatWindow w)
        {
            string[] frames = { "..", "...", "." };
            // with thinking enabled the model REALLY reasons behind these dots (until the final
            // </think>), so say so; plain models just get the typing pulse
            string label = allowThinking ? "Thinking" : "";
            int i = 0;
            // self-terminates when the reply ends or the dialogue closes
            while (state == NPCState.TalkingInInteraction)
            {
                yield return new WaitForSecondsRealtime(0.4f);
                if (state != NPCState.TalkingInInteraction) break;
                w.PopLastMessage();
                w.AddMessage(NpcName, StatusStyled(label + frames[i++ % 3]));
            }
            dotsJob = null;
        }

        static string ThinkStyled(string think)
            => $"<i><color=#9A9A9AB0>{think.Trim()}</color></i>\n";

        // status pulses (Thinking… / Compacting… / typing dots) are meta-text, not dialogue —
        // render them italic + slightly dimmed so they read as a different breed of text
        static string StatusStyled(string status)
            => $"<i><color=#CFCFCFC8>{status}</color></i>";

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
                    string tag = inThink ? "</think>" : "<think>";
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

        // Word-by-word reveal: each clause event carries its spoken DURATION, and a single
        // pacing coroutine drips the clause's words into the bubble across ~that window
        // (char-weighted per word, finishing slightly early) — the text "types itself" in step
        // with the voice instead of whole sentences popping in.
        readonly Queue<(string clause, float dur)> revealQueue = new Queue<(string, float)>();
        Coroutine revealJob;

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
                string[] words = clause.Split(' ');
                int chars = clause.Length;
                for (int i = 0; i < words.Length; i++)
                {
                    if (!revealActive || state == NPCState.Idle) break;
                    spokenShown = string.IsNullOrEmpty(spokenShown) ? words[i] : spokenShown + " " + words[i];
                    w.PopLastMessage();
                    w.AddMessage(NpcName, spokenShown);
                    float share = chars > 0 ? (words[i].Length + 1f) / chars : 0f;
                    yield return new WaitForSecondsRealtime(Mathf.Max(0.02f, dur * 0.98f * share));
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
                w.AddMessage(NpcName, pendingFullReply);
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
                w.AddMessage(NpcName, full);
                spokenShown = full;
            }
            revealActive = false;
        }

        private IEnumerator Talk(string question)
        {
            state = NPCState.TalkingInInteraction;
            int epoch = dialogueEpoch;
            replyCanceled = false;
            var w = Window;
            // Send stays interactable: sending mid-reply cancels this reply at a token boundary
            // and asks anew (InterruptThenAsk) — the state machine gates it, not the button.

            Turn turn = null;
            if (historyMode != HistoryMode.ResetEveryTime)
            {
                // recorded up-front so an Escape mid-reply still keeps the (partial) exchange
                turn = new Turn { user = question, npc = "" };
                transcript.Add(turn);
                activeTurn = turn;
            }

            StringBuilder response = new StringBuilder();
            activeResponse = response;
            bool synced = SyncedReveal;
            if (synced) { spokenShown = null; pendingFullReply = null; revealActive = true; StopRevealJob(); }
            bool showThink = w.ShowThinkingTokens;
            string visibleFull = "", thinkFull = "";
            int voicedLen = 0;          // visible chars already handed to the voice
            bool contentShown = false;  // the animated dots own the bubble until real content

            // thinking placeholder: ". / .. / ..." pulses until the first real content lands
            w.AddMessage(NpcName, StatusStyled("."));
            StartThinkingDots(w);

            // A background conversation-KV save still reading this model's GPU state holds the Busy
            // guard (SaveConversationKV is Guarded) — most often the snapshot a just-finished
            // compaction kicked. Driving Chat now would be REFUSED by the guard and the turn lost
            // (audit #13). The gate always clears (its coroutine's finally drops it), so wait it out
            // behind the "Thinking…" dots; a close mid-wait bails via the epoch. This defers the
            // send instead of dropping it.
            while (KvSaveInFlightFor(llm) && epoch == dialogueEpoch) yield return null;
            if (epoch != dialogueEpoch) { StopThinkingDots(); yield break; }

            // -1 inspector values fall back to the selected model's recommended Config preset
            yield return llm.Chat(question, max_new_tokens: maxNewTokens, temperature: temperature,
                top_k: topK >= 0 ? topK : llm.Config.DefaultTopK,
                top_p: topP >= 0f ? topP : llm.Config.DefaultTopP,
                min_p: minP >= 0f ? minP : llm.Config.DefaultMinP,
                presence_penalty: presencePenalty >= 0f ? presencePenalty : llm.Config.DefaultPresencePenalty,
                repetition_penalty: repetitionPenalty >= 0f ? repetitionPenalty : llm.Config.DefaultRepetitionPenalty,
                enable_thinking: allowThinking,
                onTokenGenerated: (token) =>
                {
                    // emoji/symbols the UI font can't render (squares) also drive the TTS into
                    // garbage sounds — strip them HERE, before anything consumes the token
                    // (window, voices, transcript all flow from `response`)
                    token = StripUnrenderable(token);
                    if (token.Length == 0) return;
                    response.Append(token);
                    SplitThink(response.ToString(), out visibleFull, out thinkFull);
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
                            w.AddMessage(NpcName, ThinkStyled(thinkFull));
                        }
                        return;
                    }
                    string display = showThink && thinkFull.Length > 0
                        ? ThinkStyled(thinkFull) + visibleFull
                        : visibleFull;
                    if (display.Length == 0) return;   // still inside <think> — dots keep pulsing
                    StopThinkingDots();
                    w.PopLastMessage();
                    w.AddMessage(NpcName, display);
                    contentShown = true;
                });
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
                w.AddMessage(NpcName, finalVisible.Length > 0 ? finalVisible : "...");
            }

            if (turn != null) turn.npc = finalVisible;
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
            if (stillOpen && historyMode == HistoryMode.ResumeFromCompact && compactRoutine == null && ContextFull())
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

            state = NPCState.Idle;
            // reveal machinery dies WITH the dialogue — a leaked revealActive/pendingFullReply
            // otherwise resurrects the previous reply's text at the NEXT dialogue's first send
            // (PrepareForNextReply settles stale state), even on ResetEveryTime
            StopThinkingDots();
            StopRevealJob();
            revealActive = false;
            pendingFullReply = null;
            spokenShown = null;
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
                && transcript.Count > 0;
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
                yield return saving.SaveConversationKV(ConversationKvKey(), SerializeTranscript(), system_prompt);
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
            // coroutine and tears the KV (audit #8)
            while (compactRoutine != null || KvSaveInFlightFor(llm) || dialogueCoroutine != null) yield return null;
            if (state == NPCState.Idle && !(usePrefetchZone && inPrefetchZone))
                ReleaseLlm(collectGarbage: true);
        }

        // Coroutines die with the component; never leave the save gate latched for a later
        // re-enable — OpenConversation spins on it. Only OUR gate entries are dropped: a
        // sibling's in-flight save (even on the same shared instance) stays latched.
        protected virtual void OnDisable()
        {
            foreach (var m in new List<LLM>(kvSavesInFlight.Keys))
                if (kvSavesInFlight[m] == this) kvSavesInFlight.Remove(m);
            compactRoutine = null;                     // its coroutine died with the component
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
            var sb = new StringBuilder(system_prompt);
            if (!string.IsNullOrEmpty(compactSummary))
                sb.Append("\n\nHISTORY:\n").Append(compactSummary);
            if (transcript.Count > 0)
                sb.Append("\n\n").Append(BuildRecentTurnsBlock(transcript.Count));
            return sb.ToString();
        }

        // The last <paramref name="lastN"/> turns formatted as a resume block (the full resume
        // prompt above and the post-summary context of a compaction share this shape).
        private string BuildRecentTurnsBlock(int lastN)
        {
            var sb = new StringBuilder();
            sb.Append("[The conversation below already happened between you and the player. ")
              .Append("Resume it naturally and stay consistent with everything said.]");
            for (int i = Mathf.Max(0, transcript.Count - lastN); i < transcript.Count; i++)
            {
                var t = transcript[i];
                sb.Append("\nPlayer: ").Append(t.user);
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

        /// <summary>Wipe this NPC's conversation back to a blank slate (right-click the component →
        /// "Reset Conversation"; also public so an in-window button can call it). Clears the
        /// transcript, the compact summary and the live/disk conversation KV, so the next open
        /// starts a fresh InitializeChat(system_prompt). Handy for ResumeFromCompact NPCs when you
        /// want to drop the accumulated (compacted) history without waiting for a natural reset.</summary>
        [ContextMenu("Reset Conversation")]
        public void ResetConversation()
        {
            transcript.Clear();
            compactSummary = null;
            chatLive = false;                       // forces a fresh InitializeChat on next open
            if (llm != null)
            {
                if (LLMPool.OwnsConversation(llm, this))
                    LLMPool.ClaimConversation(llm, null);   // drop our claim on the shared KV
                if (cacheKVCache) llm.DeleteConversationKV(ConversationKvKey());   // remove the disk snapshot
            }
            var w = Window;
            if (w != null && state != NPCState.Idle)
            {
                w.Clear();
                w.SetInfoText("— conversation reset —");
                if (w.SendButton != null) w.SendButton.interactable = true;
            }
            ConsoleMessage.Info($"[Reset] {NpcName}: conversation wiped (manual reset).");
        }

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

        /// <summary>The EXACT request the model is asked when compacting (ResumeFromCompact hitting
        /// Max Context Length). It is one bare USER turn, continuing the tracked conversation —
        /// greedy (temperature 0), capped at 256 tokens. The model's reply IS the compact, which is
        /// then re-seeded as the KV prefix:
        /// <code>[this NPC's system_prompt]\n\nHISTORY:\n[the model's reply]</code>
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
        //      compact of the whole history, then the chat recomputes as [system + HISTORY:
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
                var compact = llm.Compact(system_prompt, s => summary = s);
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
            transcript.Clear();   // the HISTORY block stands in for every turn so far
            LLMPool.ClaimConversation(llm, this);   // the compacted prefix carries OUR conversation
            ConsoleMessage.Info($"[Compact] {NpcName}: compaction done — history → " +
                                $"{summary.Length}-char HISTORY block, KV recomputed");
            // the compact text itself, so its quality can be inspected in the console
            ConsoleMessage.Info($"[Compact] {NpcName}: <i>{summary.Trim()}</i>");
            if (cacheKVCache && chatLive && !KvSaveInFlightFor(llm))
                StartCoroutine(SaveConversationKvRoutine());
        }

        private void RepopulateWindow()
        {
            var w = Window;
            // the compacted past is NOT rendered (user spec): a reopen after compaction starts
            // visually empty — the compact lives only in the system prompt's HISTORY block
            foreach (var t in transcript)
            {
                w.AddMessage("You", t.user);
                if (!string.IsNullOrEmpty(t.npc))
                    w.AddMessage(NpcName, t.npc);
            }
        }

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
