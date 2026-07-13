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
            [Tooltip("Reopening resumes the SAME conversation. While the model is resident (player inside the prefetch zone / talk trigger) the live KV is reused as-is (instant); after a release the whole conversation is restored from disk (cacheKVCache) or the transcript is re-prefilled.")]
            ContinueWhereLeftOff,
            [Tooltip("Like ContinueWhereLeftOff, but a LONG conversation is compacted in the background after the chat closes: the model writes itself a briefing and keeps [summary + the last few verbatim turns] instead of the whole history, so reopening stays a short prefix no matter how long you've talked. Triggers past Compact After Tokens, on the resident model, behind gameplay.")]
            ResumeFromCompact,
        }

        [SerializeField, ViewOnly] protected NPCState state = NPCState.Idle;
        [SerializeField] protected string npc_name = "Villager";
        [TextArea(4, 12)]
        [SerializeField] protected string system_prompt =
            "You are a friendly villager. Stay in character at all times. " +
            "Keep your replies to one to three short sentences.";
        [Tooltip("Flavor line shown in the chat window while the model loads (per NPC).")]
        [SerializeField] protected string approach_text = "They wait for you to speak...";

        [Header("Conversation")]
        [Tooltip("LlmOnly = text-only replies (talk animation follows the writing; voice fields hidden below). LlmPlusTts = replies are spoken: the talk animation follows the AUDIO, and the next sentence synthesizes while the current one plays.")]
        [SerializeField] protected ConversationMode conversationMode = ConversationMode.LlmOnly;
        protected bool speakReplies => conversationMode == ConversationMode.LlmPlusTts;
        [Tooltip("ResetEveryTime = the conversation is wiped the moment the chat closes (same session). ContinueWhereLeftOff = fully persistent, chat-to-chat AND session-to-session: live KV reused while resident, conversation KV restored from disk (or transcript re-prefilled) after a release. ResumeFromCompact = ContinueWhereLeftOff + background compaction: long histories collapse to [summary + recent turns] after the chat closes (see Compact After Tokens).")]
        [SerializeField] protected HistoryMode historyMode = HistoryMode.ResetEveryTime;
        [Tooltip("Persist this NPC's KV cache to disk (persistentDataPath/DeepUnity): the system-prompt state in EVERY mode, plus — in the continue modes — the WHOLE conversation on a clean close (KV + sampler state + transcript), so reopening after the model was released (or the scene reloaded) restores the chat from disk instead of re-prefilling. Qwen3.5 only for now; Gemma3 NPCs fall back to the re-prefill path.")]
        [SerializeField] protected bool cacheKVCache = true;
        [Tooltip("ResumeFromCompact only: estimated conversation size in tokens (~4 chars each) above which the history is compacted in the background after the chat closes — the model writes itself a briefing and the conversation re-seeds as [summary + last turns] on the resident model, behind gameplay.")]
        [Min(256)] [SerializeField] protected int compactAfterTokens = 1536;

        [Header("Text (LLM)")]
        [Tooltip("Which local LLM voices this NPC — the dropdown lists every model registered in LLMRegistry, so a freshly ported LLM appears here automatically. Sampling fields at -1 fall back to this model's Config presets.")]
        [SerializeField] protected string model = "Qwen3.5-0.8B";
        [Tooltip("Weight format. INT8 is ~lossless at half the VRAM — the recommended default. INT4 is lossy on models this small (Gemma int4 collapses outright).")]
        [SerializeField] protected LLMQuant quantization = LLMQuant.INT8;
        [Tooltip("Let a thinking-capable model (Qwen3.5) reason in <think> before answering. The reasoning is NEVER voiced and never shown as reply text (the window's ShowThinkingTokens debug toggle can render it dimmed); while the model thinks, the dialog pulses an animated 'Thinking…' placeholder until the final answer starts. Non-thinking models ignore this.")]
        [SerializeField] protected bool allowThinking = false;
        [Tooltip("Prompt-prefill pacing — governs the pause before the FIRST token (your question and the system prompt are forwarded in per-layer slices). Slices issued per frame: higher = faster first token but more GPU per frame during prefill; 1 = fully spread (smoothest, slowest). 6 is invisible at 60 fps on the small NPC models.")]
        [Range(1, 32)] [SerializeField] protected int prefillLayersPerFrame = 6;

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

        [Header("Voice (TTS)")]
        [Tooltip("PocketTTS = Kyutai 100M AR, RTF ~0.15 int8 (speaks in real time DURING generation, voice cloning — DEFAULT); Kokoro = 82M non-AR, RTF ~0.3; Chatterbox = clause-streamed (RTF~1.4); CosyVoice3 = streaming-native. 2D demo NPCs are Kokoro-only and ignore this.")]
        [SerializeField] protected TtsModel ttsModel = TtsModel.PocketTTS;
        [Tooltip("Playback pitch for this NPC. 1 = natural (the voice's own timbre); <1 = deeper/slower.")]
        [SerializeField] protected float voicePitch = 1.0f;
        [Tooltip("BAKED voice shipped inside the selected TTS engine's weights export (voices/<name> dirs for PocketTTS/CosyVoice3, voices/<name>.bin voicepacks for Kokoro) — the inspector dropdown lists what's on disk. Pick 'Clone (reference clip)' on PocketTTS to clone from an AudioClip instead; a non-null clip always overrides this name.")]
        [SerializeField] protected string ttsVoice = "jean";
        [Tooltip("PocketTTS only: reference clip to VOICE-CLONE for this NPC (overrides the baked ttsVoice). First runtime use encodes it once through the Mimi encoder and caches by content hash; press 'Precompute voice-clone cache' below to bake the embedding into the shared Resources/Cache so runtime (editor AND builds) is a pure load — no recompute, ever.")]
        [SerializeField] protected AudioClip clonedVoiceClip;
        [Tooltip("Chatterbox weight format. INT8 = T3 matmuls int8 (~300 MB less, parity-validated); s3gen stays fp16 either way. Kokoro/CosyVoice ignore this.")]
        [SerializeField] protected LLMQuant ttsQuantization = LLMQuant.INT8;

        [Header("Prefetch Zone (A/B test)")]
        [Tooltip("ON: the big sphere (3D) / circle (2D — auto-detected) around the NPC is the model-RESIDENCY zone: entering slow-prefetches the LLM + TTS in the background, both stay on the GPU while the player is inside (closing the chat releases nothing), and leaving unloads both. OFF: the small talk trigger plays that role instead — load on contact, unload when the player walks off it.")]
        [SerializeField] protected bool usePrefetchZone = false;
        [Min(0f)]
        [Tooltip("Radius of the prefetch/residency zone (drawn as a transparent green filled sphere/disc gizmo).")]
        [SerializeField] protected float prefetchRadius = 10f;
        [Min(1f)]
        [Tooltip("BOTH weight streams (LLM + TTS) are spread over ~this many seconds of walking-up time.")]
        [SerializeField] protected float slowPrefetchSeconds = 3f;

        [Tooltip("Screen-space interaction prompt (\"[I] Speak\" / \"Talk — [ E ]\") shown while the player is in the talk trigger.")]
        [SerializeField] protected GameObject interactPrompt;

        // ---------------------------------------------------------------- runtime state
        protected LLM llm;
        protected ChatterboxVoice voice;
        protected CosyVoiceModeling.CosyVoiceVoice cvVoice;
        protected KokoroVoice kkVoice;
        protected PocketTTSModeling.PocketTTSVoice pkVoice;
        protected Coroutine dialogueCoroutine;
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
        // A background conversation-KV save is reading GPU state. STATIC: pooled instances are
        // shared across NPCs, so NPC B must not reset/forward the model while NPC A's save is
        // still reading it — one global gate keeps that ordering.
        private static bool kvSaveInFlight;
        // ResumeFromCompact maintenance state: the briefing standing in for the trimmed-away
        // turns (rides in the transcript JSON on disk), the in-flight compaction coroutine, and —
        // STATIC, same pooled-model reasoning as kvSaveInFlight — which NPC is compacting, so a
        // dialogue opening on the shared instance cancels it before driving the model itself.
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

            if (interactPrompt != null) interactPrompt.SetActive(false);
            Window?.SetTitle(npc_name);

            var playerGO = GameObject.FindWithTag("Player");
            if (playerGO != null) playerZoneT = playerGO.transform;
        }

        protected virtual void Update()
        {
            if (state == NPCState.Idle && PlayerReady && Input.GetKeyDown(InteractKey))
                StartInteraction();

            // Talk-animation watch. LLM+TTS with Kokoro: the gesture follows the AUDIO (ring
            // actually audible — the NPC keeps talking after the window closes too). Text-only:
            // it follows the token stream (state). Chatterbox/CosyVoice have no audio probe wired
            // yet, so (as before) they drive no gesture in TTS mode.
            bool talking = speakReplies
                ? IsVoiceAudible
                : state == NPCState.TalkingInInteraction;
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
                    // The LLM release defers past any in-flight conversation-KV save (its GPU
                    // readbacks need the buffers alive); ContinueWhereLeftOff keeps the
                    // transcript either way — reopen restores from disk or re-prefills.
                    StartCoroutine(ReleaseLlmAfterKvSave());
                }
                inPrefetchZone = inside;
            }

            if (state != NPCState.Idle && Input.GetKeyDown(KeyCode.Escape))
                CloseInteraction();
        }

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
                    kkVoice.voiceName = ttsVoice;
                    kkVoice.loadOnStart = false;   // load-on-approach, see Update / OnPlayerContact
                    kkVoice.OnClauseSpoken -= OnClauseSpokenHandler;   // audio-synced text reveal
                    kkVoice.OnClauseSpoken += OnClauseSpokenHandler;
                    break;
                case TtsModel.CosyVoice3:
                    cvVoice = GetComponent<CosyVoiceModeling.CosyVoiceVoice>();
                    if (cvVoice == null) cvVoice = gameObject.AddComponent<CosyVoiceModeling.CosyVoiceVoice>();
                    cvVoice.pitch = voicePitch;
                    cvVoice.voiceName = ttsVoice;   // unknown names fall back inside CosyVoiceTTS (warned)
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
                    voice.voiceName = ttsVoice;   // applied in ChatterboxVoice.Start (TTS is lazy-built)
                    voice.quantization = ttsQuantization;
                    break;
            }
            var src = GetComponent<AudioSource>();
            if (src != null) ConfigureVoiceAudioSource(src);
        }

        // Acquire is cheap; weights stream to the GPU over the next frames. The instance comes
        // from LLMPool, so NPCs sharing a model (same id/quant/kv) share ONE stream + ONE VRAM
        // copy — walking between two such NPCs never double-loads.
        // Standard KV pairing from the benchmark matrix: fp16 weights → fp16 KV, int8/int4 → int8 KV.
        protected void EnsureLlm()
        {
            if (llm != null) return;
            KVQuant kv = quantization == LLMQuant.FP16 ? KVQuant.FP16 : KVQuant.INT8;
            llm = LLMPool.Acquire(model, quantization, kv);
            llm.DiskKVCache = cacheKVCache;   // per-NPC toggle: system-prompt + conversation KV disk cache
            llm.PrefillLayersPerFrame = prefillLayersPerFrame;   // per-NPC prefill pacing (shared instance)
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
            state = NPCState.PreparingForInteraction;
            if (interactPrompt != null) interactPrompt.SetActive(false);
            OnInteractionStarted();
            dialogueCoroutine = StartCoroutine(OpenConversation());
        }

        /// <summary>
        /// Opens the chat window after the camera settles and brings the conversation up per
        /// <see cref="historyMode"/>:
        ///   ResetEveryTime          — fresh InitializeChat(system_prompt) every time (the system
        ///                             prompt itself still disk-caches inside InitializeChat).
        ///   ContinueWhereLeftOff    — tier (a): the llm instance is alive with a live KV → reuse
        ///                             it untouched (no re-init, no history clear, instant);
        ///                             tier (b): the model was released → with cacheKVCache the
        ///                             WHOLE conversation state (KV + sampler state + transcript)
        ///                             is restored from disk (skips the prefill entirely); on a
        ///                             miss/mismatch it falls back to re-prefilling the recorded
        ///                             conversation (system prompt + all turns) through the
        ///                             normal chunked prefill. Correct, but pays the prefill.
        ///   ResumeFromCompact       — ContinueWhereLeftOff behavior, but the state being resumed
        ///                             may be a background-COMPACTED one: after a long-enough chat
        ///                             closes, CompactConversationRoutine collapses the history to
        ///                             [system + summary briefing + last few verbatim turns], so
        ///                             every tier below stays short (live KV, disk KV and the
        ///                             re-prefill all carry the compacted prefix).
        /// </summary>
        protected IEnumerator OpenConversation()
        {
            yield return new WaitForSeconds(DialogueOpenDelay);

            var w = Window;
            w.Open();
            // several NPCs share the one chat window — stamp THIS NPC's name every interaction
            w.SetTitle(npc_name);
            w.SetInfoText(approach_text);
            // model still loading: Send pulses dots and stays disabled, but the input field is
            // live so the first question can be typed while the weights stream in
            w.SetSendLoading(true);
            w.InputField.ActivateInputField();

            EnsureLlm();
            // an in-flight background compaction on THIS model instance must die before the
            // dialogue drives it (two coroutines forwarding one model corrupt the KV): its Chat is
            // abandoned mid-generation, the pre-compact disk save (written before it started)
            // remains the fallback, and the next clean close simply retries.
            if (compactingNpc != null && compactingNpc.llm == llm)
                compactingNpc.CancelCompaction();
            llm.DiskKVCache = cacheKVCache;   // re-assert (a resume prefill below clears it temporarily)
            llm.PrefillLayersPerFrame = prefillLayersPerFrame;   // re-assert per-NPC pacing on the shared instance

            // a background conversation-KV save still reading this model's GPU state must finish
            // before anything resets/forwards it again (the SSM snapshot would tear mid-read) —
            // and before we try to restore the very file it is writing
            while (kvSaveInFlight) yield return null;

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
                    bool resume = historyMode != HistoryMode.ResetEveryTime && transcript.Count > 0;
                    if (resume) llm.DiskKVCache = false;
                    yield return llm.InitializeChat(system_prompt: resume ? BuildResumePrompt() : system_prompt);
                    llm.DiskKVCache = cacheKVCache;
                }
                chatLive = true;
                LLMPool.ClaimConversation(llm, this);   // the shared KV now carries OUR conversation
            }

            if (historyMode != HistoryMode.ResetEveryTime && transcript.Count > 0)
                RepopulateWindow();   // the window was cleared on close — restore the visible history

            w.SetInfoText("");
            w.SetSendLoading(false);
            state = NPCState.WaitingInInteraction;
            if (!w.InputField.isFocused)
                w.InputField.ActivateInputField();
            dialogueCoroutine = null;
        }

        /// <summary>Called by the Send button / submitting the input field.</summary>
        public void AskNPC()
        {
            var w = Window;
            if (w == null || w.InputField == null || string.IsNullOrWhiteSpace(w.InputField.text)
                || state != NPCState.WaitingInInteraction)
                return;

            string question = w.InputField.text;
            PrepareForNextReply(w);   // settle a still-speaking previous reply BEFORE the user line lands
            w.AddMessage("You", question);
            w.InputField.text = "";

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
            PrepareForNextReply(w);
            dialogueCoroutine = StartCoroutine(Talk(prompt));
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
                w.AddMessage(npc_name, label + frames[i++ % 3]);
            }
            dotsJob = null;
        }

        static string ThinkStyled(string think)
            => $"<i><color=#9A9A9AB0>{think.Trim()}</color></i>\n";

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
                    w.AddMessage(npc_name, spokenShown);
                    float share = chars > 0 ? (words[i].Length + 1f) / chars : 0f;
                    yield return new WaitForSecondsRealtime(Mathf.Max(0.02f, dur * 0.92f * share));
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
            if (kkVoice != null && kkVoice.IsSpeaking) kkVoice.StopSpeaking();
            if (pkVoice != null && pkVoice.IsSpeaking) pkVoice.StopSpeaking();
            if (pendingFullReply != null && spokenShown != pendingFullReply)
            {
                w.PopLastMessage();
                w.AddMessage(npc_name, pendingFullReply);
            }
            revealActive = false;
        }

        IEnumerator FinishSyncedReveal(string full)
        {
            // wait for the voice AND for the word-pacing to drain its queued clauses
            while (revealActive && ((kkVoice != null && kkVoice.IsSpeaking)
                                    || (pkVoice != null && pkVoice.IsSpeaking)
                                    || revealJob != null || revealQueue.Count > 0))
                yield return null;
            var w = Window;
            if (!revealActive || w == null || state == NPCState.Idle) yield break;
            if (spokenShown != full)   // voice done but tail text never got audio — settle it
            {
                w.PopLastMessage();
                w.AddMessage(npc_name, full);
                spokenShown = full;
            }
            revealActive = false;
        }

        private IEnumerator Talk(string question)
        {
            state = NPCState.TalkingInInteraction;
            var w = Window;
            w.SendButton.interactable = false;

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
            w.AddMessage(npc_name, ".");
            StartThinkingDots(w);

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
                    response.Append(token);
                    SplitThink(response.ToString(), out visibleFull, out thinkFull);
                    // reasoning NEVER reaches the TTS — only newly-VISIBLE text is fed
                    if (speakReplies && visibleFull.Length > voicedLen)
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
                            w.AddMessage(npc_name, ThinkStyled(thinkFull));
                        }
                        return;
                    }
                    string display = showThink && thinkFull.Length > 0
                        ? ThinkStyled(thinkFull) + visibleFull
                        : visibleFull;
                    if (display.Length == 0) return;   // still inside <think> — dots keep pulsing
                    StopThinkingDots();
                    w.PopLastMessage();
                    w.AddMessage(npc_name, display);
                    contentShown = true;
                });
            if (speakReplies)
            {
                if (visibleFull.Length > voicedLen) FeedVoiceText(visibleFull.Substring(voicedLen));
                FlushVoiceText();   // speak the trailing clause
            }
            StopThinkingDots();
            // transcripts/window always carry the VISIBLE text (raw kept only if nothing parsed)
            string finalVisible = visibleFull.Length > 0 ? visibleFull
                                : thinkFull.Length > 0 ? "" : response.ToString();
            if (synced)
            {
                pendingFullReply = finalVisible;
                StartCoroutine(FinishSyncedReveal(pendingFullReply));
            }
            else if (!contentShown)   // reply was pure <think> with display off — settle the bubble
            {
                w.PopLastMessage();
                w.AddMessage(npc_name, finalVisible.Length > 0 ? finalVisible : "...");
            }

            if (turn != null) turn.npc = finalVisible;
            activeTurn = null;
            activeResponse = null;

            w.SendButton.interactable = true;
            w.InputField.ActivateInputField();
            state = NPCState.WaitingInInteraction;
            dialogueCoroutine = null;
            OnReplyFinished();
        }

        /// <summary>Closes the dialogue from any state — Escape, the Leave button, or scripted.</summary>
        public void CloseInteraction()
        {
            bool interrupted = dialogueCoroutine != null;
            if (interrupted)
            {
                StopCoroutine(dialogueCoroutine);
                dialogueCoroutine = null;
            }

            state = NPCState.Idle;
            CloseConversation(interrupted);

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

        /// <summary>
        /// Conversation-persistence half of closing, per <see cref="historyMode"/>. GPU residency
        /// is NOT decided here: the prefetch zone (or, without one, the talk trigger) owns it —
        /// closing the chat never releases the model while the player is still inside; walking
        /// out does (Update's zone-exit branch / OnPlayerLeft), except KeepAliveInBackground.
        ///   Interrupt (Escape mid-reply) — the KV can be half-written (a stopped coroutine can
        ///     land between a forward pass's per-layer yields) so it is marked dead
        ///     (chatLive=false): the next open restores from disk or re-inits/re-prefills on the
        ///     SAME resident instance, never paying a weight reload.
        ///   Clean close in the continue modes — with cacheKVCache the WHOLE conversation state
        ///     snapshots to disk in the background (any residency release due later waits for the
        ///     snapshot's GPU readbacks via kvSaveInFlight).
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
            // else: let the NPC finish the sentence while the player walks away

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
            bool saveConversation = cacheKVCache && !interrupted && chatLive && !kvSaveInFlight
                && llm != null && historyMode != HistoryMode.ResetEveryTime
                && transcript.Count > 0;
            if (saveConversation)
                StartCoroutine(SaveConversationKvRoutine());

            // ResumeFromCompact maintenance (task #28): once the conversation outgrows the
            // threshold, compact it now — while the model idles resident — so every future reopen
            // is a short prefix instead of an ever-growing KV/transcript. Runs AFTER the
            // full-state save above, which stays on disk as the fallback if this is canceled.
            if (historyMode == HistoryMode.ResumeFromCompact && !interrupted && chatLive
                && llm != null && compactRoutine == null
                && transcript.Count > RECENT_TURNS_KEPT
                && EstimatedTranscriptTokens() > compactAfterTokens)
                compactRoutine = StartCoroutine(CompactConversationRoutine());
        }

        // Background conversation-KV snapshot. Any residency release that becomes due while it
        // runs (player walks out of the zone/trigger) waits on kvSaveInFlight — the save reads
        // the model's GPU buffers, so releasing mid-save would tear it.
        private IEnumerator SaveConversationKvRoutine()
        {
            kvSaveInFlight = true;
            var saving = llm;
            yield return saving.SaveConversationKV(ConversationKvKey(), SerializeTranscript(), system_prompt);
            kvSaveInFlight = false;
        }

        // Zone-exit release, deferred past any in-flight conversation save (releasing mid-save
        // errors its readbacks and loses the snapshot). Skipped if a dialogue started meanwhile.
        // An in-flight compaction is CANCELED, not awaited — zone exit means unload NOW (user
        // rule), and the pre-compact save already on disk keeps the conversation safe.
        private IEnumerator ReleaseLlmAfterKvSave()
        {
            CancelCompaction();
            while (kvSaveInFlight) yield return null;
            if (state == NPCState.Idle)
                ReleaseLlm(collectGarbage: true);
        }

        // Coroutines die with the component; never leave the (global) save gate latched for a
        // later re-enable — OpenConversation spins on it. Slight over-reach now that the gate is
        // static (disabling any NPC clears a sibling's in-flight flag), but that only happens on
        // scene teardown, where the save is dead anyway.
        protected virtual void OnDisable()
        {
            kvSaveInFlight = false;
            compactRoutine = null;                     // its coroutine died with the component
            if (compactingNpc == this) compactingNpc = null;
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

        /// <summary>Trigger-contact hook for subclasses (called from their OnTriggerEnter/2D).</summary>
        protected void OnPlayerContact()
        {
            if (interactPrompt != null && state == NPCState.Idle)
                interactPrompt.SetActive(true);

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
                interactPrompt.SetActive(false);

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
        // LLM.Compact uses for its summary briefing). After a compaction the transcript holds
        // only the recent verbatim turns — the briefing block stands in for everything older,
        // mirroring exactly what CompactConversationRoutine seeded into the live KV.
        private string BuildResumePrompt()
        {
            var sb = new StringBuilder(system_prompt);
            if (!string.IsNullOrEmpty(compactSummary))
                sb.Append("\n\n[Summary of the conversation so far]\n").Append(compactSummary);
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
                    sb.Append('\n').Append(npc_name).Append(": ").Append(t.npc);
            }
            return sb.ToString();
        }

        // ---------------------------------------------------------------- background compaction

        // How many trailing turns survive a compaction VERBATIM — the summary covers everything
        // older. Immediate context stays word-exact; the long tail lives on as the briefing.
        private const int RECENT_TURNS_KEPT = 2;

        // ~4 chars/token: a model-agnostic estimate of what the conversation costs in context
        // (the briefing itself counts too once one exists — compaction re-triggers when the
        // compacted state has grown long again).
        private int EstimatedTranscriptTokens()
        {
            int chars = string.IsNullOrEmpty(compactSummary) ? 0 : compactSummary.Length;
            foreach (var t in transcript)
                chars += (t.user?.Length ?? 0) + (t.npc?.Length ?? 0);
            return chars / 4;
        }

        // ResumeFromCompact close-time maintenance, on the resident idle model:
        //   1. wait out the full-state snapshot CloseConversation kicked (it reads the same GPU
        //      buffers this will re-write),
        //   2. LLM.Compact — the model writes a greedy self-briefing, then the chat re-seeds as
        //      [system + briefing + last RECENT_TURNS_KEPT verbatim turns] (a short KV prefix),
        //   3. trim the transcript to those turns and remember the briefing,
        //   4. snapshot the compacted state to disk (overwrites the pre-compact save).
        // CancelCompaction kills it whenever a dialogue opens on this model instance or its
        // residency is released — the pre-compact save stays the fallback and the next clean
        // close retries.
        private IEnumerator CompactConversationRoutine()
        {
            compactingNpc = this;
            while (kvSaveInFlight) yield return null;
            if (!chatLive || state != NPCState.Idle || llm == null
                || !LLMPool.OwnsConversation(llm, this))
            {
                compactRoutine = null;
                compactingNpc = null;
                yield break;
            }

            // the re-seeded prefix is one-shot (same rule as the resume prefill) — don't let
            // InitializeChat's system-prompt cache write an orphan file for it
            llm.DiskKVCache = false;
            string summary = null;
            var compact = llm.Compact(system_prompt, s => summary = s,
                                      post_summary_context: BuildRecentTurnsBlock(RECENT_TURNS_KEPT));
            while (compact.MoveNext()) yield return compact.Current;
            llm.DiskKVCache = cacheKVCache;

            compactRoutine = null;
            compactingNpc = null;
            if (string.IsNullOrEmpty(summary))
                yield break;   // no briefing came out — keep the full state, retry at the next close

            compactSummary = summary;
            transcript.RemoveRange(0, transcript.Count - RECENT_TURNS_KEPT);
            LLMPool.ClaimConversation(llm, this);   // the compacted prefix carries OUR conversation
            if (cacheKVCache && chatLive && !kvSaveInFlight)
                StartCoroutine(SaveConversationKvRoutine());
        }

        // Kills an in-flight background compaction NOW. Its abandoned Chat leaves the KV
        // mid-generation → marked dead; the pre-compact disk save is the recovery path.
        private void CancelCompaction()
        {
            if (compactRoutine == null) return;
            StopCoroutine(compactRoutine);
            compactRoutine = null;
            if (compactingNpc == this) compactingNpc = null;
            chatLive = false;                                  // KV abandoned mid-forward
            if (llm != null) llm.DiskKVCache = cacheKVCache;   // the routine dropped it for the re-seed
        }

        private void RepopulateWindow()
        {
            var w = Window;
            foreach (var t in transcript)
            {
                w.AddMessage("You", t.user);
                if (!string.IsNullOrEmpty(t.npc))
                    w.AddMessage(npc_name, t.npc);
            }
        }

        // ---------------------------------------------------------------- conversation KV disk cache

        // Stable per-NPC key for the on-disk conversation snapshot (the LLM adds the
        // model/quant/kv-quant/system-prompt hash to the file name itself).
        private string ConversationKvKey()
        {
            if (string.IsNullOrEmpty(npc_name)) return "npc";
            var sb = new StringBuilder(npc_name.Length);
            foreach (char c in npc_name)
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
