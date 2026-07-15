using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        using Cfg = PocketTTSConfig;

        // The Unity-facing pocket-tts voice of an NPC — same surface as KokoroVoice/CosyVoiceVoice
        // (attach next to an AudioSource, call Say / FeedText+FlushText). pocket-tts is
        // autoregressive, so INSIDE an utterance the pipeline streams per FRAME: KV-prefill the
        // prompt, then each frame produces one latent, and every StreamChunkFrames the accumulated
        // latents are Mimi-decoded and only the NEW tail samples are pushed into a lock-protected
        // ring buffer drained by the audio thread (identical machinery to CosyVoiceVoice).
        //
        // NOTE (P5): pocket-tts has no C# SentencePiece encoder yet (that lands with P7), so the
        // working entry point is Say(int[] textIds) / FeedTokens. Speak(string) warns until the
        // tokenizer is wired. Voice cloning (audio_prompt) is the P7 headline feature.
        [RequireComponent(typeof(AudioSource))]
        public class PocketTTSVoice : MonoBehaviour
        {
            [Tooltip("Streaming: seconds buffered before playback starts (time-to-first-audio vs underrun " +
                     "safety). 1.0 is the UNIVERSAL standard: safe on low-end GPUs, +0.5s TTFA on high-end. " +
                     "The underrun tuner raises it as needed and PERSISTS the learned value per device.")]
            public float prebufferSeconds = 1f;

            [Tooltip("Streaming: ring buffer capacity in seconds.")]
            public float ringSeconds = 60f;

            [Tooltip("Playback pitch. <1 = deeper & slower.")]
            public float pitch = 1f;

            [Tooltip("Loudness gain multiplied into the synthesized samples (AudioSource.volume tops out at 1 — this can go above it; peaks clamp at full scale).")]
            [Min(0f)] public float volume = 1f;

            [Tooltip("Pause inserted between streamed clauses after a sentence ender . ! ? (seconds). Each clause is synthesized as its own utterance and EOS keeps only ~0.16 s of the model's trailing silence, so without this consecutive sentences butt together.")]
            [Min(0f)] public float sentencePauseSeconds = 0.36f;

            [Tooltip("Pause inserted after a clause cut at a semicolon (seconds).")]
            [Min(0f)] public float semicolonPauseSeconds = 0.2f;

            [Tooltip("Pause inserted after an emergency comma cut — run-on sentences past Emergency Chunk Chars (seconds).")]
            [Min(0f)] public float commaPauseSeconds = 0.15f;

            [Tooltip("Extra model-generated tail on the reply's LAST clause (seconds, in post-EOS frames of ~0.08 s). The default EOS stop keeps only ~0.16 s after the final word — an audible hard cut; this lets the model render the word's natural decay and release.")]
            [Min(0f)] public float replyTailSeconds = 0.32f;

            [Tooltip("Frames of audio produced between streaming decodes (chunk cadence). 8 = 0.64s @ 12.5Hz.")]
            public int streamChunkFrames = 8;

            [Tooltip("Milliseconds of main-thread pump per frame for the TTS pipeline.")]
            public float gpuBudgetMs = 6f;

            [Tooltip("Baked voice folder under the weights dir (voices/<name>/). Ignored if a clonedVoiceClip is set.")]
            public string voiceName = "jean";

            [Tooltip("Optional reference clip to CLONE this NPC's voice from (P8). Cloned + disk-cached " +
                     "on first use, then loaded from cache. Requires encoder weights (import_params.py pocket-tts " +
                     "--include-encoder). Leave null to use the baked voiceName.")]
            public AudioClip clonedVoiceClip;

            [Tooltip("Weights folder. int8 = same speed, half the VRAM and load bytes.")]
            public string weightsPath = Cfg.WEIGHTS_DIR_FP16;

            [Tooltip("Build + start loading the shared TTS in Start(). Off = call PrefetchNow() on approach.")]
            public bool loadOnStart = true;

            public bool IsSpeaking { get; private set; }
            public bool IsReady => tts != null && tts.IsReady;

            /// <summary>True while buffered speech is actually audible (drive talk animations from
            /// this). Includes the in-flight TAIL: after the ring empties, the PCM reader has
            /// already handed ~0.2-0.8 s of samples to Unity's DSP/stream-clip pipeline that are
            /// still playing — the same tail the grace-pause protects (bug A).</summary>
            public bool IsAudioPlaying => streamStarted &&
                (RingCount() > 0 || Time.realtimeSinceStartup - lastNonEmptyRealtime < audioTailSeconds);

            /// <summary>Playback-side accounting (headless assertable): total samples pushed into the
            /// ring vs consumed by the audio thread. Pause only fires once read == pushed + grace.</summary>
            public long SamplesPushed { get { lock (ringLock) return totalWritten; } }
            public long SamplesRead { get { lock (ringLock) return totalRead; } }

            // ---- audio-synced text reveal (mirrors KokoroVoice): fires on the main thread the
            // moment a fed clause's audio actually STARTS playing, ~clauseRevealLead early, with the
            // clause's spoken DURATION so the UI can pace a word-by-word reveal across it.
            public event Action<string, float> OnClauseSpoken;
            [Tooltip("OnClauseSpoken fires this many seconds BEFORE the clause is audible (text may lead the voice slightly).")]
            public float clauseRevealLead = 0.35f;
            sealed class ClauseMark { public string text; public long start; public long end = -1; }
            readonly Queue<ClauseMark> spokenQueue = new Queue<ClauseMark>();
            ClauseMark inflightMark;              // the clause the current streamJob is synthesizing
            long totalWritten, totalRead;         // monotonic sample counters (ringLock-guarded)

            static PocketTTS shared;
            PocketTTS tts;
            AudioSource source;

            [Tooltip("Fed text cuts ONLY at sentence enders (. ! ? ;). A comma may cut too, but only past this many pending characters — a run-on-sentence escape hatch.")]
            public int emergencyChunkChars = 220;

            // clause queue (LLM token deltas): text accumulates in pendingText, cut at sentence
            // enders into whole clauses, each tokenized on device. One utterance in flight (single KV).
            readonly System.Text.StringBuilder pendingText = new System.Text.StringBuilder();
            readonly Queue<(int[] ids, string text)> clauseQueue = new Queue<(int[], string)>();
            IEnumerator streamJob;
            bool feedingText;

            readonly System.Diagnostics.Stopwatch pumpWatch = new System.Diagnostics.Stopwatch();

            // #29 arbiter thresholds + pump budgets are CENTRALIZED in InferencePerf (one
            // documented board of every GPU-tuning knob, with low-end/high-end directions).

            // #29: hardware-neutral tick sizing. PocketTTS.GpuMacsPerTick (the slice budget of one
            // heavy pipeline tick) self-calibrates so a heavy tick costs ~TICK_COST_MIN..MAX ms
            // over the scene's own baseline frame — on ANY GPU. Slow cards converge to finer
            // slices, fast cards to coarser ones; external GPU load shrinks it (and it grows back).
            // calibration band + slice bounds live in InferencePerf (Tts TickCost / MacsTick).
            float emaBaseMs = -1f, emaHeavyMs = -1f;   // EMAs of frame cost without/with a heavy tick
            bool heavyTickLastFrame;                    // set by PumpPipeline when it took a FrameBreak

            void CalibrateTickBudget()
            {
                float ms = Time.unscaledDeltaTime * 1000f;
                if (ms <= 0f || ms > 250f) { heavyTickLastFrame = false; return; }   // hitches/loads: ignore
                if (heavyTickLastFrame)
                {
                    heavyTickLastFrame = false;
                    emaHeavyMs = emaHeavyMs < 0f ? ms : Mathf.Lerp(emaHeavyMs, ms, 0.1f);
                    // THROUGHPUT GUARD: while the ring is behind, bigger slices = more audio per
                    // frame (the tick RATE is capped) — grow fast and NEVER shrink. Smoothness is
                    // a luxury of surplus; shrinking here starved the ring during LLM generation
                    // (concurrent Qwen bursts inflate the measured tick cost) and the resulting
                    // mid-clause underruns were audible artifacts on long replies.
                    if (IsSpeaking && RingCount() < (int)(InferencePerf.TtsRefillFloorSeconds * Cfg.SAMPLE_RATE))
                    {
                        PocketTTS.GpuMacsPerTick = Math.Min(InferencePerf.TtsMacsTickCap, (long)(PocketTTS.GpuMacsPerTick * 1.1f));
                        return;
                    }
                    if (emaBaseMs < 0f) return;
                    float cost = emaHeavyMs - emaBaseMs;
                    if (cost > InferencePerf.TtsTickCostMaxMs)
                        PocketTTS.GpuMacsPerTick = Math.Max(InferencePerf.TtsMacsTickFloor, (long)(PocketTTS.GpuMacsPerTick * 0.9f));
                    else if (cost < InferencePerf.TtsTickCostMinMs)
                        PocketTTS.GpuMacsPerTick = Math.Min(InferencePerf.TtsMacsTickCap, (long)(PocketTTS.GpuMacsPerTick * 1.02f));
                }
                else if (IsSpeaking)   // baseline sampled only near the action (same scene load)
                    emaBaseMs = emaBaseMs < 0f ? ms : Mathf.Lerp(emaBaseMs, ms, 0.1f);
            }

            // ring buffer (audio thread reads, main thread writes)
            float[] ring;
            int ringWrite, ringRead, ringCount;
            readonly object ringLock = new object();
            AudioClip streamClip;
            bool streamStarted;

            // grace-pause (bug A): the PCM reader consumes the ring AHEAD of the audible position
            // (DSP buffer + stream-clip lookahead), so pausing the instant the ring empties cuts the
            // last words. Pause only after read==pushed AND the tail has had time to play out.
            float lastNonEmptyRealtime;           // last main-thread observation of a non-empty ring
            float audioTailSeconds = 0.8f;        // DSP latency + stream-clip lookahead (EnsureStream)
            float pauseGraceSeconds = 1.0f;       // continuous-empty time before Pause (>= tail)

            /// <summary>All PocketTTSVoice instances share one engine (one weight set on GPU).</summary>
            public static PocketTTS SetSharedTTS(PocketTTS instance) => shared = instance;

            void Awake()
            {
                source = GetComponent<AudioSource>();
                // per-device persistence: start from whatever the underrun tuner learned in past
                // sessions (never below the inspector values). Pure audio-side state — zero
                // interaction with prefetch zones / pooling / KV persistence.
                // #32 self-healing: if the LAST session finished CLEAN (no escalation fired at the
                // learned level), walk the learned values back ONE rung before adopting them — a
                // single contended/JIT-heavy session can no longer degrade this device forever,
                // while a truly weak GPU just re-earns its rung with one audible gap. Once per
                // session (several voices Awake in a scene).
                float learnedPre = PlayerPrefs.GetFloat(PREF_PREBUFFER, 0f);
                int learnedChunk = PlayerPrefs.GetInt(PREF_CHUNK, 0);
                if (!prefsWalkedBack)
                {
                    prefsWalkedBack = true;
                    if (PlayerPrefs.GetInt(PREF_CLEAN, 1) == 1 && (learnedPre > 0f || learnedChunk > 0))
                    {
                        if (learnedChunk > streamChunkFrames) learnedChunk -= 4;      // reverse of the +4 escalation
                        else if (learnedPre > prebufferSeconds) learnedPre *= 0.5f;   // reverse of the ×2 escalation
                        PlayerPrefs.SetFloat(PREF_PREBUFFER, learnedPre);
                        PlayerPrefs.SetInt(PREF_CHUNK, learnedChunk);
                    }
                    PlayerPrefs.SetInt(PREF_CLEAN, 1);   // re-armed; any escalation this session clears it
                    PlayerPrefs.Save();
                }
                prebufferSeconds = Mathf.Max(prebufferSeconds, learnedPre);
                streamChunkFrames = Mathf.Max(streamChunkFrames, learnedChunk);
            }

            // v3 (#32): keys are GPU-KEYED — prefs that travel with the user profile (or a GPU
            // swap in the same machine) can no longer leak one device's learned escalation onto
            // another. v2 bump history: values learned on the pre-#30 decoder were overly
            // conservative; dropped and relearned on the optimized kernels.
            static string DeviceKey(string k)
            {
                var g = SystemInfo.graphicsDeviceName;
                var sb = new System.Text.StringBuilder(k.Length + 1 + g.Length);
                sb.Append(k).Append('.');
                foreach (char c in g) sb.Append(char.IsLetterOrDigit(c) ? c : '_');
                return sb.ToString();
            }
            static string PREF_PREBUFFER => DeviceKey("DeepUnity.PocketTTS.PrebufferSeconds.v3");
            static string PREF_CHUNK => DeviceKey("DeepUnity.PocketTTS.StreamChunkFrames.v3");
            static string PREF_CLEAN => DeviceKey("DeepUnity.PocketTTS.CleanSession.v3");
            static bool prefsWalkedBack;   // once per play session

            [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
            static void ResetSessionStatics() => prefsWalkedBack = false;   // editor domain-reload-off replays

            static void SaveTunedDefaults(float prebuffer, int chunkFrames)
            {
                PlayerPrefs.SetFloat(PREF_PREBUFFER, prebuffer);
                PlayerPrefs.SetInt(PREF_CHUNK, chunkFrames);
                PlayerPrefs.SetInt(PREF_CLEAN, 0);   // this session escalated — not clean, no walk-back next boot
                PlayerPrefs.Save();
            }

#if UNITY_EDITOR
            // PocketTTS is NOT a ModelBase subclass (standalone IDisposable — WS-F unification
            // pending), so the ModelBase sweep never sees it: the shared engine must be disposed
            // HERE or its FlowLM/Mimi scratch+KV ComputeBuffers survive the play session ("Leak
            // Detected: Persistent allocates 336", root-caused 2026-07-13). Subscribed via
            // InitializeOnLoadMethod — an Awake-time hook dies with any domain reload, and a
            // MID-PLAY recompile needs beforeAssemblyReload (ExitingPlayMode never fires there).
            [UnityEditor.InitializeOnLoadMethod]
            static void HookEditorTeardown()
            {
                UnityEditor.EditorApplication.playModeStateChanged += s =>
                {
                    if (s == UnityEditor.PlayModeStateChange.ExitingPlayMode) DisposeShared();
                };
                UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += DisposeShared;
            }

            static void DisposeShared()
            {
                shared?.Dispose();
                shared = null;
                s_engineBoundClip = null;
                holders.Clear();
                warmed = false;
            }
#endif
            void Start() { if (loadOnStart) EnsureTts(); }

            // The clip actually bound into the SHARED engine right now. STATIC on purpose: the
            // engine holds ONE voice at a time for all NPCs, so the skip-rebind cache must track
            // the engine, not the component — a per-component cache made NPC A keep "already
            // bound" after NPC B rebound the engine, and A spoke with B's cloned voice.
            static AudioClip s_engineBoundClip;

            /// <summary>Assign this NPC's voice from a reference clip at runtime — cloned + disk-cached
            /// on first use, then loaded from cache. Overrides voiceName. Pass null to revert to baked.</summary>
            public void SetClonedVoice(AudioClip clip) { clonedVoiceClip = clip; }

            // Bind the right voice into the shared engine before a clause. Clone-clip takes priority
            // (cheap: cache hit is a file load, and s_engineBoundClip makes a same-clip re-bind a
            // no-op). Falls back to the baked voiceName otherwise.
            void BindVoice()
            {
                if (clonedVoiceClip != null)
                {
                    if (s_engineBoundClip == clonedVoiceClip) return;   // engine already carries this clip
                    if (tts.CloneVoice(clonedVoiceClip)) s_engineBoundClip = clonedVoiceClip;
                    else { s_engineBoundClip = null; tts.SetVoice(voiceName); }   // encoder missing -> baked fallback
                }
                else
                {
                    s_engineBoundClip = null;
                    tts.SetVoice(voiceName);                     // cheap rebind; unknown names warn + keep current
                }
            }

            void EnsureTts()
            {
                if (tts != null) return;
                shared ??= new PocketTTS(weightsPath);
                tts = shared;
            }

            // ---- residency wrappers (mirror Kokoro/CosyVoice; PocketTTSWeights owns the pump) ----
            // The ENGINE is shared but residency requests are per-NPC, so mirror LLMPool and
            // refcount the holders: with intercalated prefetch zones, walking out of NPC A's zone
            // used to defetch the weights out from under NPC B (whose zone-enter had already
            // fired) — B's pump then waited on !IsReady forever and B never spoke again.
            static readonly HashSet<PocketTTSVoice> holders = new HashSet<PocketTTSVoice>();

            /// <summary>Build the engine and start streaming weights at full speed (load-on-approach trigger).</summary>
            public void PrefetchNow() { EnsureTts(); holders.Add(this); tts.BeginLoad(); }

            /// <summary>Load-on-approach spread over ~targetSeconds (budgeted per frame).</summary>
            public void SlowPrefetchNow(float targetSeconds) { EnsureTts(); holders.Add(this); tts.SlowPrefetch(targetSeconds); }

            /// <summary>Drop THIS voice's residency claim; the weights actually unload (budgeted)
            /// only when the LAST holder lets go. A later prefetch re-streams.</summary>
            public void DefetchNow()
            {
                holders.Remove(this);
                holders.RemoveWhere(h => h == null);   // destroyed components must not pin the weights
                if (holders.Count == 0) tts?.Defetch(slow: true);
            }

            /// <summary>One tiny discarded synthesis once resident — compiles every kernel path so the
            /// first real clause has no shader-compile hitch. Call where the player isn't looking.</summary>
            public void PrewarmKernels()
            {
                if (!warmed && prewarmJob == null) prewarmJob = StartCoroutine(PrewarmRoutine());
            }
            static bool warmed;
            Coroutine prewarmJob;

            IEnumerator PrewarmRoutine()
            {
                EnsureTts();
                while (!tts.IsReady) yield return null;
                if (!warmed)
                {
                    warmed = true;
                    var wallSw = System.Diagnostics.Stopwatch.StartNew();
                    // a tiny real utterance: exercises tokenizer + prefill + KV decode + flow + Mimi
                    // decode + chunk stream, so the first real clause has no shader-compile hitch.
                    var e = tts.SynthesizeStreaming(tts.Tokenize("Hi."), _ => { }, maxFrames: 8);
                    // BUDGETED pump. The synth yields THOUSANDS of fine ticks (MAC-sliced prefill,
                    // AR bookkeeping, readback waits) — one MoveNext per frame crawled for ~15 s,
                    // dropping a ~5 ms dispatch into EVERY frame's GPU queue: it saturated the GPU
                    // for the whole first reply of the session (decode 13 → 0.8-2 tok/s, the
                    // "first message takes 5 s to speak" report). Pump a few ms + 2 heavy ticks
                    // per frame instead — done in well under a second, still off the hot path.
                    var frameSw = System.Diagnostics.Stopwatch.StartNew();
                    int heavy = 0;
                    while (e.MoveNext())
                        if ((ReferenceEquals(e.Current, PocketTTS.FrameBreak) && ++heavy >= 2)
                            || frameSw.Elapsed.TotalMilliseconds > 3.0)
                        {
                            heavy = 0;
                            yield return null;
                            frameSw.Restart();
                        }
                    Debug.Log($"[PocketTTSVoice] voice warmup done in {wallSw.ElapsedMilliseconds} ms " +
                              "(real-path warm; kernels already compiled at frame 0 — see PocketTTS.PrewarmKernels).");
                }
                prewarmJob = null;
            }

            // ---------------- streamed-text interface (LLM token deltas) ------------------------
            // Mirrors KokoroVoice/CosyVoiceVoice: deltas accumulate; whole CLAUSES (sentence-cut)
            // are tokenized on device and queued. Speech starts after the first clause while the
            // rest of the reply is still generating.
            public void FeedText(string delta)
            {
                if (string.IsNullOrEmpty(delta)) return;
                EnsureTts();
                if (!ttfaArmed && !streamStarted && streamJob == null)
                {
                    ttfaArmed = true;
                    ttfaFeed = Time.realtimeSinceStartup;
                    ttfaQueue = ttfaSynth = ttfaRing = -1f;
                }
                feedingText = true;
                pendingText.Append(delta);
                CutCompleteChunks();
            }

            public void FlushText()
            {
                CutCompleteChunks();
                string rest = pendingText.ToString().Trim();
                pendingText.Clear();
                if (rest.Length > 0) EnqueueClause(rest);
                feedingText = false;
            }

            void CutCompleteChunks()
            {
                string s = pendingText.ToString();
                int cut = -1;
                for (int i = 0; i < s.Length; i++)
                {
                    char c = s[i];
                    bool sentenceEnd = c == '.' || c == '!' || c == '?' || c == ';' || c == '\n';
                    bool emergency = c == ',' && i >= emergencyChunkChars;
                    if (sentenceEnd || emergency) cut = i;
                }
                if (cut < 0) return;
                string chunk = s.Substring(0, cut + 1).Trim();
                if (chunk.Length > 1) EnqueueClause(chunk);
                pendingText.Clear();
                pendingText.Append(s.Substring(cut + 1));
            }

            void EnqueueClause(string text)
            {
                int[] ids = tts.Tokenize(text);
                if (ids != null && ids.Length > 0)
                {
                    clauseQueue.Enqueue((ids, text));
                    if (ttfaArmed && ttfaQueue < 0f) ttfaQueue = Time.realtimeSinceStartup;
                }
            }

            // inter-clause pause: armed when a clause finishes while more speech is coming,
            // written into the ring right before the NEXT clause's first sample (so a reply's
            // last clause never gets a silent tail and the clause mark starts at real speech).
            int pendingGapSamples;

            int GapSamplesAfter(string clauseText)
            {
                float t = sentencePauseSeconds;
                if (!string.IsNullOrEmpty(clauseText))
                {
                    char c = clauseText[clauseText.Length - 1];
                    if (c == ',') t = commaPauseSeconds;          // emergency comma cut
                    else if (c == ';') t = semicolonPauseSeconds;
                }
                return t <= 0f ? 0 : Mathf.RoundToInt(t * Cfg.SAMPLE_RATE);
            }

            /// <summary>Queue an utterance from pre-tokenized SentencePiece ids (one clause).
            /// No text is known here, so no OnClauseSpoken fires for it (use FeedText for reveal).</summary>
            public void FeedTokens(int[] textIds)
            {
                if (textIds == null || textIds.Length == 0) return;
                EnsureTts();
                feedingText = true;
                clauseQueue.Enqueue((textIds, null));
            }

            /// <summary>Speak a full pre-tokenized utterance. Interrupts anything in progress.</summary>
            public void Say(int[] textIds)
            {
                EnsureTts();
                StopSpeaking();
                FeedTokens(textIds);
                FlushText();
            }

            /// <summary>Speak text. Tokenizes with the C# SentencePiece encoder (P7) and streams.
            /// Interrupts anything in progress.</summary>
            public void Say(string text)
            {
                if (string.IsNullOrWhiteSpace(text)) return;
                EnsureTts();
                Say(tts.Tokenize(text));
            }

            /// <summary>Hard stop: drop in-flight synthesis, queued clauses, and buffered audio.
            /// Cuts IMMEDIATELY (interrupt semantics) — the end-of-utterance grace only applies to
            /// the natural ring-drain pause in Update.</summary>
            // ---- leave-fade: voice doesn't cut on dialogue close — it fades to silence ---------
            Coroutine fadeJob;
            float fadePrevVolume = 1f;

            /// <summary>Fade the voice smoothly to silence over <paramref name="seconds"/>, then
            /// hard-stop (drops synthesis + queued clauses) and restore the volume for the next
            /// utterance. Used on Leave/close so speech doesn't cut mid-word. A new Say/StopSpeaking
            /// during the fade cancels it (volume restored immediately).</summary>
            public void FadeOutAndStop(float seconds = 1f)
            {
                if (fadeJob != null) return;                      // fade already in progress
                if (!IsSpeaking && !IsAudioPlaying) { StopSpeaking(); return; }   // nothing audible
                fadePrevVolume = source != null ? source.volume : 1f;
                fadeJob = StartCoroutine(FadeOutRoutine(seconds));
            }

            IEnumerator FadeOutRoutine(float seconds)
            {
                for (float t = 0f; t < seconds; t += Time.deltaTime)
                {
                    if (source != null) source.volume = Mathf.Lerp(fadePrevVolume, 0f, t / seconds);
                    yield return null;
                }
                fadeJob = null;   // cleared BEFORE StopSpeaking so its fade-cancel is a no-op
                StopSpeaking();
                if (source != null) source.volume = fadePrevVolume;   // ready for the next reply
            }

            public void StopSpeaking()
            {
                // an interrupt (new Say) during a leave-fade cancels it and restores the volume
                if (fadeJob != null)
                {
                    StopCoroutine(fadeJob);
                    fadeJob = null;
                    if (source != null) source.volume = fadePrevVolume;
                }
                streamJob = null;
                inflightMark = null;
                clauseQueue.Clear();
                pendingGapSamples = 0;
                pendingText.Clear();
                feedingText = false;
                IsSpeaking = false;
                lock (ringLock)
                {
                    ringCount = 0; ringRead = 0; ringWrite = 0;
                    totalRead = totalWritten;   // dropped audio counts as consumed (accounting stays sane)
                    spokenQueue.Clear();
                }
                streamStarted = false;
                ttfaArmed = false;
                lastNonEmptyRealtime = float.NegativeInfinity;   // no phantom tail after a hard cut
                if (source != null && source.isPlaying) source.Pause();
            }

            // ---- budget pump: advance the in-flight clause every frame within gpuBudgetMs -------
            void PumpPipeline()
            {
                if (tts == null || !tts.IsReady || prewarmJob != null) return;
                bool anyWork = streamJob != null || clauseQueue.Count > 0;
                if (!anyWork)
                {
                    if (IsSpeaking && RingCount() == 0 && !feedingText) IsSpeaking = false;
                    return;
                }
                IsSpeaking = true;
                EnsureStream();

                // reverse arbiter (weak GPUs): mark starvation ONLY while the player hears
                // SILENCE that synthesis should be filling (clause start prebuffer, or an
                // underrun re-gate) — not merely a low ring during playback, which on weak GPUs
                // is the normal state and would hold the LLM for the whole spoken reply.
                if (!streamStarted && (streamJob != null || clauseQueue.Count > 0))
                    FramePacing.NoteTtsStarving();

                // #29 cross-engine arbiter: while the LLM is actively decoding, a TTS heavy tick
                // lands in the same frame's GPU queue (the 22-27 ms GEN+SPK+AUD band). TTS has
                // throughput margin, the LLM doesn't — so cede LLM frames whenever the ring can
                // afford it: fully above InferencePerf.TtsCedeHeadroomSeconds seconds buffered, alternate frames
                // above the InferencePerf.TtsRefillFloorSeconds. Below the floor (clause start / near-starvation)
                // pump every frame — an occasional shared frame beats an audible gap.
                if (FramePacing.LlmBusy)
                {
                    int headroom = RingCount();
                    if (headroom >= (int)(InferencePerf.TtsCedeHeadroomSeconds * Cfg.SAMPLE_RATE))
                    { FramePacing.TtsDeferrals++; return; }
                    if (headroom >= (int)(InferencePerf.TtsRefillFloorSeconds * Cfg.SAMPLE_RATE) && (Time.frameCount & 1) == 1)
                    { FramePacing.TtsDeferrals++; return; }
                }

                pumpWatch.Restart();
                // #29 it.3: pipeline stages now yield FINE ticks (≲900 MMAC ≈ 4-6 ms GPU each).
                // Steady state takes ONE per frame + a short readback-spin window (smoothness &
                // CPU thrift); with the ring near-dry (clause start / behind playback) take two
                // and spin readbacks for the whole budget — ~10 ms of GPU and a busy-waiting CPU
                // beat an audible gap. (it.3 lesson: a fixed 2 ms spin everywhere starved the
                // ring on long replies and the resulting always-low-ring emergency bursts were
                // WORSE spikes than the waste it saved.)
                bool lowRing = RingCount() < (int)(InferencePerf.TtsRefillFloorSeconds * Cfg.SAMPLE_RATE);
                // silent refill (prebuffer / underrun re-gate): nothing is audible, so frame
                // smoothness buys nothing — push harder to end the gap sooner. #32: HOW MUCH
                // harder is MEASURED, not a constant — a strong GPU refills multiples of
                // real-time on one gentle tick (the fixed 4-tick turbo was the 17-21 ms GEN+SPK
                // band on the 4060), a weak one climbs to the InferencePerf cap.
                bool silentRefill = !streamStarted;
                if (silentRefill)
                {
                    float now = Time.realtimeSinceStartup;
                    int rc = RingCount();
                    if (lastSilentRingCount >= 0 && now > lastSilentTime + 1e-4f)
                    {
                        float rate = (rc - lastSilentRingCount) / (float)Cfg.SAMPLE_RATE / (now - lastSilentTime);
                        if (rate > 0f)
                            refillRateEma = refillRateEma <= 0f ? rate : Mathf.Lerp(refillRateEma, rate, 0.2f);
                        if (refillRateEma > 3f && silentTicksAdaptive > 1)
                            silentTicksAdaptive--;                       // filling ≥3× real-time: be gentle
                        else if (refillRateEma > 0f && refillRateEma < 1.5f
                                 && silentTicksAdaptive < InferencePerf.TtsSilentRefillHeavyTicks)
                            silentTicksAdaptive++;                       // barely real-time: push harder
                    }
                    lastSilentRingCount = rc; lastSilentTime = now;
                }
                else lastSilentRingCount = -1;
                int heavyTicks = 0;
                int maxHeavyTicks = silentRefill ? silentTicksAdaptive : lowRing ? 2 : 1;
                bool pushHard = (lowRing && streamStarted) || (silentRefill && silentTicksAdaptive > 1);
                double frameBudgetMs = (silentRefill && silentTicksAdaptive > 1)
                    ? gpuBudgetMs * InferencePerf.TtsSilentRefillBudgetScale : gpuBudgetMs;
                double waitSpinMs = pushHard ? frameBudgetMs : InferencePerf.TtsGpuWaitSpinMs;
                while (pumpWatch.Elapsed.TotalMilliseconds < frameBudgetMs)
                {
                    if (streamJob == null && clauseQueue.Count > 0)
                    {
                        BindVoice();               // clone-clip (cached) or baked voiceName — cheap rebind
                        tts.StreamChunkFrames = Mathf.Max(1, streamChunkFrames);
                        var (ids, text) = clauseQueue.Dequeue();
                        if (pendingGapSamples > 0)   // pause between clauses, before this clause's mark
                        {
                            PushSamples(new float[pendingGapSamples]);
                            pendingGapSamples = 0;
                        }
                        // clause mark: first sample of this clause lands at totalWritten (single
                        // synthesis in flight) -> OnClauseSpoken fires when playback reaches it
                        inflightMark = new ClauseMark { text = text };
                        lock (ringLock) { inflightMark.start = totalWritten; spokenQueue.Enqueue(inflightMark); }
                        // reply's LAST clause: extra post-EOS frames so the final word decays
                        // naturally (model-rendered) instead of cutting ~0.16 s after it.
                        bool lastClause = !feedingText && clauseQueue.Count == 0;
                        int tailFrames = 2 + (lastClause ? Mathf.Max(0, Mathf.RoundToInt(replyTailSeconds * Cfg.FRAME_RATE)) : 0);
                        streamJob = tts.SynthesizeStreaming(ids, PushSamples, framesAfterEos: tailFrames);
                        if (ttfaArmed && ttfaSynth < 0f) ttfaSynth = Time.realtimeSinceStartup;
                    }
                    if (streamJob == null) break;
                    if (!streamJob.MoveNext())
                    {
                        streamJob = null;
                        string doneClause = null;
                        if (inflightMark != null)   // exact spoken duration now known
                        {
                            lock (ringLock) inflightMark.end = totalWritten;
                            doneClause = inflightMark.text;
                            inflightMark = null;
                        }
                        if (feedingText || clauseQueue.Count > 0)   // more speech follows this clause
                            pendingGapSamples = GapSamplesAfter(doneClause);
                        // #29: end the frame — the next clause's start (embed gather + prefix build
                        // + first prefill tick) must not chain onto this clause's final flush frame.
                        break;
                    }
                    // #29: FrameBreak = that tick just ISSUED a GPU-heavy slice (prefill chunk /
                    // Mimi-decode slice). The budget clock only measures CPU issue time (~1 ms buys
                    // ~15 ms of GPU), so re-entering freely would stack the whole burst into this
                    // frame — cap heavy ticks per frame instead. Plain nulls (cheap AR bookkeeping)
                    // keep packing under budget.
                    else if (ReferenceEquals(streamJob.Current, PocketTTS.FrameBreak))
                    { heavyTickLastFrame = true; if (++heavyTicks >= maxHeavyTicks) break; }
                    // #29: GpuWait = a readback is in flight and nothing can be issued. Give it a
                    // spin window to complete mid-frame (shallow queues often do), then cede the
                    // frame. The window is the full budget while the ring is low (throughput
                    // first), 2 ms once it's comfortable (CPU thrift).
                    else if (ReferenceEquals(streamJob.Current, PocketTTS.GpuWait) &&
                             pumpWatch.Elapsed.TotalMilliseconds > waitSpinMs) break;
                }
            }

            // ---------------- streaming ring buffer ----------------------------------------------
            public int BufferedSamples => RingCount();
            int RingCount() { lock (ringLock) return ringCount; }

            void EnsureStream()
            {
                if (streamClip != null) return;
                int sr = Cfg.SAMPLE_RATE;
                ring = new float[Mathf.CeilToInt(ringSeconds * sr)];
                streamClip = AudioClip.Create("PocketTTSStream", sr, 1, sr, true, OnPcmRead);
                source.clip = streamClip;
                source.loop = true;
                // in-flight tail = DSP mix-buffer latency + the stream-clip's PCM-reader lookahead
                // (Unity reads ~0.2-0.8 s ahead of the audible position). The grace pause must wait
                // at least this long after the ring empties or the last words get cut (bug A).
                AudioSettings.GetDSPBufferSize(out int dspLen, out int dspNum);
                float dsp = AudioSettings.outputSampleRate > 0 ? (float)dspLen * dspNum / AudioSettings.outputSampleRate : 0.05f;
                audioTailSeconds = Mathf.Max(0.8f, dsp + 0.6f);
                pauseGraceSeconds = Mathf.Max(1.0f, audioTailSeconds + 0.2f);
            }

            /// <summary>Main-thread producer: SynthesizeStreaming pushes each new sample block here
            /// (null sentinel on completion).</summary>
            public void PushSamples(float[] samples)
            {
                if (samples == null) return;   // stream-complete sentinel
                if (ttfaArmed && ttfaRing < 0f && samples.Length > 0) ttfaRing = Time.realtimeSinceStartup;
                EnsureStream();
                lock (ringLock)
                {
                    for (int i = 0; i < samples.Length; i++)
                    {
                        if (ringCount >= ring.Length) break;   // full: drop tail (ringSeconds exceeded)
                        ring[ringWrite] = volume == 1f ? samples[i] : Mathf.Clamp(samples[i] * volume, -1f, 1f);
                        ringWrite = (ringWrite + 1) % ring.Length;
                        ringCount++;
                        totalWritten++;                        // only STORED samples count (drops excluded)
                    }
                }
            }

            // ---- anti-stutter (weak GPUs): playback outrunning synthesis mid-reply drains the
            // ring and dribbles word...pause...word (GTX 1650: streaming ~real-time, and the #29
            // arbiter also cedes TTS frames to the decoding LLM). When it happens, re-gate
            // playback on the prebuffer (one longer pause, then a full phrase — instead of
            // word-by-word) and GROW the prebuffer so later clauses start with more runway.
            int underruns;
            bool wasStarved;
            // escalation ceilings live in InferencePerf (TtsPrebufferCapSeconds / TtsMaxChunkFrames).
            // #32 measured silent-refill pacing: adaptive tick count (1 = strong GPU, up to the
            // InferencePerf cap on weak ones), driven by the EMA of ring-fill rate in audio
            // seconds per wall second while silent. Any real underrun snaps it back up.
            int silentTicksAdaptive = 2;
            float refillRateEma;
            int lastSilentRingCount = -1;
            float lastSilentTime;
            // [TTFA] first-speech latency breakdown per reply (log-only diagnostics): armed at
            // the reply's first text delta, one console line when playback actually starts.
            bool ttfaArmed;
            float ttfaFeed, ttfaQueue, ttfaSynth, ttfaRing;

            void Update()
            {
                if (source != null && source.pitch != pitch) source.pitch = pitch;
                CalibrateTickBudget();   // evaluates LAST frame's cost before this frame's pump
                PumpPipeline();
                if (streamClip == null) return;

                int buffered = RingCount();
                if (buffered > 0) lastNonEmptyRealtime = Time.realtimeSinceStartup;

                // mid-reply starvation: ring empty while MORE synthesis is coming (distinct from
                // the natural end-of-reply drain, where nothing is in flight).
                bool starving = streamStarted && buffered == 0 &&
                                (streamJob != null || clauseQueue.Count > 0 || feedingText);
                if (starving && !wasStarved)
                {
                    wasStarved = true;
                    // a real audible gap: the gentle measured refill was too optimistic here —
                    // snap the silent-refill turbo back up before touching the sticky knobs
                    silentTicksAdaptive = Mathf.Max(silentTicksAdaptive, InferencePerf.TtsSilentRefillHeavyTicks);
                    if (++underruns >= 2)
                    {
                        if (prebufferSeconds < InferencePerf.TtsPrebufferCapSeconds)
                        {
                            prebufferSeconds = Mathf.Min(InferencePerf.TtsPrebufferCapSeconds, prebufferSeconds * 2f);
                            underruns = 0;
                            SaveTunedDefaults(prebufferSeconds, streamChunkFrames);
                            Debug.Log($"[PocketTTSVoice] ring underruns — prebuffer raised to " +
                                      $"{prebufferSeconds:F1}s (synthesis can't outrun playback on this GPU; persisted).");
                        }
                        else if (streamChunkFrames < InferencePerf.TtsMaxChunkFrames)
                        {
                            // prebuffer maxed and still starving: the remaining tax is the
                            // per-chunk windowed re-decode — bigger chunks amortize it
                            // (takes effect from the next clause).
                            streamChunkFrames = Mathf.Min(InferencePerf.TtsMaxChunkFrames, streamChunkFrames + 4);
                            underruns = 0;
                            SaveTunedDefaults(prebufferSeconds, streamChunkFrames);
                            Debug.Log($"[PocketTTSVoice] still underrunning at max prebuffer — " +
                                      $"streamChunkFrames raised to {streamChunkFrames} " +
                                      $"({streamChunkFrames * 0.08f:F2}s per decode chunk; persisted).");
                        }
                    }
                    streamStarted = false;   // silence; the start branch below re-arms at the prebuffer
                }
                else if (!starving) wasStarved = false;
                // start at the prebuffer threshold — or as soon as the whole reply is synthesized
                // (short replies never reach the threshold; without this they'd sit forever)
                bool synthIdle = streamJob == null && clauseQueue.Count == 0 && !feedingText;
                if (!streamStarted && buffered > 0 &&
                    (buffered >= prebufferSeconds * Cfg.SAMPLE_RATE || synthIdle))
                {
                    streamStarted = true;
                    if (!source.isPlaying) source.Play();
                    if (ttfaArmed)
                    {
                        float now = Time.realtimeSinceStartup;
                        Debug.Log($"[PocketTTSVoice] TTFA {(now - ttfaFeed) * 1000f:F0} ms — " +
                                  $"first-token→clause {(ttfaQueue - ttfaFeed) * 1000f:F0} | " +
                                  $"clause→synth-start {(ttfaSynth - ttfaQueue) * 1000f:F0} | " +
                                  $"synth→first-audio {(ttfaRing - ttfaSynth) * 1000f:F0} | " +
                                  $"buffer-gate {(now - ttfaRing) * 1000f:F0} ms " +
                                  $"(ring {buffered / (float)Cfg.SAMPLE_RATE:F2}s, prebuffer {prebufferSeconds:F2}s, " +
                                  $"chunk {streamChunkFrames}f, silentTicks {silentTicksAdaptive})");
                        ttfaArmed = false;
                    }
                }
                else if (streamStarted && buffered == 0 && !IsSpeaking)
                {
                    // GRACE pause (bug A): the ring is empty but the tail the PCM reader already
                    // handed to Unity's DSP/stream-clip pipeline is STILL PLAYING. Pause only after
                    // the accounting confirms everything pushed was consumed (read == pushed) AND
                    // the tail has had pauseGraceSeconds to play out (OnPcmRead zero-fills, so the
                    // grace itself is silent). StopSpeaking() keeps its immediate hard cut.
                    long rd, wr; lock (ringLock) { rd = totalRead; wr = totalWritten; }
                    if (rd >= wr && Time.realtimeSinceStartup - lastNonEmptyRealtime >= pauseGraceSeconds)
                    {
                        streamStarted = false;
                        source.Pause();
                        // play-mode assert proxy for "the last words were audible": everything
                        // pushed was consumed BEFORE the pause, and the tail had grace to play out.
                        Debug.Log($"[PocketTTSVoice] pause after drain: read {rd} / pushed {wr} " +
                                  $"(+{pauseGraceSeconds:F1}s grace) — tail fully played.");
                    }
                }

                // audio-synced clause reveal (mirrors KokoroVoice.Update): pop every clause whose
                // playback position has been reached, ~clauseRevealLead early.
                long lead = (long)(clauseRevealLead * Cfg.SAMPLE_RATE);
                while (true)
                {
                    string fire = null; float dur = 0f; bool dequeued = false;
                    lock (ringLock)
                    {
                        if (spokenQueue.Count > 0 && streamStarted && totalRead + lead >= spokenQueue.Peek().start)
                        {
                            ClauseMark mk = spokenQueue.Dequeue();
                            dequeued = true;
                            if (mk.text != null)   // token-fed clauses carry no text -> nothing to reveal
                            {
                                fire = mk.text;
                                dur = mk.end > mk.start
                                    ? (mk.end - mk.start) / (float)Cfg.SAMPLE_RATE
                                    : mk.text.Length * 0.055f;   // still synthesizing — chars estimate
                            }
                        }
                    }
                    if (!dequeued) break;
                    if (fire != null) OnClauseSpoken?.Invoke(fire, dur);
                }
            }

            void OnPcmRead(float[] data)   // AUDIO THREAD
            {
                lock (ringLock)
                {
                    for (int i = 0; i < data.Length; i++)
                    {
                        if (!streamStarted || ringCount == 0) { data[i] = 0f; continue; }   // starved: silence
                        data[i] = ring[ringRead];
                        ringRead = (ringRead + 1) % ring.Length;
                        ringCount--;
                        totalRead++;                            // consumed (zero-fill doesn't count)
                    }
                }
            }

            void OnDestroy()
            {
                holders.Remove(this);   // a destroyed voice must not pin the shared weights
                if (streamClip != null) Destroy(streamClip);
            }
        }
    }
}
