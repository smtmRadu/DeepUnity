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
                     "safety). TIER-DRIVEN — NPCChatBase pushes the Backend Tradeoff row's value in " +
                     "EnsureVoice (1.5s at Very Smooth down to 0.5s at Very Fast), so editing it here only " +
                     "matters on a voice component driven by something other than an NPC. It can stay short " +
                     "because the tier's tick count is what keeps the ring ahead of playback — a prebuffer " +
                     "cannot outrun a synthesis deficit, it only delays first audio (measured: 3472 ms TTFA " +
                     "at 1.0s vs 6203 ms at 3.0s). Nothing escalates it at runtime any more.")]
            public float prebufferSeconds = 1f;

            [Tooltip("Streaming: ring buffer capacity in seconds.")]
            public float ringSeconds = 60f;

            [Tooltip("Playback pitch. <1 = deeper & slower.")]
            public float pitch = 1f;

            [Tooltip("Loudness gain multiplied into the synthesized samples (AudioSource.volume tops out at 1 — this can go above it; peaks clamp at full scale).")]
            [Min(0f)] public float volume = 1f;

            // ONE pause, not three (user 2026-07-26). Cuts only ever happen at an ender, and the model
            // renders every pause INSIDE a clause itself — a comma is prosody, not a seam, so a
            // comma-specific knob was describing something the engine no longer does (the emergency
            // comma cut now needs 1000 pending chars to fire). Grading . ! ? against ; separately was
            // likewise a distinction nobody could hear across a synthesis boundary.
            [Tooltip("Pause inserted between streamed clauses (seconds). Each clause is synthesized as its own utterance and EOS keeps only ~0.16 s of the model's trailing silence, so without this consecutive sentences butt together. Pauses INSIDE a clause are the model's own.")]
            [Min(0f)] [UnityEngine.Serialization.FormerlySerializedAs("sentencePauseSeconds")]
            public float clausePauseSeconds = 0.36f;

            [Tooltip("Extra model-generated tail on the reply's LAST clause (seconds, in post-EOS frames of ~0.08 s). The default EOS stop keeps only ~0.16 s after the final word — an audible hard cut; this lets the model render the word's natural decay and release.")]
            [Min(0f)] public float replyTailSeconds = 0.32f;

            [Tooltip("Sentences per synthesized chunk. Smaller = faster response, lower quality (prosody resets each sentence); larger = higher quality (intonation flows across sentences), slower response.")]
            [Range(1, 3)] public int clausesPerChunk = 1;

            [Tooltip("Frames of audio produced between streaming decodes (chunk cadence). 8 = 0.64s @ 12.5Hz. " +
                     "TIER-DRIVEN — NPCChatBase pushes the Backend Tradeoff row's value in EnsureVoice (16 at " +
                     "Very Smooth and Smooth, 8 from Balanced up); bigger chunks amortize the fixed per-chunk cost at the " +
                     "price of a coarser text-reveal cadence. Nothing escalates it at runtime any more.")]
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

            /// <summary>True from the MOMENT text/clauses are queued — unlike <see cref="IsSpeaking"/>,
            /// which only latches in the next Update() when the pump picks the work up. Callers that
            /// decide "is this voice done?" right after FeedText/FlushText must use THIS, or they race
            /// the pump (with clausesPerChunk>1 a short reply queues everything at flush time and was
            /// judged 'finished' one frame before it ever spoke).</summary>
            public bool HasPendingSpeech => IsSpeaking || feedingText || clauseQueue.Count > 0 || streamJob != null;
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
            // 0, not -0.25 (2026-07-26). The -0.25 existed to cancel the reader's lead by hand, back
            // when the reveal compared against raw totalRead. The comparison now subtracts
            // audioTailSeconds itself, so the same shift applied TWICE and the text landed ~0.8 s LATE.
            // This is a deliberate artistic offset on top of a correct baseline now — leave it at 0
            // unless the reveal genuinely needs to lead or trail the voice.
            [Tooltip("Seconds to shift the clause text reveal relative to the AUDIBLE playback position (the reader's own lead is already compensated). Negative = text trails the voice, positive = text leads it. 0 = in sync.")]
            public float clauseRevealLead = 0f;
            sealed class ClauseMark
            {
                public string text;
                public long start;             // WRITTEN-sample index of the clause's first sample
                public long end = -1;          // WRITTEN-sample index one past its last sample
                public long streamStart = -1;  // STREAM-time position of `start` — stamped on the
                                               // audio thread when that sample is actually handed
                                               // to the PCM reader (zero-fill shifts it correctly)
            }
            readonly Queue<ClauseMark> spokenQueue = new Queue<ClauseMark>();
            // Marks awaiting their streamStart stamp, in enqueue order (audio thread, ringLock).
            readonly Queue<ClauseMark> stampQueue = new Queue<ClauseMark>();
            ClauseMark inflightMark;              // the clause the current streamJob is synthesizing
            long totalWritten, totalRead;         // monotonic sample counters (ringLock-guarded)
            long streamPos;                       // cumulative samples handed to the PCM reader (zero-fill INCLUDED)
            long audiblePlayed;                   // cumulative stream samples the DSP has actually played
            int lastTimeSamples;                  // source.timeSamples at the last Update (wrap tracking)

            static PocketTTS shared;
            PocketTTS tts;
            AudioSource source;

            [Tooltip("Fed text cuts ONLY at sentence enders (. ! ? ;). A comma may cut too, but only past this many pending characters — a run-on-sentence escape hatch.")]
            // 1000, not 220 (user 2026-07-26): this is a LAST-RESORT valve for a sentence that never
            // ends, not a pacing knob. At 220 it fired on ordinary NPC replies and cut mid-sentence at
            // a comma, ignoring clausesPerChunk entirely — the audible symptom was speech stopping in
            // the middle of a sentence instead of after the Nth ender.
            public int emergencyChunkChars = 1000;

            // clause queue (LLM token deltas): text accumulates in pendingText, cut at sentence
            // enders into whole clauses, each tokenized on device. One utterance in flight (single KV).
            readonly System.Text.StringBuilder pendingText = new System.Text.StringBuilder();
            readonly Queue<(int[] ids, string text)> clauseQueue = new Queue<(int[], string)>();
            IEnumerator streamJob;
            bool feedingText;

            readonly System.Diagnostics.Stopwatch pumpWatch = new System.Diagnostics.Stopwatch();

            // Nothing GPU-tuning is authored in this file. It is split by KIND: whatever is a
            // statement about the MACHINE is a Backend Tradeoff row (ticks per frame, prebuffer,
            // chunk frames, cede headroom, tick MACs), and whatever is a rate/shape of the #29
            // arbitration itself stays on InferencePerf's board (cede stride, refill floor, budget
            // scale, readback spin).

            // #29 tick sizing is the DIAL's, not measured (2026-07-27). PocketTTS.GpuMacsPerTick — the
            // slice budget of one heavy pipeline tick — now reads BackendTradeoffTable's row, so slow
            // cards get finer slices and fast ones coarser ones by declaration. What used to be here:
            // CalibrateTickBudget, two frame-cost EMAs (with/without a heavy tick) and a per-frame
            // heavy-tick COUNT, walking GpuMacsPerTick between InferencePerf's 200M-4G bounds toward a
            // 3-7 ms measured cost. Deleted with its knobs. It could not be made to work: slice COUNT
            // is derived from GpuMacsPerTick, so its shrink branch produced MORE frame-bound ticks per
            // clause prefill — a feedback loop into the starvation it existed to prevent — and the
            // 2026-07-26 fix for that (skip multi-tick frames as unpriceable) only stopped it from
            // mispricing the frames it now had to ignore, which on this machine is most of them.

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
            }

            // NOTHING is learned per device here any more (2026-07-27). Deleted: the GPU-keyed
            // PlayerPrefs ("DeepUnity.PocketTTS.PrebufferSeconds/StreamChunkFrames/CleanSession.v3"),
            // their DeviceKey/SaveTunedDefaults helpers, the once-per-session prefsWalkedBack flag
            // with its RuntimeInitializeOnLoadMethod reset, and the #32 self-healing walk-back that
            // ran right here in Awake. The tier already states how weak the machine is, so
            // re-discovering it per device (a) cost one audible gap for every rung climbed, (b) then
            // walked the rung back after a clean session and re-earned it with another gap in the
            // next one, and (c) had — on the reference machine, from sessions nobody could see —
            // persisted a 3.0 s prebuffer that spent ~3.2 s of TTFA on every single reply and that no
            // author ever chose. prebufferSeconds / streamChunkFrames are pushed from the dial by
            // NPCChatBase instead.

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

            /// <summary>Conversation-open boost: finish a still-streaming voice at the tier's full
            /// upload rate. Prefetch policy (2026-07-30): the zone only ever pays walk-up pacing —
            /// the boost belongs to the dialogue opening, nowhere else.</summary>
            public void BoostPrefetchNow() { EnsureTts(); holders.Add(this); tts.BoostPrefetch(); }

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
                // Pre-size every clause-lifetime buffer BEFORE the warmup synth (2026-07-30 spike
                // hunt): the synth's own EnsureKV was the session's cold allocation — a 174 ms
                // frame — and the first REAL clause then regrew it all for its bigger prompt+cap,
                // a 286 ms frame mid-conversation. Spread here, one driver allocation per frame
                // (~6-40 ms each, see PreallocateYielding's cost-honesty note) across the walk-up.
                // NOTE `warmed` is static and PrewarmKernels gates on it: this whole routine runs
                // for the session's FIRST voice only, which is sufficient BECAUSE the bounds below
                // are voice-independent worst cases (clone cap, not the bound prompt).
                var pa = tts.PrewarmAllocationsYielding();
                while (pa.MoveNext()) yield return null;
                // Defetched mid-drain (zone-edge turnaround: the exact frames the weights finish
                // streaming)? Then NOTHING below would warm anything — bail WITHOUT burning the
                // once-per-session `warmed` latch, or no voice ever prewarms again and both stalls
                // return mid-conversation (verifier finding E, 2026-07-30).
                if (!tts.IsReady) { prewarmJob = null; yield break; }
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
                // cut after the Nth sentence ender (clausesPerChunk, see TtsClauseCut): the batched
                // sentences reach the model as ONE utterance, so prosody flows across their
                // boundaries instead of resetting per sentence. Loops: one delta can complete
                // several chunks.
                while (true)
                {
                    string s = pendingText.ToString();
                    int cut = TtsClauseCut.FindCut(s, clausesPerChunk, emergencyChunkChars);
                    if (cut < 0) return;
                    string chunk = s.Substring(0, cut + 1).Trim();
                    if (chunk.Length > 1) EnqueueClause(chunk);
                    pendingText.Clear();
                    pendingText.Append(s.Substring(cut + 1));
                }
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

            int GapSamples() =>
                clausePauseSeconds <= 0f ? 0 : Mathf.RoundToInt(clausePauseSeconds * Cfg.SAMPLE_RATE);

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
            // While a fade runs, the synth pipeline may STILL be writing the dying reply's tail
            // (pocket synthesizes ahead of playback) — those writes must be dropped, or the
            // fade's new-reply detector (totalWritten delta) trips on the OLD reply and the fade
            // aborts, leaving the voice talking until the caller's hard-stop deadline.
            bool fadingOut;

            public void FadeOutAndStop(float seconds = 1f)
            {
                if (fadeJob != null) return;                      // fade already in progress
                if (!IsSpeaking && !IsAudioPlaying) { StopSpeaking(); return; }   // nothing audible
                fadingOut = true;
                fadePrevVolume = source != null ? source.volume : 1f;
                fadeJob = StartCoroutine(FadeOutRoutine(seconds));
            }

            IEnumerator FadeOutRoutine(float seconds)
            {
                long written0; lock (ringLock) written0 = totalWritten;
                // unscaled: fades must finish under pause menus (timeScale 0)
                for (float t = 0f; t < seconds; t += Time.unscaledDeltaTime)
                {
                    long w; lock (ringLock) w = totalWritten;
                    if (w != written0) break;   // a NEW reply started writing — abort, never touch it
                    // exponential (constant-dB, ~-60 dB over the fade): hearing is logarithmic, so
                    // a LINEAR amplitude ramp reads as loud-loud-then-sudden-cut — this one reads
                    // as a perfectly even trail-off
                    if (source != null) source.volume = fadePrevVolume * Mathf.Pow(0.001f, t / seconds);
                    yield return null;
                }
                fadeJob = null;   // cleared BEFORE StopSpeaking so its fade-cancel is a no-op
                long w1; lock (ringLock) w1 = totalWritten;
                if (w1 == written0) StopSpeaking();   // complete the stop only while it is still OUR audio
                fadingOut = false;
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
                fadingOut = false;
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
                    stampQueue.Clear();         // stale marks must not receive a next-reply stamp
                }
                streamStarted = false;
                ttfaArmed = false;
                // A hard cut ends the reply, so the mid-reply resume threshold must not survive it:
                // left set, the NEXT reply — which had nothing wrong with it — opened its gate on
                // regateSeconds instead of the full prebuffer and starved immediately. Same for the
                // silence counters: unreset, an interrupted reply dumped its accumulated silence into
                // the next clean reply's log line, inflating exactly the number decisions are made on.
                regateSeconds = -1f;
                lock (ringLock) { zeroDrySamples = 0; zeroGateSamples = 0; zeroBursts = 0; zeroRunOpen = false; }
                lastNonEmptyRealtime = float.NegativeInfinity;   // no phantom tail after a hard cut
                // A hard cut leaves the stream clip TAINTED: Unity's PCM reader already pulled
                // ~0.2-0.8 s of the OLD speech into its internal streaming buffer, and a later
                // Play would render THAT before asking for new samples ("one second of the
                // previous reply before the new one"). Destroy the clip — the next reply's
                // EnsureStream builds a fresh, silent one. (The natural end-of-reply pause keeps
                // its clip: the reader buffered only the zero-fill there.)
                if (source != null)
                {
                    source.Stop();
                    source.clip = null;
                }
                if (streamClip != null) { Destroy(streamClip); streamClip = null; }
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

                // A clause that has not pushed its first sample is in its prompt prefill and producing
                // NOTHING — see prefillBoost below. This has to be known BEFORE the cede gate: the
                // arbiter reads only the ring level, which looks identical mid-clause and mid-prefill,
                // so above the headroom it stopped the next clause from even STARTING until the ring
                // drained back down. The dead window then ran against a reserve that was already spent.
                // A clause not yet dequeued counts too: the cede gate sits above the dequeue at all.
                bool clausePrefilling = ClausePrefilling;
                // ...and mark ourselves starving through it, so the LLM yields the frames instead of
                // holding them: the mark above only fires once playback has ALREADY gone silent, which
                // is too late for a gap that is still preventable.
                if (clausePrefilling && streamStarted) FramePacing.NoteTtsStarving();

                // Panic band (2026-07-30): the ring is below TtsPanicFloorSeconds while the player
                // is LISTENING and synthesis is in flight. Mid-reply low ring deliberately does not
                // hold the LLM (reverse arbiter above — low is the normal state on weak GPUs), but
                // this band is not "low": it is one playback chunk from an audible hole, and the
                // hole is preventable — hold the LLM (bounded by InferencePerf.LlmHoldMaxFrames)
                // and let the hurry-flush (PocketTTS.StreamHurry) land whatever already exists.
                if (streamStarted &&
                    RingCount() < (int)(InferencePerf.TtsPanicFloorSeconds * Cfg.SAMPLE_RATE))
                    FramePacing.NoteTtsStarving();

                // #29 cross-engine arbiter: while the LLM is actively decoding, a TTS heavy tick
                // lands in the same frame's GPU queue (the 22-27 ms GEN+SPK+AUD band). TTS has
                // throughput margin, the LLM doesn't — so cede LLM frames whenever the ring can
                // afford it: fully above the tier's BackendTradeoffTable.TtsCedeHeadroomSeconds buffered,
                // alternate frames above the tier's TtsRefillFloorSeconds. Below the floor (clause
                // start / near-starvation) pump every frame — an occasional shared frame beats a gap.
                // NEVER cede through a clause's dead window (2026-07-26): banked samples are irrelevant
                // when the thing they have to cover is a fixed 24-60 tick prefill that produces nothing.
                if (FramePacing.LlmBusy && !clausePrefilling)
                {
                    int headroom = RingCount();
                    if (headroom >= (int)(BackendTradeoffTable.TtsCedeHeadroomSeconds * Cfg.SAMPLE_RATE))
                    { FramePacing.TtsDeferrals++; return; }
                    // Middle band (floor..headroom): cede on odd frames only, which is HALF as often
                    // as the band above — and the rate bound in FramePacing.LlmBusy (at most one cede
                    // in TtsCedeFrameStride frames) has already applied by the time we get here, so
                    // this is the second, band-specific halving that yields the documented ≈1/6.
                    // Do NOT "fix" this into a `% TtsCedeFrameStride` test: that stacks the same bound
                    // twice and makes ceding both rarer and irregular (caught 2026-07-28).
                    if (headroom >= (int)(BackendTradeoffTable.TtsRefillFloorSeconds * Cfg.SAMPLE_RATE) &&
                        (Time.frameCount & 1) == 1)
                    { FramePacing.TtsDeferrals++; return; }
                }

                pumpWatch.Restart();
                // #29 it.3: pipeline stages yield FINE ticks (one tier slice of MACs ≈ 4-6 ms GPU on
                // the 1650). How many a frame may issue is the tier's, not a constant and no longer
                // measured (2026-07-27): BackendTradeoffTable's speaking/silent columns, whose whole
                // derivation — 1 tick/frame = 0.35-0.8× real-time, 4 = 1.3-2.5× — is documented there.
                // The readback-spin window still tracks the ring: full budget while pushing hard,
                // otherwise a short spin. (it.3 lesson: a fixed 2 ms spin everywhere starved the ring
                // on long replies and the resulting always-low-ring emergency bursts were WORSE spikes
                // than the waste it saved.)
                bool lowRing = RingCount() < (int)(BackendTradeoffTable.TtsRefillFloorSeconds * Cfg.SAMPLE_RATE);
                // silent refill (prebuffer / underrun re-gate): nothing is audible, so frame
                // smoothness buys nothing — push harder to end the gap sooner. HOW much harder is the
                // tier's silent column, which is always above its speaking one; the refill-rate EMA
                // that used to decide it (refillRateEma / lastSilentRingCount / lastSilentTime driving
                // a silentTicksAdaptive counter) is deleted. It measured the wrong thing anyway:
                // samples land in 0.64 s bursts once per ~24 frames, so it read the BURST rate (~24×
                // real-time) as throughput (~1×) and sat at 1 on every device — every TTFA line ever
                // logged ended `silentTicks 1`, i.e. that turbo never once ran.
                bool silentRefill = !streamStarted;
                // #33 (2026-07-26): a clause that has not pushed its first sample yet is still in its
                // PROMPT PREFILL, and that prefill is FRAME-bound, not GPU-bound — ~24 FrameBreak ticks
                // (4 per transformer layer) of a few ms each, admitted only at the rate the cap below
                // allows. ~0.1 s of GPU becomes 0.6-1.0 s of producing nothing; that is exactly the
                // `synth→first-audio` figure in the TTFA lines. Playback keeps draining at 1× through
                // all of it, so the ring runs out INSIDE the previous clause's tail and the player hears
                // speech stop mid-sentence. The window is per-CLAUSE and nearly independent of clause
                // length (dominated by the 125-frame voice prompt, identical for every clause) — which
                // is why halving clausesPerChunk made it much worse rather than better. So spend frames
                // here, on the same reasoning as silentRefill: a dropped frame beats an audible hole.
                int heavyTicks = 0;
                // ONE test, three states (fix 2026-07-28). The three asks used to be MAXed separately
                // and `lowRing && streamStarted` was not among them — so the single most urgent state
                // in the whole system, "the ring is running dry WHILE the player is listening", was
                // the one state left on the SPEAKING column, i.e. the smallest allowance. That is what
                // every `ring starved mid-reply` warning in the 2026-07-28 log is reporting when it
                // prints `4 ticks/frame speaking`; the tier's silent column (6) sat unused through
                // exactly the frames it exists for. And because Smooth already carries the table's
                // highest TTS ticks (tied with Very Smooth), the warning's own advice — drop a tier —
                // could not have helped: there was no lower rung to take.
                // pushHard is that one test now, and it picks the tick cap as well as the budget.
                bool pushHard = (lowRing && streamStarted) || clausePrefilling || silentRefill;
                int maxHeavyTicks = pushHard
                    ? Math.Max(BackendTradeoffTable.TtsSpeakingTicksPerFrame,
                               BackendTradeoffTable.TtsSilentTicksPerFrame)
                    : BackendTradeoffTable.TtsSpeakingTicksPerFrame;
                maxTicksLastFrame = maxHeavyTicks;
                double frameBudgetMs = (clausePrefilling || silentRefill)
                    ? gpuBudgetMs * InferencePerf.TtsSilentRefillBudgetScale : gpuBudgetMs;
                double waitSpinMs = pushHard ? frameBudgetMs : InferencePerf.TtsGpuWaitSpinMs;
                while (pumpWatch.Elapsed.TotalMilliseconds < frameBudgetMs)
                {
                    if (streamJob == null && clauseQueue.Count > 0)
                    {
                        BindVoice();               // clone-clip (cached) or baked voiceName — cheap rebind
                        tts.StreamChunkFrames = Mathf.Max(1, streamChunkFrames);
                        // emergency-flush hook — see PocketTTS.StreamHurry and
                        // InferencePerf.TtsPanicFloorSeconds: suspend the chunk cadence while the
                        // player is hearing (playback gated) or about to hear (ring in the panic
                        // band) silence. Re-bound at every dequeue: tts is shared across voices,
                        // and a single synthesis is in flight engine-wide.
                        tts.StreamHurry = hurryHook ??= () =>
                            !streamStarted ||
                            RingCount() < (int)(InferencePerf.TtsPanicFloorSeconds * Cfg.SAMPLE_RATE);
                        var (ids, text) = clauseQueue.Dequeue();
                        if (pendingGapSamples > 0)   // pause between clauses, before this clause's mark
                        {
                            PushSamples(new float[pendingGapSamples]);
                            pendingGapSamples = 0;
                        }
                        // clause mark: first sample of this clause lands at totalWritten (single
                        // synthesis in flight) -> OnClauseSpoken fires when playback reaches it
                        inflightMark = new ClauseMark { text = text };
                        lock (ringLock)
                        {
                            inflightMark.start = totalWritten;
                            spokenQueue.Enqueue(inflightMark);
                            stampQueue.Enqueue(inflightMark);   // audio thread stamps streamStart
                        }
                        // reply's LAST clause: extra post-EOS frames so the final word decays
                        // naturally (model-rendered) instead of cutting ~0.16 s after it.
                        bool lastClause = !feedingText && clauseQueue.Count == 0;
                        int tailFrames = 2 + (lastClause ? Mathf.Max(0, Mathf.RoundToInt(replyTailSeconds * Cfg.FRAME_RATE)) : 0);
                        streamJob = tts.SynthesizeStreaming(ids, PushSamples, framesAfterEos: tailFrames);
                        lock (ringLock) clauseStartWritten = totalWritten;   // arms prefillBoost
                        clauseStartRealtime = Time.realtimeSinceStartup;    // bounds it in TIME too
                        if (ttfaArmed && ttfaSynth < 0f) ttfaSynth = Time.realtimeSinceStartup;
                    }
                    if (streamJob == null) break;
                    if (!streamJob.MoveNext())
                    {
                        streamJob = null;
                        if (inflightMark != null)   // exact spoken duration now known
                        {
                            lock (ringLock) inflightMark.end = totalWritten;
                            inflightMark = null;
                        }
                        if (feedingText || clauseQueue.Count > 0)   // more speech follows this clause
                            pendingGapSamples = GapSamples();
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
                    { if (++heavyTicks >= maxHeavyTicks) break; }
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
                // A fresh clip is a fresh STREAM TIMELINE: the measured-reveal counters must
                // restart with it, or a hard cut (which destroys the clip with 0.2-0.8 s pulled
                // but never played) leaves audiblePlayed permanently behind streamPos.
                lock (ringLock) streamPos = 0;
                audiblePlayed = 0;
                lastTimeSamples = 0;
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
                if (fadingOut) return;         // dying reply's tail — dropped, see fadingOut
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
            // playback on a short threshold (one pause, then a full phrase — instead of
            // word-by-word). DETECTION ONLY as of 2026-07-27: the voice no longer modifies itself in
            // response. The escalation ladder that used to double prebufferSeconds and then grow
            // streamChunkFrames (persisting both) is gone — those two are the dial's now, and a
            // deficit in synthesis RATE is not something a bigger buffer can repay anyway. What stays
            // is the warning below, which is the only thing that ever told anyone this happened.
            int underruns;
            bool wasStarved;
            // What the pump was actually ALLOWED to issue on its last run. The starve warning prints
            // this rather than the speaking column: after the 2026-07-28 fix the two differ exactly
            // when it matters, and printing the column instead of the allowance is how the earlier
            // logs hid the fact that the urgent state was running at the smallest setting.
            int maxTicksLastFrame;
            // #33 (2026-07-26): the silence the PLAYER actually heard INSIDE a reply, counted on the
            // AUDIO thread. The `starving` test below runs after PumpPipeline in the same Update, so a
            // dry period that opens and closes within one frame is invisible to it. This counter cannot
            // miss it, and it is what "audio stopped mid-sentence" means in numbers.
            long zeroDrySamples;    // ring ran dry while playback was live AND more speech was coming
            long zeroGateSamples;   // playback re-gated mid-reply and is waiting on the ring again
            int zeroBursts;         // contiguous runs of either
            bool zeroRunOpen;
            // Set by Update, read on the audio thread: is any more audio expected at all? Without it the
            // counter charged the END-OF-REPLY GRACE as a dropout — the ring is legitimately empty for
            // pauseGraceSeconds while the DSP tail plays out, so every clean reply reported ~1.0 s of
            // "dry" (measured 0.96-1.12 s across five replies, 2026-07-26) and buried the real events.
            bool moreAudioExpected;
            // Mid-reply resume threshold. A starve must NOT re-arm on the full prebuffer: that turned a
            // ~1 s dry spell into a 2-6 s hole (log, 2026-07-26: 0.80 / 2.00 / 6.00 s "re-gated"), which
            // is what the player heard as speech freezing mid-clause. -1 = use prebufferSeconds.
            float regateSeconds = -1f;
            // totalWritten as of the current streamJob's creation. While it has not moved, this clause
            // is still in its prompt prefill and producing NOTHING — see prefillBoost in the pump.
            long clauseStartWritten;
            float clauseStartRealtime;
            // Cached delegate for PocketTTS.StreamHurry (re-bound at every clause dequeue —
            // allocation-free after the first). See InferencePerf.TtsPanicFloorSeconds.
            Func<bool> hurryHook;
            /// <summary>Hard ceiling on how long a clause may be called "still prefilling", whatever the
            /// sample counter says. The measured dead window is 258-604 ms (`synth→first-audio`), so 1.0 s
            /// is generous cover with no room to latch. REQUIRED because the sample test alone is
            /// unbounded for a SHORT clause: one that produces less than a chunk of audio never crosses
            /// the threshold, so the flag stayed true for its whole duration and the pump then (a) never
            /// ceded a frame and (b) called NoteTtsStarving every frame — measured in the 14:22 run as
            /// `held 109 frames`, decode down to 9.4 tok/s from 12-13, and 1.00 s of dry ring, because
            /// starving the LLM starves the TEXT the voice is waiting for. Bounding it in time keeps the
            /// 2026-07-28 clause-boundary fix and removes the latch.</summary>
            const float ClausePrefillGuardSeconds = 1.0f;

            /// <summary>True while the current — or the next, not yet dequeued — clause is inside its
            /// prompt prefill and therefore producing no samples at all. This is a BOUNDED window
            /// (~410 ms measured: a ~150-row prefill that yields ~24 frame-breaks), not a synthesis
            /// deficit, and samples are guaranteed on the far side of it. Two callers depend on telling
            /// those two situations apart: the pump spends extra frames here rather than ceding them,
            /// and the starve handler must NOT re-gate playback here — waiting to re-bank a second of
            /// audio turned a 0.1 s dry ring into a 1.6 s hole (2026-07-27).</summary>
            /// <para>The test is "this clause has not delivered a STEADY-STATE chunk yet", not "it has
            /// written nothing yet" (fix 2026-07-28). It used to be <c>totalWritten ==
            /// clauseStartWritten</c>, which goes false at the clause's FIRST flush — and
            /// <c>PocketTTS.StreamFirstChunkFrames</c> is 2 latents, i.e. 0.16 s. The next delivery is a
            /// whole chunk (1.28 s at chunk 16) about a second later, so the guard covered 0.16 s of a
            /// ~1.3 s supply gap and the ring hit zero AFTER it had released: the starve was then
            /// misfiled as "synthesis lost the race", playback re-gated, and the 2026-07-27 fix was
            /// silently reopened by the 12→16 chunk change. Measuring the gap in samples ties the guard
            /// to the actual delivery schedule instead of to the first byte.</para></summary>
            bool ClausePrefilling
            {
                get
                {
                    // AND a time bound (fix 2026-07-28, same day): the sample test is unbounded for a
                    // clause shorter than one chunk — see ClausePrefillGuardSeconds for the measured
                    // damage that caused.
                    bool withinWindow =
                        Time.realtimeSinceStartup - clauseStartRealtime < ClausePrefillGuardSeconds;
                    lock (ringLock)
                        return (streamJob != null && withinWindow &&
                                totalWritten - clauseStartWritten
                                    < (long)Mathf.Max(1, streamChunkFrames) * Cfg.SAMPLES_PER_LATENT)
                               || (streamJob == null && clauseQueue.Count > 0);
                }
            }
            // [TTFA] first-speech latency breakdown per reply (log-only diagnostics): armed at
            // the reply's first text delta, one console line when playback actually starts.
            bool ttfaArmed;
            float ttfaFeed, ttfaQueue, ttfaSynth, ttfaRing;

            void Update()
            {
                if (source != null && source.pitch != pitch) source.pitch = pitch;
                PumpPipeline();
                if (streamClip == null) return;

                int buffered = RingCount();
                if (buffered > 0) lastNonEmptyRealtime = Time.realtimeSinceStartup;

                // mid-reply starvation: ring empty while MORE synthesis is coming (distinct from
                // the natural end-of-reply drain, where nothing is in flight).
                bool moreComing = streamJob != null || clauseQueue.Count > 0 || feedingText;
                lock (ringLock) moreAudioExpected = moreComing;   // the audio thread's dropout test
                // But STARVATION is narrower than "more audio is coming" (fix 2026-07-28): with
                // feedingText in the test, a ring that empties while the LLM still owes tokens counted
                // as a starve even with nothing in flight to synthesize. PumpPipeline has already
                // returned in that state, so the warning blamed the voice's tier for a slow LLM — and
                // the LLM is being HELD by the reverse arbiter at that moment, so the two fed each
                // other — and it re-gated playback on a text stall it could do nothing about. Worse,
                // it inflated the very counter the refill-floor decision is read from.
                bool synthInFlight = streamJob != null || clauseQueue.Count > 0;
                bool starving = streamStarted && buffered == 0 && synthInFlight;
                if (starving && !wasStarved)
                {
                    wasStarved = true;
                    underruns++;
                    // Report EVERY starve (fix 2026-07-26, kept 2026-07-27). It used to be silent on
                    // the first one of a session and, once both escalation ceilings were reached,
                    // silent forever — `underruns = 0` lived inside the escalation branches, so the
                    // counter latched and a clean log was read as "no underrun happened". It did not
                    // mean that. With the ladder deleted this is the whole response to a starve: the
                    // diagnostic. Report the tier's FLOOR against this clause's measured dead window,
                    // because that comparison is the diagnosis: playback coasts through the dead
                    // window on banked audio, so a floor below it is a starve waiting for its turn.
                    // (The old text advised dropping a tier. On Smooth that was a dead end — it and
                    // Very Smooth carry the same TTS ticks — so the line named a fix that did not
                    // exist. Print the numbers instead and let them point at the column.)
                    Debug.LogWarning($"[PocketTTSVoice] ring starved mid-reply (#{underruns}) — " +
                                     $"tier {BackendTradeoffTable.Label}: floor " +
                                     $"{BackendTradeoffTable.TtsRefillFloorSeconds:F2}s vs last clause dead window " +
                                     $"{(ttfaRing > 0f && ttfaSynth > 0f ? (ttfaRing - ttfaSynth) : 0f) * 1000f:F0}ms, " +
                                     $"{maxTicksLastFrame} ticks/frame allowed, " +
                                     $"prebuffer {prebufferSeconds:F1}s, chunk {streamChunkFrames}f.");
                    // DON'T re-gate at all when the dry spell is a clause's own BOUNDED dead window
                    // (fix 2026-07-27). This is the amplifier that produced the reported symptom, and
                    // the numbers are damning: measured 0.08-0.40 s of actual dry ring turning into
                    // 1.20-2.00 s of silence, because playback then sat waiting to bank a whole second
                    // before resuming. A tenth of a second nobody would notice became a freeze the
                    // player called out. Re-gating exists to avoid word...pause...word dribble when
                    // synthesis genuinely cannot keep up — but a clause prefill is a KNOWN, finite
                    // ~410 ms of producing nothing (see prefillBoost), with samples guaranteed on the
                    // far side of it. Waiting is the wrong response: resume the instant audio exists.
                    // Only a starve with no clause in flight — synthesis actually losing the race —
                    // still re-gates, and then on the short threshold rather than the full prebuffer.
                    if (ClausePrefilling)
                    {
                        // leave streamStarted TRUE: OnPcmRead zero-fills without advancing ringRead, so
                        // nothing is lost and the words continue verbatim the moment samples land.
                        regateSeconds = -1f;
                    }
                    else
                    {
                        regateSeconds = Mathf.Min(prebufferSeconds, InferencePerf.TtsRegateSeconds);
                        streamStarted = false;   // silence; the start branch below re-arms at regateSeconds
                    }
                }
                else if (!starving) wasStarved = false;
                // start at the prebuffer threshold — or as soon as the whole reply is synthesized
                // (short replies never reach the threshold; without this they'd sit forever)
                bool synthIdle = streamJob == null && clauseQueue.Count == 0 && !feedingText;
                float gateSeconds = regateSeconds > 0f ? regateSeconds : prebufferSeconds;
                if (!streamStarted && buffered > 0 &&
                    (buffered >= gateSeconds * Cfg.SAMPLE_RATE || synthIdle))
                {
                    streamStarted = true;
                    regateSeconds = -1f;   // next gate is a fresh reply's — back to the full prebuffer
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
                                  $"chunk {streamChunkFrames}f, tier {BackendTradeoffTable.Label}, " +
                                  $"silentTicks {BackendTradeoffTable.TtsSilentTicksPerFrame})");
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
                    long rd, wr, zDry, zGate; int zB;
                    lock (ringLock)
                    {
                        rd = totalRead; wr = totalWritten;
                        zDry = zeroDrySamples; zGate = zeroGateSamples; zB = zeroBursts;
                    }
                    if (rd >= wr && Time.realtimeSinceStartup - lastNonEmptyRealtime >= pauseGraceSeconds)
                    {
                        // Drain any mark still queued BEFORE clearing streamStarted — the reveal loop
                        // below requires it, so a mark left here would survive into the next reply and
                        // be typed into its bubble. Everything pushed has been consumed by now (rd >= wr
                        // is the branch condition), so these clauses have all been heard.
                        while (true)
                        {
                            string late = null; float lateDur = 0f;
                            lock (ringLock)
                            {
                                if (spokenQueue.Count == 0) break;
                                ClauseMark mk = spokenQueue.Dequeue();
                                if (mk.text != null)
                                {
                                    late = mk.text;
                                    lateDur = mk.end > mk.start ? (mk.end - mk.start) / (float)Cfg.SAMPLE_RATE
                                                                : mk.text.Length * 0.065f;
                                }
                            }
                            if (late != null) OnClauseSpoken?.Invoke(late, lateDur);
                        }
                        streamStarted = false;
                        source.Pause();
                        // play-mode assert proxy for "the last words were audible": everything
                        // pushed was consumed BEFORE the pause, and the tail had grace to play out.
                        // #33: plus how much silence the player heard INSIDE the reply. This is the
                        // number that settles "it stopped mid-sentence" — 0 means every gap they heard
                        // was an inter-clause pause, not a dropout.
                        Debug.Log($"[PocketTTSVoice] pause after drain: read {rd} / pushed {wr} " +
                                  $"(+{pauseGraceSeconds:F1}s grace) — tail fully played. " +
                                  $"in-reply silence {(zDry + zGate) / (float)Cfg.SAMPLE_RATE:F2}s " +
                                  $"({zDry / (float)Cfg.SAMPLE_RATE:F2}s dry + " +
                                  $"{zGate / (float)Cfg.SAMPLE_RATE:F2}s re-gated) in {zB} bursts.");
                        lock (ringLock) { zeroDrySamples = 0; zeroGateSamples = 0; zeroBursts = 0; }
                    }
                }

                // audio-synced clause reveal: pop every clause whose MEASURED playback position has
                // been reached, ~clauseRevealLead early.
                //
                // History, because this spot has now failed in BOTH directions:
                //  - raw totalRead (pre 2026-07-26): the PCM reader runs AHEAD of the sound, so text
                //    led the voice ("textul a fost randat inaintea intregului audio").
                //  - totalRead - audioTailSeconds (2026-07-26..30): an ESTIMATE of that lead, tuned
                //    in the starved-ring era. The moment #33 made the ring healthy the reader ran a
                //    different (shorter-lag) regime and the fixed 0.8 s over-corrected: text trailed
                //    the voice by the difference (user 2026-07-30). Any constant here is regime-
                //    dependent and breaks when TTS throughput changes.
                // So it is MEASURED now, no estimate anywhere: the audio thread stamps each mark
                // with the STREAM position where its first sample was handed to the reader
                // (streamStart — zero-fill included, so dropouts shift text exactly like they shift
                // sound), and audiblePlayed below is the stream position the DSP has actually
                // played, tracked from source.timeSamples. Reveal fires when played crosses stamp —
                // within one DSP buffer (~20-40 ms) of the speakers by construction.
                if (source != null && streamClip != null)
                {
                    int ts = source.timeSamples;
                    if (source.isPlaying)
                    {
                        int len = streamClip.samples;
                        audiblePlayed += (((long)ts - lastTimeSamples) % len + len) % len;   // wrap-safe
                    }
                    lastTimeSamples = ts;   // paused/stopped frames just resync (delta 0 anyway)
                }
                long lead = (long)(clauseRevealLead * Cfg.SAMPLE_RATE);
                while (true)
                {
                    string fire = null; float dur = 0f; bool dequeued = false;
                    lock (ringLock)
                    {
                        // The `totalRead >= mk.end` fallback stays: a mark whose WHOLE clause has been
                        // handed to the reader has nothing honest left to wait for. Without it the
                        // LAST clause of a short reply could sit un-revealed into the drain path (the
                        // 2026-07-26 stale-mark regression, caught in audit) — and it also covers a
                        // mark that somehow never got its stamp.
                        if (spokenQueue.Count > 0 && streamStarted &&
                            ((spokenQueue.Peek().streamStart >= 0 &&
                              audiblePlayed + lead >= spokenQueue.Peek().streamStart) ||
                             (spokenQueue.Peek().end > 0 && totalRead >= spokenQueue.Peek().end)))
                        {
                            ClauseMark mk = spokenQueue.Dequeue();
                            dequeued = true;
                            if (mk.text != null)   // token-fed clauses carry no text -> nothing to reveal
                            {
                                fire = mk.text;
                                dur = mk.end > mk.start
                                    ? (mk.end - mk.start) / (float)Cfg.SAMPLE_RATE
                                    : mk.text.Length * 0.065f;   // still synthesizing — chars estimate (~15 chars/s speech)
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
                        if (!streamStarted || ringCount == 0)
                        {
                            data[i] = 0f;                                   // starved: silence
                            // #33: this loop is the ONLY place that knows what the player heard. Split
                            // the two causes — a dry ring during live playback vs. a mid-reply re-gate
                            // waiting on the ring again — because they need different fixes. Neither
                            // counts once no more audio is expected: the ring is SUPPOSED to sit empty
                            // through the end-of-reply grace while the DSP tail plays out.
                            if (moreAudioExpected)
                            {
                                if (streamStarted) zeroDrySamples++; else zeroGateSamples++;
                                if (!zeroRunOpen) { zeroRunOpen = true; zeroBursts++; }
                            }
                            continue;
                        }
                        zeroRunOpen = false;
                        data[i] = ring[ringRead];
                        ringRead = (ringRead + 1) % ring.Length;
                        ringCount--;
                        totalRead++;                            // consumed (zero-fill doesn't count)
                        // stamp: this stream sample is a queued clause's FIRST sample — record
                        // where it sits in STREAM time, for the measured reveal in Update. One
                        // Peek per real sample; marks are stamped in enqueue order.
                        while (stampQueue.Count > 0 && totalRead > stampQueue.Peek().start)
                            stampQueue.Dequeue().streamStart = streamPos + i;
                    }
                    streamPos += data.Length;   // reader consumption advances stream time, zeros too
                }
            }

            // Disabling the GameObject kills every coroutine WITHOUT running its continuation, so two
            // latches have to be released by hand or the voice never speaks again (audit 2026-07-28 —
            // same failure class as the refcount bug documented above: "B never spoke again"):
            //   fadingOut  — set by FadeOutAndStop, cleared only at the end of FadeOutRoutine. Left
            //                true, PushSamples drops EVERY sample for the rest of the session and
            //                FadeOutAndStop early-returns on the non-null fadeJob.
            //   prewarmJob — PumpPipeline returns unconditionally while it is non-null, and `warmed` is
            //                already true by then so PrewarmKernels will not restart it.
            void OnDisable()
            {
                if (fadeJob != null) { StopCoroutine(fadeJob); fadeJob = null; }
                if (fadingOut)
                {
                    fadingOut = false;
                    if (source != null) source.volume = fadePrevVolume;
                }
                if (prewarmJob != null) { StopCoroutine(prewarmJob); prewarmJob = null; }
            }

            void OnDestroy()
            {
                holders.Remove(this);   // a destroyed voice must not pin the shared weights
                if (streamClip != null) Destroy(streamClip);
            }
        }
    }
}
