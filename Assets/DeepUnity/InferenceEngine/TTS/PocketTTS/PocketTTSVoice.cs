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
            [Tooltip("Streaming: seconds buffered before playback starts (time-to-first-audio vs underrun safety).")]
            public float prebufferSeconds = 0.5f;

            [Tooltip("Streaming: ring buffer capacity in seconds.")]
            public float ringSeconds = 60f;

            [Tooltip("Playback pitch. <1 = deeper & slower.")]
            public float pitch = 1f;

            [Tooltip("Frames of audio produced between streaming decodes (chunk cadence). 8 = 0.64s @ 12.5Hz.")]
            public int streamChunkFrames = 8;

            [Tooltip("Milliseconds of main-thread pump per frame for the TTS pipeline.")]
            public float gpuBudgetMs = 6f;

            [Tooltip("Baked voice folder under the weights dir (voices/<name>/). Ignored if a clonedVoiceClip is set.")]
            public string voiceName = "jean";

            [Tooltip("Optional reference clip to CLONE this NPC's voice from (P8). Cloned + disk-cached " +
                     "on first use, then loaded from cache. Requires encoder weights (import_pocket_tts.py " +
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

            void Awake() => source = GetComponent<AudioSource>();
            void Start() { if (loadOnStart) EnsureTts(); }

            AudioClip _boundClip;   // the clip currently cloned into the shared engine (avoid re-clone)

            /// <summary>Assign this NPC's voice from a reference clip at runtime — cloned + disk-cached
            /// on first use, then loaded from cache. Overrides voiceName. Pass null to revert to baked.</summary>
            public void SetClonedVoice(AudioClip clip) { clonedVoiceClip = clip; _boundClip = null; }

            // Bind the right voice into the shared engine before a clause. Clone-clip takes priority
            // (cheap: cache hit is a file load, and the shared engine caches the bound clip so a
            // re-bind of the same clip is a no-op). Falls back to the baked voiceName otherwise.
            void BindVoice()
            {
                if (clonedVoiceClip != null)
                {
                    if (_boundClip == clonedVoiceClip) return;   // already bound this clip
                    if (tts.CloneVoice(clonedVoiceClip)) _boundClip = clonedVoiceClip;
                    else tts.SetVoice(voiceName);                 // encoder missing -> baked fallback
                }
                else
                {
                    _boundClip = null;
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

            /// <summary>Build the engine and start streaming weights at full speed (load-on-approach trigger).</summary>
            public void PrefetchNow() { EnsureTts(); tts.BeginLoad(); }

            /// <summary>Load-on-approach spread over ~targetSeconds (budgeted per frame).</summary>
            public void SlowPrefetchNow(float targetSeconds) { EnsureTts(); tts.SlowPrefetch(targetSeconds); }

            /// <summary>Unload the weights (budgeted); a later prefetch re-streams.</summary>
            public void DefetchNow() => tts?.Defetch(slow: true);

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
                    // a tiny real utterance: exercises tokenizer + prefill + KV decode + flow + Mimi
                    // decode + chunk stream, so the first real clause has no shader-compile hitch.
                    var e = tts.SynthesizeStreaming(tts.Tokenize("Hi."), _ => { }, maxFrames: 8);
                    while (e.MoveNext()) yield return null;
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
                if (ids != null && ids.Length > 0) clauseQueue.Enqueue((ids, text));
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
            public void StopSpeaking()
            {
                streamJob = null;
                inflightMark = null;
                clauseQueue.Clear();
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

                pumpWatch.Restart();
                while (pumpWatch.Elapsed.TotalMilliseconds < gpuBudgetMs)
                {
                    if (streamJob == null && clauseQueue.Count > 0)
                    {
                        BindVoice();               // clone-clip (cached) or baked voiceName — cheap rebind
                        tts.StreamChunkFrames = Mathf.Max(1, streamChunkFrames);
                        var (ids, text) = clauseQueue.Dequeue();
                        // clause mark: first sample of this clause lands at totalWritten (single
                        // synthesis in flight) -> OnClauseSpoken fires when playback reaches it
                        inflightMark = new ClauseMark { text = text };
                        lock (ringLock) { inflightMark.start = totalWritten; spokenQueue.Enqueue(inflightMark); }
                        streamJob = tts.SynthesizeStreaming(ids, PushSamples);
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
                    }
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
                EnsureStream();
                lock (ringLock)
                {
                    for (int i = 0; i < samples.Length; i++)
                    {
                        if (ringCount >= ring.Length) break;   // full: drop tail (ringSeconds exceeded)
                        ring[ringWrite] = samples[i];
                        ringWrite = (ringWrite + 1) % ring.Length;
                        ringCount++;
                        totalWritten++;                        // only STORED samples count (drops excluded)
                    }
                }
            }

            void Update()
            {
                if (source != null && source.pitch != pitch) source.pitch = pitch;
                PumpPipeline();
                if (streamClip == null) return;

                int buffered = RingCount();
                if (buffered > 0) lastNonEmptyRealtime = Time.realtimeSinceStartup;
                // start at the prebuffer threshold — or as soon as the whole reply is synthesized
                // (short replies never reach the threshold; without this they'd sit forever)
                bool synthIdle = streamJob == null && clauseQueue.Count == 0 && !feedingText;
                if (!streamStarted && buffered > 0 &&
                    (buffered >= prebufferSeconds * Cfg.SAMPLE_RATE || synthIdle))
                {
                    streamStarted = true;
                    if (!source.isPlaying) source.Play();
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
                if (streamClip != null) Destroy(streamClip);
            }
        }
    }
}
