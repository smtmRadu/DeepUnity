using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    // The Unity-facing voice for Kokoro TTS: attach next to (or let it add) an AudioSource and
    // call Say(text). Mirrors ChatterboxVoice's two playback modes:
    //
    //   1. Clip mode (default): synthesize the full utterance -> AudioClip -> Play().
    //   2. Streaming mode: a persistent looping AudioClip (stream:true) whose PCMReaderCallback
    //      drains a lock-protected ring buffer on the AUDIO thread. KokoroTTS's per-CHUNK
    //      callback pushes samples as each text chunk finishes (Kokoro is non-autoregressive —
    //      chunk-level streaming ONLY, SPEC §12.2: InstanceNorm uses whole-chunk statistics, so
    //      the first audio arrives after the first chunk synthesizes, not per-token).
    //
    // All KokoroVoice instances share one KokoroTTS engine by default (one weight set / one CPU
    // worker); SetSharedTTS to control creation timing (e.g. a loading screen prewarm).
    [RequireComponent(typeof(AudioSource))]
    public class KokoroVoice : MonoBehaviour
    {
        [Tooltip("Play chunk-wise through a streaming ring buffer instead of one clip per utterance.")]
        public bool streaming = false;

        [Tooltip("Streaming: seconds buffered before playback starts (latency vs underrun safety).")]
        public float prebufferSeconds = 0.4f;

        [Tooltip("Streaming: ring buffer capacity in seconds.")]
        public float ringSeconds = 30f;

        [Tooltip("Voicepack to load (any voices/<name> row in the weights manifest).")]
        public string voiceName = "af_heart";

        [Tooltip("Weights folder. int8 = matmuls quantized (max recon err ~0.01), smaller upload; norms/convs/voices stay fp16.")]
        public string weightsPath = "Assets/Resources/Weights/weights_kokoro_int8";

        [Tooltip("Speech speed (divides predicted durations; 1 = reference pace).")]
        public float speed = 1f;

        [Tooltip("Playback pitch. <1 = deeper & slower.")]
        public float pitch = 1f;

        [Tooltip("Loudness gain multiplied into the synthesized samples (AudioSource.volume tops out at 1 — this can go above it; peaks clamp at full scale).")]
        [Min(0f)] public float volume = 1f;

        [Tooltip("Build + start loading the shared TTS in Start(). Off = call PrefetchNow() when " +
                 "the player gets close, so the weights stream on approach instead of scene load.")]
        public bool loadOnStart = true;

        [Tooltip("Fed text cuts ONLY at sentence enders (. ! ? ;). A comma may cut too, but only past this many pending characters — an escape hatch for run-on sentences, not the normal path.")]
        public int emergencyChunkChars = 220;

        public bool IsSpeaking { get; private set; }
        public bool IsReady => tts != null && tts.IsReady;

        /// <summary>True while buffered speech is actually audible (ring playing + non-empty) —
        /// drive talk animations from THIS, not from IsSpeaking (which includes synthesis).</summary>
        public bool IsAudioPlaying => streamStarted && RingCount() > 0;

        static KokoroTTS shared;
        KokoroTTS tts;
        AudioSource source;
        Coroutine sayJob;

        // clause queue for the streamed-text interface (LLM token deltas)
        readonly System.Text.StringBuilder pendingText = new System.Text.StringBuilder();
        readonly Queue<string> clauseQueue = new Queue<string>();
        bool feedingText;

        // ---- audio-synced text reveal: fires (on the main thread) the moment a fed clause's
        // audio actually STARTS playing, ~leadSeconds early, carrying the clause's spoken
        // DURATION in seconds so the UI can pace a word-by-word reveal across it. Duration is
        // exact when the synth already finished writing the clause (the common case at RTF~0.3);
        // for a clause still synthesizing it falls back to a chars-based estimate.
        public event Action<string, float> OnClauseSpoken;
        [Tooltip("OnClauseSpoken fires this many seconds BEFORE the clause is audible (text may lead the voice slightly).")]
        public float clauseRevealLead = 0.35f;
        sealed class ClauseMark { public string text; public long start; public long end = -1; }
        readonly Queue<ClauseMark> spokenQueue = new Queue<ClauseMark>();
        long totalWritten, totalRead;   // monotonic sample counters (ringLock-guarded)

        // ring buffer (audio thread reads, main thread writes)
        float[] ring;
        int ringWrite, ringRead, ringCount;
        readonly object ringLock = new object();
        AudioClip streamClip;
        bool streamStarted;

        /// <summary>Share one engine across all voices; call early to control creation timing.</summary>
        public static KokoroTTS SetSharedTTS(KokoroTTS instance) => shared = instance;

        void Awake() => source = GetComponent<AudioSource>();

#if UNITY_EDITOR
        // engine buffers are released by the ModelBase teardown sweep — the statics must not
        // outlive them. InitializeOnLoadMethod so the hook survives every domain reload.
        [UnityEditor.InitializeOnLoadMethod]
        static void HookEditorTeardown()
        {
            UnityEditor.EditorApplication.playModeStateChanged += s =>
            {
                if (s == UnityEditor.PlayModeStateChange.ExitingPlayMode) ClearShared();
            };
            UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += ClearShared;
        }

        static void ClearShared()
        {
            shared = null;
            warmed = false;
        }
#endif

        void Start() { if (loadOnStart) EnsureTts(); }

        /// <summary>Build the shared TTS and start streaming its weights (budgeted, no frame
        /// drops). Call from a proximity trigger for load-on-approach.</summary>
        public void PrefetchNow() => EnsureTts();

        /// <summary>Load-on-approach, spread over ~targetSeconds (SlowPrefetch): tiny per-frame
        /// upload slices while the player walks up.</summary>
        public void SlowPrefetchNow(float targetSeconds)
        {
            EnsureTts();
            tts.SlowPrefetch(targetSeconds);
        }

        /// <summary>Release the GPU weights (budgeted, frame-friendly). Prefetch again re-streams.</summary>
        public void DefetchNow() => tts?.Defetch(DefetchMode.Slow);

        /// <summary>One tiny discarded synthesis once the weights are resident — compiles every
        /// kernel path so the first REAL clause has no shader-compile hitches. Call it where the
        /// player isn't looking (prefetch zone / loading screen).</summary>
        public void PrewarmKernels()
        {
            if (!warmed && prewarmJob == null) prewarmJob = StartCoroutine(PrewarmRoutine());
        }

        static bool warmed;   // per-session, engine is shared
        Coroutine prewarmJob;

        IEnumerator PrewarmRoutine()
        {
            EnsureTts();
            while (!tts.IsReady) yield return null;
            if (!warmed)
            {
                warmed = true;
                float[] _ = null;
                yield return tts.Synthesize("Mm.", w => _ = w);
            }
            prewarmJob = null;
        }

        void EnsureTts()
        {
            if (tts != null) return;
            shared ??= new KokoroTTS(weightsPath, voice: voiceName);
            tts = shared;
        }

        // ---------------- streamed-text interface (LLM token deltas) ----------------
        // Kokoro is non-autoregressive: each completed clause synthesizes as one fast chunk
        // (RTF ~0.3 on a 4060), so speech starts right after the first clause is cut — while
        // the rest of the reply is still generating. Requires streaming mode.
        public void FeedText(string delta)
        {
            if (string.IsNullOrEmpty(delta)) return;
            EnsureTts();
            streaming = true;
            feedingText = true;
            pendingText.Append(delta);
            CutCompleteChunks();
        }

        public void FlushText()
        {
            CutCompleteChunks();
            string rest = pendingText.ToString().Trim();
            pendingText.Clear();
            if (rest.Length > 0) clauseQueue.Enqueue(rest);
            feedingText = false;
        }

        void CutCompleteChunks()
        {
            // sentence-level cuts only — prosody stays whole; the pump already synthesizes the
            // next sentence WHILE the current one plays, so there is no latency reason to cut
            // at commas (the comma path exists purely for run-on sentences)
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
            if (chunk.Length > 1) clauseQueue.Enqueue(chunk);
            pendingText.Clear();
            pendingText.Append(s.Substring(cut + 1));
        }

        void Update()
        {
            if (source != null && source.pitch != pitch) source.pitch = pitch;
            // clause pump: one synthesis in flight (the engine has a single scratch set —
            // never overlap with a running prewarm either); queued clauses share the ring
            if (sayJob == null && prewarmJob == null && clauseQueue.Count > 0)
            {
                IsSpeaking = true;
                sayJob = StartCoroutine(SpeakClauseRoutine(clauseQueue.Dequeue()));
            }
            else if (IsSpeaking && sayJob == null && clauseQueue.Count == 0 && !feedingText && RingCount() == 0)
            {
                IsSpeaking = false;
            }

            // audio-synced reveal: pop every clause whose playback position has been reached
            long lead = (long)(clauseRevealLead * KokoroTTS.SAMPLE_RATE);
            while (true)
            {
                string fire = null; float dur = 0f;
                lock (ringLock)
                {
                    if (spokenQueue.Count > 0 && totalRead + lead >= spokenQueue.Peek().start && streamStarted)
                    {
                        ClauseMark m = spokenQueue.Dequeue();
                        fire = m.text;
                        dur = m.end > m.start
                            ? (m.end - m.start) / (float)KokoroTTS.SAMPLE_RATE
                            : m.text.Length * 0.055f;   // still synthesizing — chars estimate
                    }
                }
                if (fire == null) break;
                OnClauseSpoken?.Invoke(fire, dur);
            }
        }

        IEnumerator SpeakClauseRoutine(string clause)
        {
            yield return tts.Warmup();   // no-op once kernels are warm
            EnsureStreamClip();
            // single synthesis in flight -> the clause's first sample lands at totalWritten
            var mark = new ClauseMark { text = clause };
            lock (ringLock) { mark.start = totalWritten; spokenQueue.Enqueue(mark); }
            bool done = false;
            yield return tts.Synthesize(clause, _ => done = true, PushSamples, speed);
            while (!done) yield return null;
            lock (ringLock) mark.end = totalWritten;   // exact spoken duration now known
            sayJob = null;
        }

        /// <summary>Speak the text. Interrupts any utterance in progress.</summary>
        public void Say(string text)
        {
            EnsureTts();
            if (sayJob != null) StopCoroutine(sayJob);
            if (streaming) ClearRing();
            sayJob = StartCoroutine(SayRoutine(text));
        }

        // ---- leave-fade: voice doesn't cut on dialogue close — it fades to silence -------------
        Coroutine fadeJob;
        float fadePrevVolume = 1f;

        /// <summary>Fade smoothly to silence over <paramref name="seconds"/>, then hard-stop and
        /// restore the volume for the next utterance (mirrors PocketTTSVoice.FadeOutAndStop).</summary>
        public void FadeOutAndStop(float seconds = 1f)
        {
            if (fadeJob != null) return;
            if (!IsSpeaking && !IsAudioPlaying) { StopSpeaking(); return; }
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
            if (source != null) source.volume = fadePrevVolume;
        }

        public void StopSpeaking()
        {
            if (fadeJob != null)
            {
                StopCoroutine(fadeJob);
                fadeJob = null;
                if (source != null) source.volume = fadePrevVolume;
            }
            if (sayJob != null) { StopCoroutine(sayJob); sayJob = null; }
            clauseQueue.Clear();
            pendingText.Clear();
            feedingText = false;
            lock (ringLock) spokenQueue.Clear();
            if (streaming) ClearRing();
            else source.Stop();
            IsSpeaking = false;
        }

        IEnumerator SayRoutine(string text)
        {
            IsSpeaking = true;
            yield return tts.Warmup();
            if (streaming)
            {
                EnsureStreamClip();
                bool done = false;
                yield return tts.Synthesize(text, _ => done = true, PushSamples, speed);
                // hold IsSpeaking until the ring drains
                while (!done || RingCount() > 0) yield return null;
            }
            else
            {
                AudioClip clip = null;
                yield return tts.Speak(text, c => clip = c);
                if (clip != null)
                {
                    source.clip = clip;
                    source.Play();
                    while (source.isPlaying) yield return null;
                }
            }
            IsSpeaking = false;
            sayJob = null;
        }

        // ---------------- streaming ring buffer ----------------
        void EnsureStreamClip()
        {
            if (streamClip != null) return;
            ring = new float[Mathf.CeilToInt(ringSeconds * KokoroTTS.SAMPLE_RATE)];
            // 1s looping clip; PCMReaderCallback pulls from the ring on the audio thread
            streamClip = AudioClip.Create("KokoroStream", KokoroTTS.SAMPLE_RATE, 1,
                                          KokoroTTS.SAMPLE_RATE, true, OnPcmRead);
            source.clip = streamClip;
            source.loop = true;
        }

        int RingCount() { lock (ringLock) return ringCount; }

        void ClearRing()
        {
            lock (ringLock) { ringRead = ringWrite = ringCount = 0; }
            streamStarted = false;
        }

        /// <summary>Main-thread producer: KokoroTTS onChunk lands here per synthesized chunk.</summary>
        public void PushSamples(float[] samples)
        {
            if (ring == null) return;
            lock (ringLock)
            {
                foreach (float s in samples)
                {
                    if (ringCount == ring.Length) break;   // full: drop tail (ringSeconds exceeded)
                    ring[ringWrite] = volume == 1f ? s : Mathf.Clamp(s * volume, -1f, 1f);
                    ringWrite = (ringWrite + 1) % ring.Length;
                    ringCount++;
                    totalWritten++;
                }
            }
            if (!streamStarted && RingCount() >= prebufferSeconds * KokoroTTS.SAMPLE_RATE)
            {
                streamStarted = true;
                if (!source.isPlaying) source.Play();
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
                    totalRead++;
                }
            }
        }

        void OnDestroy()
        {
            if (streamClip != null) Destroy(streamClip);
        }
    }
}
