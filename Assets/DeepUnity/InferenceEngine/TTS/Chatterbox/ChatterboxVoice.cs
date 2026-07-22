using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    // The Unity-facing voice of an NPC: attach next to (or let it add) an AudioSource and call
    // Say(text). Owns a ChatterboxTTS instance (or shares a global one via SetSharedTTS) and
    // handles BOTH playback modes:
    //
    //   1. Clip mode (default today): synthesize the full utterance -> AudioClip -> Play().
    //      Simple, correct, ~RTF-limited latency (whole utterance must finish synthesizing).
    //
    //   2. Streaming mode (the real-time path): a persistent looping AudioClip created with
    //      stream:true whose PCMReaderCallback drains a lock-protected ring buffer on the AUDIO
    //      thread. Producers (the TTS pipeline, chunk by chunk) push samples from the main thread
    //      via PushSamples(); playback starts once `prebufferSeconds` is buffered and pauses when
    //      starved. This is the component the sentence-chunked LLM->TTS pipeline feeds, so speech
    //      begins after the FIRST clause is synthesized instead of the whole reply.
    //
    // Spatialization, volume, mixer routing etc. are all AudioSource settings — this component
    // only produces samples.
    [RequireComponent(typeof(AudioSource))]
    public class ChatterboxVoice : MonoBehaviour
    {
        [Tooltip("Play chunk-wise through a streaming ring buffer (real-time mode) instead of one clip per utterance.")]
        public bool streaming = false;

        [Tooltip("Streaming: seconds buffered before playback starts (time-to-first-audio vs underrun safety).")]
        public float prebufferSeconds = 0.4f;

        [Tooltip("Streaming: ring buffer capacity in seconds.")]
        public float ringSeconds = 30f;

        [Tooltip("Playback pitch. <1 = deeper & slower (0.88-0.92 reads as an elder voice).")]
        public float pitch = 1f;

        [Tooltip("Loudness gain multiplied into the synthesized samples (AudioSource.volume tops out at 1 — this can go above it; peaks clamp at full scale).")]
        [Min(0f)] public float volume = 1f;

        [Tooltip("Min characters before a comma/semicolon may cut a speech chunk (sentence enders always cut).")]
        public int minChunkChars = 40;

        [Tooltip("First chunk cuts at this smaller size so speech starts ASAP; later chunks use minChunkChars.")]
        public int firstChunkChars = 16;

        [Tooltip("Sentences per synthesized chunk. Smaller = faster response, lower quality (prosody resets each sentence); larger = higher quality (intonation flows across sentences), slower response. Soft cuts (comma/semicolon/colon past the min size) still fire immediately.")]
        [Range(1, 3)] public int clausesPerChunk = 1;

        [Tooltip("Milliseconds of main-thread pump per frame for the TTS pipelines. This decouples " +
                 "synthesis speed from the framerate: T3 (next chunk) and S3Gen (current chunk) both " +
                 "advance every frame within this budget, in parallel with the LLM's own coroutine.")]
        public float gpuBudgetMs = 6f;

        [Tooltip("Voice conds to load (\"conds\" = baked default; \"conds_elder\" after make_voice.py).")]
        public string voiceName = "conds";

        [Tooltip("TTS weight format. INT8 = T3's matmuls int8 (~300 MB less VRAM/disk, ~lossless); s3gen stays fp16 either way. Needs the int8 export (import_params.py chatterbox --quant int8).")]
        public LLMQuant quantization = LLMQuant.FP16;

        public bool IsSpeaking { get; private set; }
        public bool IsReady => tts != null && tts.IsReady;

        static ChatterboxTTS shared;
        ChatterboxTTS tts;
        AudioSource source;

        // ---- budget-pumped pipeline state (FeedText -> chunker -> T3 -> S3Gen -> ring buffer) ----
        readonly System.Text.StringBuilder pendingText = new System.Text.StringBuilder();
        readonly Queue<string> chunkQueue = new Queue<string>();       // text awaiting T3
        readonly Queue<List<int>> tokenQueue = new Queue<List<int>>(); // speech tokens awaiting S3Gen
        IEnumerator t3Job, s3Job;                                      // in-flight stage iterators
        List<int> t3Out;
        bool feedingText, firstCutDone;
        readonly System.Diagnostics.Stopwatch pumpWatch = new System.Diagnostics.Stopwatch();

        // ring buffer (audio thread reads, main thread writes)
        float[] ring;
        int ringWrite, ringRead, ringCount;
        readonly object ringLock = new object();
        AudioClip streamClip;
        bool streamStarted;

        /// <summary>All ChatterboxVoice instances share one TTS engine by default (one weight set
        /// on the GPU). Call this early (e.g. from a prewarm script) to control creation timing.</summary>
        public static ChatterboxTTS SetSharedTTS(ChatterboxTTS instance) => shared = instance;

        void Awake()
        {
            source = GetComponent<AudioSource>();
            // TTS construction is deferred to Start so a spawner (e.g. NPCInteractor3D) can set
            // voiceName after AddComponent but before the engine loads.
        }

        void Start() => EnsureTts();

        void EnsureTts()
        {
            if (tts != null) return;
            shared ??= new ChatterboxTTS(voice: voiceName, quantization: quantization);
            tts = shared;
        }

        void OnDestroy()
        {
            if (streamClip != null) Destroy(streamClip);
        }

        /// <summary>Speak a full utterance. Streams tokens internally; in clip mode the audio plays
        /// when synthesis finishes, in streaming mode playback starts as soon as samples arrive
        /// (today: after synthesis of the utterance; with the chunked pipeline: after the first chunk).</summary>
        public void Say(string text) => StartCoroutine(SayRoutine(text));

        public IEnumerator SayRoutine(string text)
        {
            EnsureTts();
            while (!tts.IsReady) yield return null;
            IsSpeaking = true;

            if (!streaming)
            {
                AudioClip clip = null;
                yield return tts.Speak(text, c => clip = c);
                if (clip != null)
                {
                    source.clip = clip;
                    source.Play();
                    yield return new WaitWhile(() => source.isPlaying);
                }
                IsSpeaking = false;
            }
            else
            {
                // route through the budget-pumped pipeline (same path as streamed LLM text)
                FeedText(text);
                FlushText();
                yield return null;
                while (IsSpeaking) yield return null;
            }
        }

        // ---------------- streamed-text interface (the real-time LLM->speech path) --------------
        // Feed the LLM's token deltas as they arrive; complete clauses are cut and synthesized
        // while the rest of the reply is still generating, playing through the streaming ring.
        //   llm.Chat(q, onTokenGenerated: t => { window.Append(t); voice.FeedText(t); });
        //   voice.FlushText();   // after the reply ends — speaks the trailing partial clause

        public void FeedText(string delta)
        {
            if (string.IsNullOrEmpty(delta)) return;
            feedingText = true;
            pendingText.Append(delta);
            CutCompleteChunks();
            EnsurePump();
        }

        public void FlushText()
        {
            CutCompleteChunks();
            string rest = pendingText.ToString().Trim();
            pendingText.Clear();
            if (rest.Length > 0) chunkQueue.Enqueue(rest);
            feedingText = false;
            EnsurePump();
        }

        /// <summary>Hard stop: drop in-flight synthesis, queued text and buffered audio.</summary>
        public void StopSpeaking()
        {
            t3Job = null; s3Job = null; t3Out = null;
            chunkQueue.Clear();
            tokenQueue.Clear();
            pendingText.Clear();
            feedingText = false;
            firstCutDone = false;
            IsSpeaking = false;
            lock (ringLock) { ringCount = 0; ringRead = 0; ringWrite = 0; }
            if (streamStarted) { streamStarted = false; source.Pause(); }
            if (!streaming) source.Stop();
        }

        void CutCompleteChunks()
        {
            // sentence cuts land after the Nth ender (clausesPerChunk): batched sentences render
            // as ONE utterance with flowing prosody. Soft cuts (, ; : past the min size) stay
            // immediate — they are the latency/length guard, and the FIRST chunk keeps its much
            // smaller threshold (time-to-first-audio beats prosody there). Loops: one delta can
            // complete several chunks.
            while (true)
            {
                string s = pendingText.ToString();
                int minChars = firstCutDone ? minChunkChars : firstChunkChars;
                int need = clausesPerChunk < 1 ? 1 : clausesPerChunk, enders = 0, cut = -1;
                for (int i = 0; i < s.Length; i++)
                {
                    char c = s[i];
                    if ((c == ',' || c == ';' || c == ':') && i >= minChars) { cut = i; break; }
                    bool sentenceEnd = c == '.' || c == '!' || c == '?' || c == '\n';
                    if (!sentenceEnd) continue;
                    // an ender run ("...", "?!") counts once, at its end; a run touching the
                    // buffer end waits for the next delta (FlushText covers the reply end)
                    if (i + 1 >= s.Length || s[i + 1] == '.' || s[i + 1] == '!' || s[i + 1] == '?' || s[i + 1] == '\n') continue;
                    if (++enders >= need) { cut = i; break; }
                }
                if (cut < 0) return;
                string chunk = s.Substring(0, cut + 1).Trim();
                if (chunk.Length > 1) { chunkQueue.Enqueue(chunk); firstCutDone = true; }
                pendingText.Clear();
                pendingText.Append(s.Substring(cut + 1));
            }
        }

        void EnsurePump() => streaming = true;   // the pump lives in Update(); nothing to start

        // ---- the budget pump: BOTH stages advance every frame, decoupled from the framerate ----
        // T3 decodes chunk N+1 WHILE S3Gen synthesizes chunk N (their GPU dispatches interleave in
        // the command queue with the LLM's) — this is what makes speech continuous instead of
        // sentence-by-sentence bursts. Runs inside gpuBudgetMs of main-thread time per frame.
        void PumpPipelines()
        {
            if (tts == null) EnsureTts();
            if (tts == null || !tts.IsReady) return;
            bool anyWork = t3Job != null || s3Job != null || chunkQueue.Count > 0 || tokenQueue.Count > 0;
            if (!anyWork)
            {
                if (IsSpeaking && BufferedSamples == 0 && !feedingText) { IsSpeaking = false; firstCutDone = false; }
                return;
            }
            IsSpeaking = true;
            EnsureStream();

            pumpWatch.Restart();
            bool progressed = true;
            while (pumpWatch.Elapsed.TotalMilliseconds < gpuBudgetMs && progressed)
            {
                progressed = false;

                // stage 1: T3 — decode the NEXT chunk's speech tokens (sync sampler: fps-decoupled)
                if (t3Job == null && chunkQueue.Count > 0)
                {
                    t3Out = new List<int>();
                    t3Job = tts.GenerateSpeechTokens(chunkQueue.Dequeue(), t3Out, syncSample: true);
                }
                if (t3Job != null)
                {
                    if (!t3Job.MoveNext())
                    {
                        if (t3Out.Count > 0) tokenQueue.Enqueue(t3Out);
                        t3Job = null; t3Out = null;
                    }
                    progressed = true;
                }

                // stage 2: S3Gen — audio for the CURRENT chunk
                if (s3Job == null && tokenQueue.Count > 0)
                    s3Job = tts.SynthesizeFromTokens(tokenQueue.Dequeue(), w => { if (w != null) PushSamples(w); });
                if (s3Job != null)
                {
                    if (!s3Job.MoveNext()) s3Job = null;
                    progressed = true;
                }
            }
        }

        // ---------------- streaming machinery ----------------
        public int BufferedSamples { get { lock (ringLock) return ringCount; } }

        void EnsureStream()
        {
            if (streamClip != null) return;
            int cap = Mathf.CeilToInt(ringSeconds * ChatterboxTTS.SampleRate);
            ring = new float[cap];
            // Looping streamed clip; Unity pulls samples on the audio thread via the callback.
            streamClip = AudioClip.Create("ChatterboxStream", ChatterboxTTS.SampleRate, 1,
                                          ChatterboxTTS.SampleRate, true, OnAudioRead);
            source.clip = streamClip;
            source.loop = true;
        }

        /// <summary>Feed synthesized samples (24kHz mono) into the streaming ring buffer.
        /// Call from the main thread; the audio thread drains it.</summary>
        public void PushSamples(float[] samples)
        {
            EnsureStream();
            lock (ringLock)
            {
                for (int i = 0; i < samples.Length; i++)
                {
                    if (ringCount >= ring.Length) break;   // full: drop (better than blocking main thread)
                    ring[ringWrite] = volume == 1f ? samples[i] : Mathf.Clamp(samples[i] * volume, -1f, 1f);
                    ringWrite = (ringWrite + 1) % ring.Length;
                    ringCount++;
                }
            }
        }

        void Update()
        {
            if (source.pitch != pitch) source.pitch = pitch;
            PumpPipelines();
            if (streamClip == null) return;
            int buffered = BufferedSamples;
            if (!streamStarted && buffered >= prebufferSeconds * ChatterboxTTS.SampleRate)
            {
                streamStarted = true;
                source.Play();
            }
            else if (streamStarted && buffered == 0 && !IsSpeaking)
            {
                streamStarted = false;
                source.Pause();   // starved and nothing coming: silence without ticking the ring
            }
        }

        // AUDIO THREAD: fill Unity's requested block from the ring; zeros when empty (silence).
        void OnAudioRead(float[] data)
        {
            lock (ringLock)
            {
                int n = Mathf.Min(data.Length, ringCount);
                for (int i = 0; i < n; i++)
                {
                    data[i] = ring[ringRead];
                    ringRead = (ringRead + 1) % ring.Length;
                }
                ringCount -= n;
                for (int i = n; i < data.Length; i++) data[i] = 0f;
            }
        }
    }
}
