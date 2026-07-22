using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // The Unity-facing CosyVoice3 voice of an NPC (ChatterboxVoice's surface, CosyVoice's
        // guts): attach next to an AudioSource and call Say(text), or feed LLM token deltas via
        // FeedText/FlushText. Complete clauses are cut and synthesized while the reply is still
        // generating; INSIDE each clause the pipeline is token-level streaming (audio starts
        // ~28 speech tokens in, not after the whole clause) — samples land in a lock-protected
        // ring buffer drained by the audio thread.
        [RequireComponent(typeof(AudioSource))]
        public class CosyVoiceVoice : MonoBehaviour
        {
            [Tooltip("Streaming: seconds buffered before playback starts (time-to-first-audio vs underrun safety).")]
            public float prebufferSeconds = 0.5f;

            [Tooltip("Streaming: ring buffer capacity in seconds.")]
            public float ringSeconds = 60f;

            [Tooltip("Playback pitch. <1 = deeper & slower.")]
            public float pitch = 1f;

            [Tooltip("Loudness gain multiplied into the synthesized samples (AudioSource.volume tops out at 1 — this can go above it; peaks clamp at full scale).")]
            [Min(0f)] public float volume = 1f;

            [Tooltip("Fed text cuts ONLY at sentence enders (. ! ? ;). A comma may cut too, but only past this many pending characters — an escape hatch for run-on sentences.")]
            public int emergencyChunkChars = 220;

            [Tooltip("Sentences per synthesized chunk. Smaller = faster response, lower quality (prosody resets each sentence); larger = higher quality (intonation flows across sentences), slower response.")]
            [Range(1, 3)] public int clausesPerChunk = 1;

            [Tooltip("Milliseconds of main-thread pump per frame for the TTS pipeline.")]
            public float gpuBudgetMs = 6f;

            [Tooltip("Baked voice folder under the weights dir (voices/<name>/).")]
            public string voiceName = "default";

            [Tooltip("Weights folder. int8 = same speed & quality, half the VRAM and load bytes.")]
            public string weightsPath = "Assets/Resources/Weights/weights_cosyvoice3_int8";

            public bool IsSpeaking { get; private set; }
            public bool IsReady => tts != null && tts.IsReady;

            static CosyVoiceTTS shared;
            CosyVoiceTTS tts;
            AudioSource source;

            readonly System.Text.StringBuilder pendingText = new System.Text.StringBuilder();
            readonly Queue<string> clauseQueue = new Queue<string>();
            IEnumerator streamJob;                    // one clause in flight (LM cache is single)
            bool feedingText;

            [Tooltip("Buffer each COMPLETE clause before it may play. While synthesis is slower " +
                     "than real-time (streaming RTF > 1) sample-streaming starves mid-sentence " +
                     "and stutters; whole-clause mode trades latency for clean, uninterrupted " +
                     "sentences (gaps land BETWEEN clauses). Turn off once RTF < 1.")]
            public bool playWholeClauses = true;
            readonly List<float> clauseAccum = new List<float>();
            readonly System.Diagnostics.Stopwatch pumpWatch = new System.Diagnostics.Stopwatch();

            float[] ring;
            int ringWrite, ringRead, ringCount;
            readonly object ringLock = new object();
            AudioClip streamClip;
            bool streamStarted;

            /// <summary>All CosyVoiceVoice instances share one TTS engine (one weight set on GPU).</summary>
            public static CosyVoiceTTS SetSharedTTS(CosyVoiceTTS instance) => shared = instance;

            [Tooltip("OFF = load-on-approach: nothing streams until PrefetchNow/SlowPrefetchNow " +
                     "(the NPC prefetch zone drives it, mirroring KokoroVoice).")]
            public bool loadOnStart = true;

            void Awake() => source = GetComponent<AudioSource>();

#if UNITY_EDITOR
            // engine buffers are released by the ModelBase teardown sweep — the static must not
            // outlive them. InitializeOnLoadMethod so the hook survives every domain reload.
            [UnityEditor.InitializeOnLoadMethod]
            static void HookEditorTeardown()
            {
                UnityEditor.EditorApplication.playModeStateChanged += s =>
                {
                    if (s == UnityEditor.PlayModeStateChange.ExitingPlayMode) shared = null;
                };
                UnityEditor.AssemblyReloadEvents.beforeAssemblyReload += () => shared = null;
            }
#endif
            void Start() { if (loadOnStart) EnsureTts(); }

            void EnsureTts(bool beginLoad = true)
            {
                if (tts != null) return;
                shared ??= new CosyVoiceTTS(weightsPath, voice: voiceName, beginLoad: beginLoad);
                tts = shared;
            }

            // ---- residency wrappers (NPC prefetch zone / talk trigger), KokoroVoice-shaped ----

            /// <summary>Start (or boost) the weight stream at full speed.</summary>
            public void PrefetchNow() { EnsureTts(beginLoad: false); tts.Prefetch(); }

            /// <summary>Load-on-approach spread over ~targetSeconds (budgeted per frame).</summary>
            public void SlowPrefetchNow(float targetSeconds)
            {
                EnsureTts(beginLoad: false);
                tts.SlowPrefetch(targetSeconds);
            }

            /// <summary>Unload the weights (safe mid-prefetch); a later prefetch starts fresh.</summary>
            public void DefetchNow() => tts?.Defetch();

            void OnDestroy()
            {
                if (streamClip != null) Destroy(streamClip);
            }

            public void Say(string text) => StartCoroutine(SayRoutine(text));

            public IEnumerator SayRoutine(string text)
            {
                EnsureTts();
                while (!tts.IsReady) yield return null;
                FeedText(text);
                FlushText();
                yield return null;
                while (IsSpeaking) yield return null;
            }

            // ---------------- streamed-text interface (LLM token deltas) ------------------------
            public void FeedText(string delta)
            {
                if (string.IsNullOrEmpty(delta)) return;
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

            /// <summary>Hard stop: drop in-flight synthesis, queued text and buffered audio.</summary>
            public void StopSpeaking()
            {
                streamJob = null;
                clauseQueue.Clear();
                pendingText.Clear();
                clauseAccum.Clear();
                feedingText = false;
                IsSpeaking = false;
                lock (ringLock) { ringCount = 0; ringRead = 0; ringWrite = 0; }
                if (streamStarted) { streamStarted = false; source.Pause(); }
            }

            void CutCompleteChunks()
            {
                // sentence-level cuts only (comma = run-on escape hatch; see KokoroVoice). The cut
                // lands after the Nth sentence ender (clausesPerChunk, see TtsClauseCut) so batched
                // sentences render as ONE utterance with flowing prosody. Loops: one delta can
                // complete several chunks.
                while (true)
                {
                    string s = pendingText.ToString();
                    int cut = TtsClauseCut.FindCut(s, clausesPerChunk, emergencyChunkChars);
                    if (cut < 0) return;
                    string chunk = s.Substring(0, cut + 1).Trim();
                    if (chunk.Length > 1) clauseQueue.Enqueue(chunk);
                    pendingText.Clear();
                    pendingText.Append(s.Substring(cut + 1));
                }
            }

            // ---- budget pump: advance the in-flight clause every frame within gpuBudgetMs -------
            void PumpPipeline()
            {
                if (tts == null || !tts.IsReady) return;
                bool anyWork = streamJob != null || clauseQueue.Count > 0;
                if (!anyWork)
                {
                    if (IsSpeaking && BufferedSamples == 0 && !feedingText) IsSpeaking = false;
                    return;
                }
                IsSpeaking = true;
                EnsureStream();

                pumpWatch.Restart();
                while (pumpWatch.Elapsed.TotalMilliseconds < gpuBudgetMs)
                {
                    if (streamJob == null && clauseQueue.Count > 0)
                        streamJob = tts.SynthesizeStreaming(clauseQueue.Dequeue(), w =>
                        {
                            if (w == null) return;
                            // whole-clause mode holds samples aside until the clause completes,
                            // so the playing ring only ever contains FINISHED sentences
                            if (playWholeClauses) clauseAccum.AddRange(w);
                            else PushSamples(w);
                        });
                    if (streamJob == null) break;
                    if (!streamJob.MoveNext())
                    {
                        streamJob = null;
                        if (clauseAccum.Count > 0)
                        {
                            PushSamples(clauseAccum.ToArray());
                            clauseAccum.Clear();
                        }
                    }
                }
            }

            // ---------------- streaming machinery ------------------------------------------------
            public int BufferedSamples { get { lock (ringLock) return ringCount; } }

            void EnsureStream()
            {
                if (streamClip != null) return;
                int sr = CosyVoiceConfig.SAMPLE_RATE;
                ring = new float[Mathf.CeilToInt(ringSeconds * sr)];
                streamClip = AudioClip.Create("CosyVoiceStream", sr, 1, sr, true, OnAudioRead);
                source.clip = streamClip;
                source.loop = true;
            }

            public void PushSamples(float[] samples)
            {
                EnsureStream();
                lock (ringLock)
                {
                    for (int i = 0; i < samples.Length; i++)
                    {
                        if (ringCount >= ring.Length) break;
                        ring[ringWrite] = volume == 1f ? samples[i] : Mathf.Clamp(samples[i] * volume, -1f, 1f);
                        ringWrite = (ringWrite + 1) % ring.Length;
                        ringCount++;
                    }
                }
            }

            void Update()
            {
                if (source.pitch != pitch) source.pitch = pitch;
                PumpPipeline();
                if (streamClip == null) return;
                int buffered = BufferedSamples;
                // start at the prebuffer threshold — or as soon as the whole queued reply is
                // synthesized (short replies never reach the threshold; without this they'd
                // sit in the ring forever and the NPC stays silent)
                bool synthIdle = streamJob == null && clauseQueue.Count == 0 && !feedingText;
                // whole-clause mode: anything in the ring is a COMPLETE sentence — play at once
                if (!streamStarted && buffered > 0 &&
                    (playWholeClauses || buffered >= prebufferSeconds * CosyVoiceConfig.SAMPLE_RATE || synthIdle))
                {
                    streamStarted = true;
                    source.Play();
                }
                else if (streamStarted && buffered == 0 && !IsSpeaking)
                {
                    streamStarted = false;
                    source.Pause();
                }
            }

            // AUDIO THREAD
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
}
