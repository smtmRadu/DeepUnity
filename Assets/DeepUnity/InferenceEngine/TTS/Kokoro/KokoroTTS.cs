using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    // Kokoro-82M text-to-speech for DeepUnity — 24 kHz mono, non-autoregressive (SPEC.md).
    //
    //   var tts = new KokoroTTS();                        // af_heart, weights prefetch begins
    //   yield return tts.Warmup();
    //   yield return tts.Speak("Hello world!", clip => audioSource.PlayOneShot(clip));
    //
    // ModelBase residency: Prefetch/SlowPrefetch/BoostFetch/PausePrefetch/Defetch all forward to
    // KokoroWeights (epoch-invalidated loader, CosyVoiceWeights pattern).
    //
    // The runtime is GPU-ONLY: KokoroModel dispatches KokoroCS.compute (fp16 weights / fp32
    // activations) on the main thread; the 6 biLSTMs + duration head + NSF phase pipeline run
    // CPU-side on worker Tasks inside that same pipeline (SPEC §9 hybrid boundary — sequential
    // microsecond-scale cells where dispatch overhead would dominate). KokoroCPU itself is the
    // VALIDATION ORACLE only (KokoroKernelProbe / validation/harness~), never a runtime backend.
    //
    // Chunking: one forward per text chunk (InstanceNorm uses whole-chunk statistics — SPEC
    // §12.2 — so there is NO intra-chunk streaming); sentences are packed greedily up to the
    // 510-phoneme limit. KokoroVoice streams chunk wavs into its ring buffer.
    public class KokoroTTS : TTS
    {
        readonly KokoroModeling.KokoroWeights weights;   // GPU residency store
        readonly KokoroModeling.KokoroTensors tensors;   // CPU fp32 weights (LSTM/NSF stages, voicepacks)
        readonly KokoroModeling.KokoroCPU cpu;           // CPU-stage host inside the GPU pipeline
        readonly KokoroModeling.KokoroModel model;       // the GPU forward
        readonly KokoroModeling.KokoroG2P g2p;
        readonly string voice;
        float[] voicepack;                               // [510*256], lazily read
        readonly System.Random rng = new System.Random();

        public const int SAMPLE_RATE = 24000;
        public const int MAX_PHONEMES = 510;

        public override int SampleRate => SAMPLE_RATE;
        readonly string paramsPath;
        public override string ResidencyLabel => ResidencyLog.Label(paramsPath);
        public override bool IsReady => g2p.IsReady && weights.IsReady;
        public override ModelResidency Residency => weights.Residency;
        public override long TotalWeightBytes => weights.BytesTotal;
        public override long UploadedWeightBytes => weights.BytesUploaded;
        public override long LoadBudgetBytesPerFrame
        {
            get => weights.BudgetBytesPerFrame;
            set => weights.BudgetBytesPerFrame = value;  // LIVE — pump samples it each frame
        }

        public KokoroTTS(
            string params_path = null,                   // null -> Resources resolve
            string voice = "af_heart",
            string g2p_path = "Assets/DeepUnity/InferenceEngine/TTS/Kokoro/KokoroG2P",
            bool prefetch = true)                        // false = construct cold, Prefetch() later
        {
            params_path ??= "Assets/Resources/Weights/weights_kokoro_fp16";
            paramsPath = params_path;
            if (!SystemInfo.supportsComputeShaders)
                ConsoleMessage.Error("KokoroTTS requires compute shader support (the runtime is GPU-only).");
            weights = new KokoroModeling.KokoroWeights(params_path, beginLoad: prefetch);
            tensors = new KokoroModeling.KokoroTensors(params_path);
            cpu = new KokoroModeling.KokoroCPU(tensors);
            model = new KokoroModeling.KokoroModel(weights, cpu);
            g2p = new KokoroModeling.KokoroG2P(g2p_path);
            if (!tensors.Has($"voices/{voice}"))
            {
                ConsoleMessage.Warning($"KokoroTTS: voice '{voice}' not exported — using af_heart. " +
                                       "Add the .pt to staging and re-run validation/import_kokoro.py.");
                voice = "af_heart";
            }
            this.voice = voice;

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += s =>
            { if (s == UnityEditor.PlayModeStateChange.ExitingPlayMode) Release(); };
#endif
        }

        protected override void StartPrefetch(long bytesPerFrame)
        {
            weights.BudgetBytesPerFrame = bytesPerFrame;
            weights.BeginLoad();
        }

        public override void Defetch(DefetchMode mode)
            => weights.Defetch(mode == DefetchMode.Slow ? Math.Max(64 * 1024, LoadBudgetBytesPerFrame) : 0);

        public override IEnumerator Warmup()
        {
            while (!IsReady) yield return null;
            voicepack ??= tensors.D($"voices/{voice}");
            yield return null;
        }

        /// <summary>Text -> 24 kHz samples via the GPU forward (KokoroModel); dispatches run on
        /// the main thread, the CPU LSTM/NSF stages on worker Tasks inside the pipeline.
        /// onChunk (optional) fires per chunk for streaming consumers.</summary>
        public IEnumerator Synthesize(string text, Action<float[]> onWav, Action<float[]> onChunk,
                                      float speed = 1f)
        {
            while (!IsReady) yield return null;
            voicepack ??= tensors.D($"voices/{voice}");

            List<string> chunks = Chunk(text);
            var all = new List<float>();
            var sw = System.Diagnostics.Stopwatch.StartNew();
            foreach (string chunk in chunks)
            {
                string ps = g2p.Phonemize(chunk);
                if (ps.Length == 0) continue;
                if (ps.Length > MAX_PHONEMES) ps = ps.Substring(0, MAX_PHONEMES);
                int[] ids = cpu.PhonemesToIds(ps);
                float[] refS = new float[256];
                Array.Copy(voicepack, (ps.Length - 1) * 256, refS, 0, 256);

                float[] wav = null;
                bool done = false;
                var fwd = model.ForwardYielding(ids, refS, speed, RandU, RandN,
                                                w => { wav = w; done = true; });
                while (fwd.MoveNext()) yield return fwd.Current;
                if (!done || wav == null)
                {
                    ConsoleMessage.Warning("KokoroTTS chunk failed (see earlier warnings).");
                    onWav?.Invoke(null);
                    yield break;
                }
                onChunk?.Invoke(wav);
                all.AddRange(wav);
                TokensPerSecond = all.Count / 600f / Math.Max(0.001f, (float)sw.Elapsed.TotalSeconds);
            }
            TokensPerSecond = 0;
            onWav?.Invoke(all.ToArray());
        }

        public override IEnumerator Synthesize(string text, Action<float[]> onWav)
            => Synthesize(text, onWav, null);

        float[] RandU(int n)
        {
            var a = new float[n];
            for (int i = 0; i < n; i++) a[i] = (float)rng.NextDouble();
            return a;
        }

        float[] RandN(int n)   // Box-Muller
        {
            var a = new float[n];
            for (int i = 0; i < n; i += 2)
            {
                double r1 = 1 - rng.NextDouble(), r2 = rng.NextDouble();
                double m = Math.Sqrt(-2 * Math.Log(r1));
                a[i] = (float)(m * Math.Cos(2 * Math.PI * r2));
                if (i + 1 < n) a[i + 1] = (float)(m * Math.Sin(2 * Math.PI * r2));
            }
            return a;
        }

        /// <summary>Greedy sentence packer: each chunk's PHONEME length stays ≤ 510 (estimated
        /// via ~1.05 phonemes/char then verified by the caller's hard truncate).</summary>
        static List<string> Chunk(string text)
        {
            var chunks = new List<string>();
            var current = new System.Text.StringBuilder();
            foreach (string sent in System.Text.RegularExpressions.Regex.Split(text, @"(?<=[.!?…])\s+"))
            {
                if (sent.Length == 0) continue;
                // phoneme strings run slightly longer than text; keep a safety margin
                if (current.Length > 0 && (current.Length + 1 + sent.Length) * 1.2f > MAX_PHONEMES)
                {
                    chunks.Add(current.ToString());
                    current.Clear();
                }
                if (current.Length > 0) current.Append(' ');
                current.Append(sent);
            }
            if (current.Length > 0) chunks.Add(current.ToString());
            return chunks;
        }

        public override void Release()
        {
            model?.Dispose();
            weights.Dispose();
        }
    }
}
