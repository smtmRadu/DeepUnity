#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // RTF-only perf probe — needs NO Python reference dumps (text is tokenized with the C#
        // SentencePiece encoder), so it benchmarks the LOCAL GPU on any machine, unlike the
        // P4/P5 parity probes whose dump/ folder lives on the main dev box. Reports the offline
        // KV path (prefill/loop/decode breakdown) and a streaming wall-clock run. Informational
        // only: no parity gates.
        public static class PocketTTSRtfProbe
        {
            const string TEXT =
                "The old lighthouse keeper climbed the spiral stairs every evening at dusk. " +
                "He lit the great lamp and watched the beam sweep across the darkening waves; " +
                "ships far at sea counted on that light to find their way home safely.";

            [MenuItem("DeepUnity/PocketTTS/RTF Benchmark (fp16)")]
            public static void RunFp16() => Run(PocketTTSConfig.WEIGHTS_DIR_FP16, "fp16");

            [MenuItem("DeepUnity/PocketTTS/RTF Benchmark (int8)")]
            public static void RunInt8() => Run(PocketTTSConfig.WEIGHTS_DIR_INT8, "int8");

            // #30: same run but with PocketTTSMimi.StageProfile on — the decode serializes at each
            // stage (sync readback splits), so the TOTAL is inflated; read the splits as SHARES.
            [MenuItem("DeepUnity/PocketTTS/RTF Benchmark (int8, decode stage profile)")]
            public static void RunInt8Profiled() => Run(PocketTTSConfig.WEIGHTS_DIR_INT8, "int8", profile: true);

            static void Run(string weightsDir, string tag, bool profile = false)
            {
                PocketTTS tts = null;
                try
                {
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    EditorUtility.DisplayProgressBar("pocket-tts RTF", "Loading weights…", 0.1f);
                    tts = new PocketTTS(weightsDir);
                    tts.LoadBlocking();
                    float loadMs = (float)sw.Elapsed.TotalMilliseconds;

                    int[] ids = tts.Tokenize(TEXT);

                    EditorUtility.DisplayProgressBar("pocket-tts RTF", "Warmup (shader compiles)…", 0.3f);
                    tts.GenerateOffline(ids, null, useKvCache: true);   // untimed: compiles every kernel

                    EditorUtility.DisplayProgressBar("pocket-tts RTF", "Offline KV run…", 0.5f);
                    if (profile) PocketTTSMimi.StageProfile = new System.Collections.Generic.List<(string, float)>();
                    float[] wav = tts.GenerateOffline(ids, null, useKvCache: true);
                    if (profile)
                    {
                        var prof = PocketTTSMimi.StageProfile;
                        PocketTTSMimi.StageProfile = null;
                        float ptotal = 0f; foreach (var (_, ms) in prof) ptotal += ms;
                        var sb = new System.Text.StringBuilder();
                        sb.AppendLine($"[PocketRTF] {tag} MIMI DECODE STAGE PROFILE (serialized total {ptotal:F0} ms — shares are the signal):");
                        foreach (var (name, ms) in prof)
                            sb.AppendLine($"    {name,-16} {ms,8:F1} ms   {100f * ms / ptotal,5:F1}%");
                        Debug.Log(sb.ToString());
                    }
                    float sec = wav.Length / (float)PocketTTSConfig.SAMPLE_RATE;
                    float total = tts.GenMs + tts.DecodeMs;
                    Debug.Log($"[PocketRTF] {tag} OFFLINE (KV): {ids.Length} ids -> {sec:F2}s audio | " +
                              $"prefill {tts.PrefillMs:F0} + loop {tts.LoopMs:F0} + mimi decode {tts.DecodeMs:F0} ms | " +
                              $"total {total:F0} ms -> RTF {total / 1000f / sec:F3} | TTFA(proxy) {tts.TtfaMs:F0} ms | " +
                              $"load {loadMs:F0} ms | GPU {SystemInfo.graphicsDeviceName}");

                    // NO edit-mode streaming run here: a tight while(MoveNext) pump over
                    // SynthesizeStreaming can deadlock the editor main thread when the async GPU
                    // readbacks never progress without player-loop frames (froze the editor on the
                    // GTX 1650 box, 2026-07-13 — P5 got away with it on the dev machine). The
                    // streaming number belongs to play mode: use NpcTalkPerfProbe / the demos.
                }
                finally
                {
                    tts?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }
        }
    }
}
#endif
