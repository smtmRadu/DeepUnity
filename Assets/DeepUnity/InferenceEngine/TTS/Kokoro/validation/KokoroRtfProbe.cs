using System;
using System.Collections;
using System.Globalization;
using System.IO;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // RTF / TTFA / load benchmark for the BENCHMARK.md TTS table — same 3-sentence lighthouse
        // passage and metrics as PocketTTSRtfProbe, but PLAY MODE (Kokoro synthesis is a
        // coroutine + CPU-Task pipeline; edit-mode tight-pumping can deadlock the editor — see
        // the PocketTTSRtfProbe streaming note). One quant per run; summary.json follows the LM
        // probe conventions (machine block, InvariantCulture) so runs are self-describing.
        public class KokoroRtfProbe : MonoBehaviour
        {
            public string quant = "fp16";       // fp16 | int8 (weights_kokoro_<quant>)
            public string reportDirectory;      // set by KokoroRtfBatchRunner

            const string TEXT =
                "The old lighthouse keeper climbed the spiral stairs every evening at dusk. " +
                "He lit the great lamp and watched the beam sweep across the darkening waves; " +
                "ships far at sea counted on that light to find their way home safely.";
            const string Status = "ClaudeBridge/kokoro_rtf_status.txt";
            const string Marker = "ProbeLogs/kokoro_rtf.done";
            const int TIMED_RUNS = 3;

            IEnumerator Start()
            {
                var inv = CultureInfo.InvariantCulture;
                Directory.CreateDirectory(reportDirectory);
                Directory.CreateDirectory("ClaudeBridge");
                void Status_(string s) => File.WriteAllText(Status, $"[{DateTime.Now:HH:mm:ss}] {s}");
                bool ok = false;
                double loadMs = 0, wall = 0, ttfaMs = -1;
                float audioSec = 0;
                KokoroTTS tts = null;
                try
                {
                    Status_($"boot {quant}");
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    tts = new KokoroTTS($"Assets/Resources/Weights/weights_kokoro_{quant}");
                }
                catch (Exception e) { Debug.LogException(e); }
                if (tts == null) { Finish(false, 0, 0, 0, -1, inv); yield break; }

                var swLoad = System.Diagnostics.Stopwatch.StartNew();
                while (!tts.IsReady) yield return null;
                var wu = tts.Warmup();
                while (wu.MoveNext()) yield return wu.Current;
                loadMs = swLoad.Elapsed.TotalMilliseconds;
                Status_($"loaded {quant} in {loadMs:F0} ms, warm synth…");

                float[] full = null;
                var warm = tts.Synthesize(TEXT, w => full = w);   // untimed: compiles every kernel
                while (warm.MoveNext()) yield return warm.Current;
                if (full == null) { Finish(false, loadMs, 0, 0, -1, inv); yield break; }
                audioSec = full.Length / (float)KokoroTTS.SAMPLE_RATE;

                var walls = new double[TIMED_RUNS];
                for (int i = 0; i < TIMED_RUNS; i++)
                {
                    Status_($"timed run {i + 1}/{TIMED_RUNS}");
                    double t0 = Time.realtimeSinceStartupAsDouble, tFirst = -1;
                    var r = tts.Synthesize(TEXT, w => full = w,
                        w => { if (tFirst < 0) tFirst = Time.realtimeSinceStartupAsDouble; });
                    while (r.MoveNext()) yield return r.Current;
                    walls[i] = Time.realtimeSinceStartupAsDouble - t0;
                    if (i == 0 && tFirst > 0) ttfaMs = (tFirst - t0) * 1000.0;
                }
                Array.Sort(walls);
                wall = walls[TIMED_RUNS / 2];
                ok = full != null && audioSec > 1f;
                tts.Release();
                Finish(ok, loadMs, wall, audioSec, ttfaMs, inv);
            }

            void Finish(bool ok, double loadMs, double wall, float audioSec, double ttfaMs, CultureInfo inv)
            {
                double rtf = audioSec > 0 ? wall / audioSec : 0;
                string json =
                    "{\n" +
                    "  \"probe\": \"tts_rtf\",\n" +
                    "  \"model\": \"kokoro-82M\",\n" +
                    $"  \"quant\": \"{quant.ToUpperInvariant()}\",\n" +
                    $"  \"success\": {(ok ? "true" : "false")},\n" +
                    $"  \"audio_sec\": {audioSec.ToString("F2", inv)},\n" +
                    $"  \"gen_ms\": {(wall * 1000.0).ToString("F0", inv)},\n" +
                    $"  \"rtf\": {rtf.ToString("F3", inv)},\n" +
                    $"  \"ttfa_ms\": {ttfaMs.ToString("F0", inv)},\n" +
                    $"  \"load_ms\": {loadMs.ToString("F0", inv)},\n" +
                    $"  \"machine\": {LMProbeCommon.MachineJson()}\n" +
                    "}\n";
                File.WriteAllText(Path.Combine(reportDirectory, "summary.json"), json);
                Debug.Log($"[KokoroRTF] {quant}: {(ok ? "OK" : "FAIL")} | {audioSec:F2}s audio | " +
                          $"gen {wall * 1000:F0} ms -> RTF {rtf:F3} | TTFA {ttfaMs:F0} ms | load {loadMs:F0} ms");
                File.WriteAllText(Marker, ok ? "PASS" : "FAIL");
            }
        }
    }
}
