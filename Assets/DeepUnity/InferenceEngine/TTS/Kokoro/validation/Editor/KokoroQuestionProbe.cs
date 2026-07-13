using System;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Empirical check that sentence-final punctuation actually shapes prosody: synthesizes
        // the same sentence as statement (.) vs question (?) on the CPU oracle and compares the
        // F0 tail. Editor-mode synchronous (ClaudeBridge menu) — no play mode, no GPU.
        // Report: ProbeLogs/kokoro_question_report.md + per-case wavs for listen QA.
        public static class KokoroQuestionProbe
        {
            const string WEIGHTS = "Assets/Resources/Weights/weights_kokoro_fp16";
            const string G2P = "Assets/DeepUnity/InferenceEngine/TTS/Kokoro/KokoroG2P";
            const string REPORT = "ProbeLogs/kokoro_question_report.md";

            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[KokoroQuestion] " + s); }

            [MenuItem("DeepUnity/TTS/Run Kokoro Question-Prosody Probe")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                Log($"# Kokoro question prosody A/B — {DateTime.Now:yyyy-MM-dd HH:mm}");

                var tensors = new KokoroTensors(WEIGHTS);
                var cpu = new KokoroCPU(tensors);
                var g2p = new KokoroG2P(G2P);
                while (!g2p.IsReady) System.Threading.Thread.Sleep(50);

                string sentence = "You really believe the northern pass is safe";
                foreach (string voice in new[] { "velmire_elder", "af_heart" })
                {
                    float[] pack = tensors.D($"voices/{voice}");
                    float tailDot = 0, tailQ = 0;
                    foreach (char punct in new[] { '.', '?' })
                    {
                        string text = sentence + punct;
                        string ps = g2p.Phonemize(text);
                        int[] ids = cpu.PhonemesToIds(ps);
                        bool punctInPs = ps.Length > 0 && ps[ps.Length - 1] == punct;
                        bool punctInIds = ids.Length > 1 && ids[ids.Length - 2] == (punct == '?' ? 6 : 4);
                        var rng = new System.Random(1234);   // same noise both cases
                        var S = cpu.Forward(ids, Row(pack, ps.Length), 1f, U01(rng), N01(rng));

                        // mean voiced F0 over the last 25% of frames — where the rise lives
                        int F = S.F0.Length, from = F - F / 4; float sum = 0; int n = 0;
                        for (int i = from; i < F; i++) if (S.F0[i] > 1f) { sum += S.F0[i]; n++; }
                        float tail = n > 0 ? sum / n : 0;
                        if (punct == '.') tailDot = tail; else tailQ = tail;

                        string wavPath = $"ProbeLogs/kokoro_q_{voice}_{(punct == '?' ? "question" : "statement")}.wav";
                        SaveWav(wavPath, S.wav);
                        Log($"- {voice} '{punct}': ps-ends-with-punct {(punctInPs ? "YES" : "**NO**")}, " +
                            $"id-present {(punctInIds ? "YES" : "**NO**")}, tail-F0 {tail:F1} Hz, " +
                            $"{S.wav.Length / 24000f:F2}s -> {wavPath}");
                    }
                    float rise = tailQ - tailDot;
                    Log($"- **{voice}: question tail-F0 rise = {rise:+0.0;-0.0} Hz** " +
                        $"({(rise > 5 ? "question intonation PRESENT" : "FLAT — punctuation reaches the model but barely moves prosody")})");
                }

                File.WriteAllText(REPORT, report.ToString());
                Log($"report -> {REPORT}");
            }

            static float[] Row(float[] pack, int psLen)
            {
                var r = new float[256];
                Array.Copy(pack, (psLen - 1) * 256, r, 0, 256);
                return r;
            }

            static Func<int, float[]> U01(System.Random rng) => n =>
            {
                var a = new float[n];
                for (int i = 0; i < n; i++) a[i] = (float)rng.NextDouble();
                return a;
            };

            static Func<int, float[]> N01(System.Random rng) => n =>
            {
                var a = new float[n];
                for (int j = 0; j < n; j += 2)
                {
                    double r1 = 1 - rng.NextDouble(), rr = rng.NextDouble();
                    double m = Math.Sqrt(-2 * Math.Log(r1));
                    a[j] = (float)(m * Math.Cos(2 * Math.PI * rr));
                    if (j + 1 < n) a[j + 1] = (float)(m * Math.Sin(2 * Math.PI * rr));
                }
                return a;
            };

            static void SaveWav(string path, float[] wav)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int n = wav.Length;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + n * 2);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16); w.Write((short)1);
                w.Write((short)1); w.Write(24000); w.Write(24000 * 2); w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(n * 2);
                foreach (float s in wav) w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}
