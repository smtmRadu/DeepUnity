using System;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // P1 parity probe — Mimi decoder (latents[T,32] -> 24kHz wav) vs the python dump
        // (validation/dump_reference.py). EDITOR-MODE + SYNCHRONOUS (ClaudeBridge invoke, no play
        // mode). Grades wav corr > 0.99 + the mimi_upsampled_f0 / mimi_xf_out_f0 intermediates.
        public static class PocketTTSMimiProbe
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_mimi_parity.md";
            const string DONE = "ProbeLogs/pockettts_mimi_parity.done";

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketMimi] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P1 Mimi Parity")]
            public static void Run() { WEIGHTS = WEIGHTS_FP16; RunInner(); }

            [MenuItem("DeepUnity/PocketTTS/P1 Mimi Parity (int8)")]
            public static void RunInt8()
            {
                WEIGHTS = WEIGHTS_INT8;
                try { RunInner(); } finally { WEIGHTS = WEIGHTS_FP16; }
            }

            static void RunInner()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                PocketTTSWeights weights = null;
                PocketTTSMimi mimi = null;
                try
                {
                    Log($"# pocket-tts P1 — Mimi decoder parity — {DateTime.Now:yyyy-MM-dd HH:mm}");

                    float[] latents = Floats("latents", out int[] ls);      // [T,32] — ALREADY denormed (quantizer input)
                    int T = ls[0];
                    float[] refWav = Floats("wav", out int[] ws);           // [S]
                    Log($"dump: latents [{T},{ls[ls.Length - 1]}], wav {refWav.Length} samples");

                    weights = new PocketTTSWeights(WEIGHTS, beginLoad: false);
                    weights.LoadBlocking("mimi/");
                    Log("mimi/* weights resident (blocking load).");

                    // intermediate gates: mimi_upsampled_f0 [1,512,16], mimi_xf_out_f0 [1,512,16]
                    // (first-latent outputs, channel-major). Compare our first 16 frames.
                    var interm = new List<string>();
                    mimi = new PocketTTSMimi(weights);
                    mimi.DebugTap = (name, buf, count) =>
                    {
                        // quant_out gate: ref quant_out_f0 [512] (single frame). upsample/xf: [1,512,16].
                        if (name == "quant_out")
                        {
                            string qp = Path.Combine(DUMP, "quant_out_f0.npy");
                            if (!File.Exists(qp)) return;
                            float[] qref = (float[])LoadNpy(qp, out int[] qs);   // [512] or [1,512,1]
                            int C0 = qref.Length;
                            float[] q0 = new float[C0];
                            buf.GetData(q0, 0, 0, C0);                           // our frame-0 quant out [512]
                            var (qx, qmae, qcorr) = Diff(q0, qref);
                            interm.Add($"   {"quant_out_f0",-18} [{C0}] maxAbs {qx:F4} MAE {qmae:F5} corr {qcorr:F6}" + (qcorr < 0.99f ? "  <-- LOW" : ""));
                            return;
                        }
                        string refName = name == "mimi_upsampled" ? "mimi_upsampled_f0"
                                       : name == "mimi_xf_out" ? "mimi_xf_out_f0"
                                       : name.StartsWith("seanet_") ? name + "_f0" : null;
                        if (refName == null) return;
                        string p = Path.Combine(DUMP, refName + ".npy");
                        if (!File.Exists(p)) return;
                        float[] rf = (float[])LoadNpy(p, out int[] rs);     // [1,512,16] -> C=512,steps=16
                        int C = rs[1], steps = rs[2];
                        float[] refTC = TransposeCT(rf, C, steps);          // [16,512]
                        float[] ours = new float[steps * C];
                        buf.GetData(ours, 0, 0, steps * C);                 // our first `steps` frames [steps,512]
                        var (mx, mae, corr) = Diff(ours, refTC);
                        interm.Add($"   {refName,-18} [{steps},{C}] maxAbs {mx:F4} MAE {mae:F5} corr {corr:F6}" + (corr < 0.99f ? "  <-- LOW" : ""));

                        // deep-tail gate: compare our latent-40 slice (abs frames 640..) to the ref call-40
                        // output. Directly confirms the context-window fix (was the cross-frame divergence).
                        if (name == "mimi_xf_out")
                        {
                            string p40 = Path.Combine(DUMP, "mimi_xf_out_f40.npy");
                            if (File.Exists(p40))
                            {
                                float[] rf40 = (float[])LoadNpy(p40, out int[] rs40);
                                int C4 = rs40[1], st4 = rs40[2];
                                float[] ref40 = TransposeCT(rf40, C4, st4);     // [16,512]
                                float[] o40 = new float[st4 * C4];
                                buf.GetData(o40, 0, 40 * 16 * C4, st4 * C4);     // GPU offset = frame 640
                                var (x4, m4, c4) = Diff(o40, ref40);
                                interm.Add($"   {"mimi_xf_out_f40",-18} [{st4},{C4}] maxAbs {x4:F4} MAE {m4:F5} corr {c4:F6}" + (c4 < 0.99f ? "  <-- LOW (context window)" : ""));
                            }
                        }
                    };

                    float[] wav = mimi.Decode(latents, T);   // latents already denormed (P1 isolates the Mimi decoder)

                    if (interm.Count > 0) { Log("## Intermediate gates:"); foreach (var s in interm) Log(s); }

                    var (wMax, wMae, wCorr) = Diff(wav, refWav);
                    Log($"## WAV: ours {wav.Length} vs ref {refWav.Length}; maxAbs {wMax:F4} MAE {wMae:F5} corr {wCorr:F6}");
                    if (wCorr < 0.99f) { failed = true; Log("   FAIL (corr < 0.99)"); }
                    SaveWav("ProbeLogs/pockettts_mimi_unity.wav", wav, PocketTTSConfig.SAMPLE_RATE);
                    float sec = wav.Length / (float)PocketTTSConfig.SAMPLE_RATE;
                    Log($"   [perf] decode {mimi.DecodeMs:F0} ms for {sec:F2}s (RTF {mimi.DecodeMs / 1000f / sec:F3}); wav -> ProbeLogs/pockettts_mimi_unity.wav");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    mimi?.Dispose();
                    weights?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            static Array LoadNpy(string path, out int[] shape)
            {
                byte[] all = File.ReadAllBytes(path);
                if (all[0] != 0x93) throw new Exception($"not npy: {path}");
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);
                string shapeStr = header.Substring(header.IndexOf("'shape':", StringComparison.Ordinal) + 8);
                shapeStr = shapeStr.Substring(shapeStr.IndexOf('(') + 1);
                shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
                var dims = new List<int>();
                foreach (string s in shapeStr.Split(','))
                    if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                shape = dims.ToArray();
                long count = 1; foreach (int dd in shape) count *= dd;
                if (!header.Contains("f4")) throw new Exception($"expected f4 npy: {header}");
                float[] r = new float[count];
                Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                return r;
            }

            static float[] Floats(string name, out int[] shape)
                => (float[])LoadNpy(Path.Combine(DUMP, name + ".npy"), out shape);

            static (float, float, float) Diff(float[] a, float[] b)
            {
                int n = Mathf.Min(a.Length, b.Length);
                double mx = 0, mae = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    double dd = Math.Abs(a[i] - b[i]); mx = Math.Max(mx, dd); mae += dd;
                    sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return ((float)mx, (float)(mae / n), (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-20))));
            }

            static float[] TransposeCT(float[] src, int C, int Tn)   // [C,T] -> [T,C]
            {
                float[] r = new float[C * Tn];
                for (int cc = 0; cc < C; cc++)
                    for (int t = 0; t < Tn; t++)
                        r[t * C + cc] = src[cc * Tn + t];
                return r;
            }

            static void SaveWav(string path, float[] s, int sr)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var wr = new BinaryWriter(fs);
                int bl = s.Length * 2;
                wr.Write(Encoding.ASCII.GetBytes("RIFF")); wr.Write(36 + bl);
                wr.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); wr.Write(16);
                wr.Write((short)1); wr.Write((short)1); wr.Write(sr); wr.Write(sr * 2);
                wr.Write((short)2); wr.Write((short)16);
                wr.Write(Encoding.ASCII.GetBytes("data")); wr.Write(bl);
                foreach (float v in s) wr.Write((short)Mathf.Clamp(Mathf.RoundToInt(v * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}
