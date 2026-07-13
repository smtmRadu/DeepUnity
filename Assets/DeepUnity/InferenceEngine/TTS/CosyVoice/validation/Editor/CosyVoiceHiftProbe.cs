using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // A1 parity probe — CausalHiFT vocoder vs the Python reference dump
        // (validation/dump_reference.py). EDITOR-MODE + SYNCHRONOUS: pumps the vocoder
        // coroutine to completion inside one invoke, so it runs through ClaudeBridge with the
        // editor open (no play mode, no batch mode needed).
        //
        // Grades (chatterbox recipe — vocoder isolated from the LM/flow by feeding the
        // reference flow_mel and injecting the reference NSF source):
        //   F0   predicted F0 vs hift_f0.npy      (fp32 GPU vs float64 reference — SPEC §3)
        //   WAV  waveform vs wav.npy, corr > 0.99 + audio saved for listening
        public static class CosyVoiceHiftProbe
        {
            const string DUMP_DIR = "Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump";
            const string WEIGHTS_DIR = "Assets/Resources/Weights/weights_cosyvoice3_fp16";
            const string REPORT = "ProbeLogs/cosyvoice_hift_parity.md";
            const string DONE = "ProbeLogs/cosyvoice_hift_parity.done";

            static readonly StringBuilder report = new StringBuilder();

            static void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[CosyVoiceHiftParity] " + line);
            }

            [MenuItem("DeepUnity/CosyVoice/A1 HiFT Vocoder Parity")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                CosyVoiceWeights weights = null;
                HiFTVocoder voc = null;
                ComputeBuffer melBuf = null;
                try
                {
                    Log($"# CosyVoice3 A1 — CausalHiFT vocoder parity — {DateTime.Now:yyyy-MM-dd HH:mm}");

                    // reference tensors ([B,C,T] channel-major -> [T,C])
                    float[] melRaw = Floats("flow_mel", out int[] ms);          // [1, 80, Tg]
                    int Tg = ms[2];
                    float[] mel = TransposeCT(melRaw, ms[1], Tg);
                    float[] refF0 = Floats("hift_f0", out _);                   // [1, Tg]
                    float[] refSrc = Floats("hift_source", out _);              // [1, S, 1] -> flat [S]
                    float[] refWav = Floats("wav", out _);                      // [1, S]
                    Log($"dump: mel [{Tg},80], source {refSrc.Length}, wav {refWav.Length} samples");

                    weights = new CosyVoiceWeights(WEIGHTS_DIR, beginLoad: false);
                    weights.LoadBlocking("hift/");
                    Log("hift/* weights resident (blocking load).");

                    melBuf = new ComputeBuffer(Tg * CosyVoiceConfig.MEL_DIM, 4, ComputeBufferType.Structured);
                    melBuf.SetData(mel);

                    // per-stage refs (dump_hift_stages.py) — bisect mode when present
                    string stagesDir = Path.Combine(DUMP_DIR, "hift_stages");
                    var stageResults = new List<string>();
                    voc = new HiFTVocoder(weights) { InjectSource = refSrc };
                    if (Directory.Exists(stagesDir))
                        voc.DebugTap = (name, buf, count) =>
                        {
                            string p = Path.Combine(stagesDir, name + ".npy");
                            if (!File.Exists(p)) return;
                            float[] rf = (float[])LoadNpy(p, out int[] rs);      // [1,C,T]
                            float[] refTC = TransposeCT(rf, rs[1], rs[2]);
                            float[] ours = new float[count];
                            buf.GetData(ours, 0, 0, count);
                            var (sMax, sMae, sCorr) = Diff(ours, refTC);
                            stageResults.Add($"   stage {name,-10} [{rs[2]},{rs[1]}] " +
                                             $"maxAbs {sMax:F4}  MAE {sMae:F5}  corr {sCorr:F6}" +
                                             (sCorr < 0.995f ? "  <-- DIVERGES" : ""));
                        };
                    float[] wav = null;
                    IEnumerator it = voc.VocodeYielding(melBuf, Tg, w => wav = w);
                    while (it.MoveNext()) { }
                    if (stageResults.Count > 0)
                    {
                        Log("## Stage bisection (vs dump/hift_stages):");
                        foreach (string s in stageResults) Log(s);
                    }

                    // ---- F0 (deterministic conv chain; fp16 weights + fp64 reference -> drift)
                    float[] ourF0 = new float[Tg];
                    voc.DebugF0.GetData(ourF0, 0, 0, Tg);
                    var (fMax, fMae, fCorr) = Diff(ourF0, refF0);
                    Log($"## F0: maxAbs {fMax:F4}  MAE {fMae:F5}  corr {fCorr:F6}");
                    if (fCorr < 0.999f) { failed = true; Log("   FAIL (< 0.999)"); }

                    // ---- WAV
                    if (wav == null) { failed = true; Log("## WAV: SYNTHESIS FAILED (null)"); }
                    else
                    {
                        var (wMax, wMae, wCorr) = Diff(wav, refWav);
                        Log($"## WAV: ours {wav.Length} vs ref {refWav.Length} samples; " +
                            $"maxAbs {wMax:F4}  MAE {wMae:F5}  corr {wCorr:F6}");
                        if (wav.Length != refWav.Length || wCorr < 0.99f) { failed = true; Log("   FAIL (len mismatch or corr < 0.99)"); }
                        SaveWav("ProbeLogs/cosyvoice_hift_unity.wav", wav, CosyVoiceConfig.SAMPLE_RATE);
                        float sec = wav.Length / (float)CosyVoiceConfig.SAMPLE_RATE;
                        Log($"   [perf] vocoder {voc.VocoderMs:F0} ms for {sec:F2}s of audio " +
                            $"(RTF {voc.VocoderMs / 1000f / sec:F3}); audio -> ProbeLogs/cosyvoice_hift_unity.wav");
                    }
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    voc?.Dispose();
                    melBuf?.Release();
                    weights?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            // ---------------- minimal .npy loader (v1.0/2.0, little-endian, C-order) ------------
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
                long count = 1; foreach (int d in shape) count *= d;

                if (header.Contains("f4"))
                {
                    float[] r = new float[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    return r;
                }
                if (header.Contains("i8"))
                {
                    long[] r = new long[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8);
                    return r;
                }
                if (header.Contains("i4"))
                {
                    int[] r = new int[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    return r;
                }
                throw new Exception($"unsupported npy dtype in {path}: {header}");
            }

            static float[] Floats(string name, out int[] shape)
                => (float[])LoadNpy(Path.Combine(DUMP_DIR, name + ".npy"), out shape);

            static (float maxAbs, float mae, float corr) Diff(float[] a, float[] b)
            {
                int n = Mathf.Min(a.Length, b.Length);
                double maxAbs = 0, mae = 0, sa = 0, sb = 0, saa = 0, sbb = 0, sab = 0;
                for (int i = 0; i < n; i++)
                {
                    double d = Math.Abs(a[i] - b[i]);
                    maxAbs = Math.Max(maxAbs, d); mae += d;
                    sa += a[i]; sb += b[i]; saa += (double)a[i] * a[i]; sbb += (double)b[i] * b[i]; sab += (double)a[i] * b[i];
                }
                double cov = sab / n - (sa / n) * (sb / n);
                double va = saa / n - (sa / n) * (sa / n), vb = sbb / n - (sb / n) * (sb / n);
                return ((float)maxAbs, (float)(mae / n), (float)(cov / Math.Sqrt(Math.Max(va * vb, 1e-20))));
            }

            static float[] TransposeCT(float[] src, int C, int T)   // [C,T] -> [T,C]
            {
                float[] r = new float[src.Length];
                for (int c = 0; c < C; c++)
                    for (int t = 0; t < T; t++)
                        r[t * C + c] = src[c * T + t];
                return r;
            }

            static void SaveWav(string path, float[] samples, int sr)
            {
                using var fs = new FileStream(path, FileMode.Create);
                using var w = new BinaryWriter(fs);
                int byteLen = samples.Length * 2;
                w.Write(Encoding.ASCII.GetBytes("RIFF")); w.Write(36 + byteLen);
                w.Write(Encoding.ASCII.GetBytes("WAVEfmt ")); w.Write(16);
                w.Write((short)1); w.Write((short)1); w.Write(sr); w.Write(sr * 2);
                w.Write((short)2); w.Write((short)16);
                w.Write(Encoding.ASCII.GetBytes("data")); w.Write(byteLen);
                foreach (float s in samples)
                    w.Write((short)Mathf.Clamp(Mathf.RoundToInt(s * 32767f), short.MinValue, short.MaxValue));
            }
        }
    }
}
