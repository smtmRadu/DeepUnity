using System;
using System.IO;
using System.Text;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // P8 probe — runtime voice-clone parity. Feeds the dump's fixed reference wav (voice_ref_audio)
        // through the C# Mimi encoder + speaker_proj (PocketTTS.EncodeToPrompt) and gates the result
        // against audio_prompt_ref (Python encode_to_latent -> speaker_proj) at corr>0.99. Also gates
        // the pre-speaker_proj latents (voice_ref_latents). Proves the encoder port + clone pipeline.
        // Requires the encoder weights: export with `import_params.py pocket-tts --include-encoder`.
        public static class PocketTTSCloneProbe
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_clone.md";
            const string DONE = "ProbeLogs/pockettts_clone.done";

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketClone] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P8 Voice Clone Parity")]
            public static void Run() { WEIGHTS = WEIGHTS_FP16; RunInner(); }

            [MenuItem("DeepUnity/PocketTTS/P8 Voice Clone Parity (int8)")]
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
                PocketTTS tts = null;
                try
                {
                    bool int8 = WEIGHTS == WEIGHTS_INT8;
                    Log($"# pocket-tts P8 — voice clone parity — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    Log($"weights: {WEIGHTS}");
                    tts = new PocketTTS(WEIGHTS);
                    tts.LoadBlocking();
                    if (!tts.HasEncoder)
                    {
                        failed = true;
                        Log("**FAIL**: encoder weights not in this dir. Re-export with " +
                            "`import_params.py pocket-tts --include-encoder` (and --quant int8 for the int8 dir).");
                        return;
                    }
                    Log("weights resident (incl. encoder).");

                    float[] refAudio = Floats("voice_ref_audio", out int[] ash);
                    float[] latRef = Floats("voice_ref_latents", out int[] lsh);   // [T,32]
                    float[] promptRef = Floats("audio_prompt_ref", out int[] psh); // [T,1024]
                    int ld = lsh[lsh.Length - 1], dim = psh[psh.Length - 1];
                    Log($"ref: {refAudio.Length} samples ({refAudio.Length / (float)PocketTTSConfig.SAMPLE_RATE:F2}s) " +
                        $"-> latents [{lsh[0]},{ld}] -> audio_prompt [{psh[0]},{dim}]");

                    // gate 1: encoder latents (pre speaker_proj) — isolates the SEANet+transformer+downsample port
                    float[] lat = tts.EncodeToLatents(refAudio, out int T);
                    Log($"frames: ours {T} vs ref {lsh[0]}" + (T != lsh[0] ? "  <-- FRAME COUNT MISMATCH" : ""));
                    if (T != lsh[0]) failed = true;
                    failed |= Grade("encoder latents", lat, latRef, int8 ? 0.99f : 0.99f);

                    // gate 2: full audio_prompt (encoder + speaker_proj) — the cloned voice prefix
                    float[] prompt = tts.EncodeToPrompt(refAudio);
                    failed |= Grade("audio_prompt (clone)", prompt, promptRef, int8 ? 0.99f : 0.99f);

                    // gate 3: cache round-trip — CloneVoice writes+reads a .bin; a second call is a hit
                    string cacheProbe = "p8probe";
                    bool ok1 = tts.CloneVoice(refAudio, PocketTTSConfig.SAMPLE_RATE, cacheProbe);
                    bool ok2 = tts.CloneVoice(refAudio, PocketTTSConfig.SAMPLE_RATE, cacheProbe);  // cache hit
                    Log($"cache round-trip: encode {ok1}, hit {ok2}, bound voice '{tts.CurrentVoice}'");
                    if (!(ok1 && ok2)) { failed = true; Log("   <-- FAIL cache round-trip"); }
                }
                catch (Exception ex)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {ex.GetType().Name}: {ex.Message}\n{ex.StackTrace}");
                }
                finally
                {
                    tts?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            static bool Grade(string name, float[] ours, float[] rf, float gate)
            {
                var (mx, mae, corr) = Diff(ours, rf);
                bool bad = corr < gate;
                Log($"## {name}: ours {ours.Length} vs {rf.Length}; maxAbs {mx:F4} MAE {mae:F5} corr {corr:F6}" + (bad ? "  <-- FAIL" : "  PASS"));
                return bad;
            }

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

            static float[] Floats(string name, out int[] sh)
            {
                byte[] all = File.ReadAllBytes(Path.Combine(DUMP, name + ".npy"));
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);
                string shapeStr = header.Substring(header.IndexOf("'shape':", StringComparison.Ordinal) + 8);
                shapeStr = shapeStr.Substring(shapeStr.IndexOf('(') + 1);
                shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
                var dims = new List<int>();
                foreach (string s in shapeStr.Split(',')) if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                sh = dims.ToArray();
                long count = 1; foreach (int dd in sh) count *= dd;
                float[] r = new float[count];
                Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                return r;
            }
        }
    }
}
