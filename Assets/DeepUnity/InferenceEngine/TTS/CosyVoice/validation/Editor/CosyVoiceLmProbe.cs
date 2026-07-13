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
        // A3 parity probe — CosyVoice3LM vs the Python reference dump. Editor-mode
        // synchronous (ClaudeBridge invoke).
        //
        // Grades:
        //   logp step0   log_softmax(speech-head logits) after prefilling
        //                [sos | prompt_text ++ text | task | prompt_speech] vs llm_logp_step0
        //                (corr > 0.999 + argmax match)
        //   decode perf  201 injected reference tokens -> tok/s (real-time floor: 25)
        public static class CosyVoiceLmProbe
        {
            const string DUMP_DIR = "Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/validation/dump";
            const string WEIGHTS_DIR = "Assets/Resources/Weights/weights_cosyvoice3_fp16";
            const string REPORT = "ProbeLogs/cosyvoice_lm_parity.md";
            const string DONE = "ProbeLogs/cosyvoice_lm_parity.done";

            static readonly StringBuilder report = new StringBuilder();

            static void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[CosyVoiceLmParity] " + line);
            }

            static string weightsDir = WEIGHTS_DIR;

            // A6 — same gates against the same fp32 dumps, int8 weights (expect ~lossless)
            [MenuItem("DeepUnity/CosyVoice/A6 LM Parity INT8")]
            public static void RunInt8()
            {
                weightsDir = "Assets/Resources/Weights/weights_cosyvoice3_int8";
                try { Run(); } finally { weightsDir = WEIGHTS_DIR; }
            }

            // A/B bisect for the Phase-6 fused decode kernels (same weights, FastLM off)
            [MenuItem("DeepUnity/CosyVoice/A3 LM Parity LEGACY (FastLM off)")]
            public static void RunLegacyLm()
            {
                bool prev = CosyVoiceLM.FastLM;
                CosyVoiceLM.FastLM = false;
                try { Run(); } finally { CosyVoiceLM.FastLM = prev; }
            }

            [MenuItem("DeepUnity/CosyVoice/A3 LM Parity")]
            public static void Run()
            {
                report.Clear();
                Directory.CreateDirectory("ProbeLogs");
                bool failed = false;
                CosyVoiceWeights weights = null;
                CosyVoiceLM lm = null;
                try
                {
                    Log($"# CosyVoice3 A3 — LM parity — {DateTime.Now:yyyy-MM-dd HH:mm}");

                    int[] promptText = Ints("prompt_text_tokens");    // [16] incl. <|endofprompt|>
                    int[] text = Ints("text_tokens");                 // [50] utterance text
                    int[] promptSpeech = Ints("prompt_speech_tokens");// [87]
                    int[] refSpeech = Ints("speech_tokens");          // [201] RAS-sampled reference
                    float[] refLogp = Floats("llm_logp_step0", out _);// [6761] log_softmax @ step 0
                    int[] textFull = new int[promptText.Length + text.Length];
                    promptText.CopyTo(textFull, 0);
                    text.CopyTo(textFull, promptText.Length);
                    Log($"dump: text {promptText.Length}+{text.Length}, prompt speech {promptSpeech.Length}, ref speech {refSpeech.Length}");

                    var swLoad = System.Diagnostics.Stopwatch.StartNew();
                    Log($"weights: {weightsDir}");
                    weights = new CosyVoiceWeights(weightsDir, beginLoad: false);
                    weights.LoadBlocking("llm/");
                    swLoad.Stop();
                    Log($"llm/* weights resident (blocking load, {swLoad.Elapsed.TotalSeconds:F1}s).");

                    // stage bisection (dump_lm_stages.py refs) — single-chunk prefill for full-seq taps.
                    // Non-bisect uses the production default (64 since A6-max Phase 4) so the
                    // "prefill N tokens: X ms" line reflects what A5/TTFA actually pays; the
                    // burst size cannot change results (per-query causal attention).
                    string stagesDir = Path.Combine(DUMP_DIR, "lm_stages");
                    bool bisect = Directory.Exists(stagesDir);
                    lm = bisect ? new CosyVoiceLM(weights, prefillChunk: 256) : new CosyVoiceLM(weights);
                    var stageResults = new List<string>();
                    if (bisect)
                        lm.DebugTap = (name, buf, count) =>
                        {
                            string p = Path.Combine(stagesDir,
                                (name == "embeds" ? "lm_embeds" : name == "final_norm" ? "lm_hidden_last" : "lm_" + name) + ".npy");
                            if (!File.Exists(p)) return;
                            float[] rf = Floats2(p, out _);
                            float[] ours = new float[count];
                            buf.GetData(ours, 0, 0, count);
                            var (sMax, sMae, sCorr) = Diff(ours, rf);
                            double oa = 0; for (int i = 0; i < count; i++) oa += Math.Abs(ours[i]);
                            stageResults.Add($"   stage {name,-11} maxAbs {sMax:F4}  MAE {sMae:F5}  corr {sCorr:F6}" +
                                             $"  (ours mean|x| {oa / count:F4}; [{ours[0]:F4},{ours[1]:F4},{ours[2]:F4}] vs ref [{rf[0]:F4},{rf[1]:F4},{rf[2]:F4}])" +
                                             (sCorr < 0.995f ? "  <-- DIVERGES" : ""));
                        };
                    int L = lm.BuildPrefillEmbeds(textFull, promptSpeech);
                    IEnumerator pf = lm.PrefillYielding(L);
                    while (pf.MoveNext()) { }
                    Log($"prefill {L} tokens: {lm.PrefillMs:F0} ms");
                    if (stageResults.Count > 0)
                    {
                        Log("## Stage bisection (vs dump/lm_stages):");
                        foreach (string s in stageResults) Log(s);
                    }
                    lm.DebugTap = null;

                    // ---- step-0 logp vs reference
                    float[] logits = lm.ReadLogits();
                    float[] ourLogp = LogSoftmax(logits);
                    var (lMax, lMae, lCorr) = Diff(ourLogp, refLogp);
                    int ourArg = ArgMax(ourLogp), refArg = ArgMax(refLogp);
                    Log($"## logp step0: maxAbs {lMax:F4}  MAE {lMae:F5}  corr {lCorr:F6}");
                    Log($"   argmax: ours {ourArg} vs ref {refArg} — {(ourArg == refArg ? "MATCH" : "MISMATCH")}");
                    if (lCorr < 0.999f || ourArg != refArg) { failed = true; Log("   FAIL (corr < 0.999 or argmax mismatch)"); }

                    // ---- injected decode: perf over the 201 reference tokens
                    var swDec = System.Diagnostics.Stopwatch.StartNew();
                    for (int i = 0; i < refSpeech.Length; i++)
                    {
                        IEnumerator d = lm.DecodeStepYielding(refSpeech[i]);
                        while (d.MoveNext()) { }
                        lm.ReadLogits();   // sync each step like the real sampler loop
                    }
                    swDec.Stop();
                    float tps = refSpeech.Length / (float)swDec.Elapsed.TotalSeconds;
                    Log($"## decode: {refSpeech.Length} tokens in {swDec.Elapsed.TotalSeconds:F1}s = {tps:F1} tok/s " +
                        $"(real-time needs {CosyVoiceConfig.TOKEN_FRAME_RATE}; {tps / CosyVoiceConfig.TOKEN_FRAME_RATE:F2}x RT)");
                    if (tps < CosyVoiceConfig.TOKEN_FRAME_RATE) Log("   WARNING: below real-time (editor-sync path; revisit at A5/A6)");
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    lm?.Dispose();
                    weights?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            static float[] LogSoftmax(float[] logits)
            {
                double max = double.MinValue;
                foreach (float v in logits) if (v > max) max = v;
                double sum = 0;
                foreach (float v in logits) sum += Math.Exp(v - max);
                double lse = max + Math.Log(sum);
                float[] r = new float[logits.Length];
                for (int i = 0; i < logits.Length; i++) r[i] = (float)(logits[i] - lse);
                return r;
            }

            static int ArgMax(float[] a)
            {
                int bi = 0; float bv = a[0];
                for (int i = 1; i < a.Length; i++) if (a[i] > bv) { bv = a[i]; bi = i; }
                return bi;
            }

            // ---------------- npy / diff helpers (self-contained per probe file) ----------------
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

            static float[] Floats2(string fullPath, out int[] shape)
                => (float[])LoadNpy(fullPath, out shape);

            static int[] Ints(string name)
            {
                Array a = LoadNpy(Path.Combine(DUMP_DIR, name + ".npy"), out _);
                if (a is int[] i) return i;
                if (a is long[] l)
                {
                    int[] r = new int[l.Length];
                    for (int j = 0; j < l.Length; j++) r[j] = (int)l[j];
                    return r;
                }
                float[] f = (float[])a;   // some dumps store token ids as float32
                int[] r2 = new int[f.Length];
                for (int j = 0; j < f.Length; j++) r2[j] = (int)Math.Round(f[j]);
                return r2;
            }

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
        }
    }
}
