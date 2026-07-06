using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // Play-mode parity probe: compares the DeepUnity Chatterbox-Turbo port against the Python
        // reference dump produced by validation/dump_reference.py (same text, GREEDY T3 decode,
        // injected noise for S3Gen). Run via the ChatterboxParityBatchRunner (Unity closed) or by
        // dropping this component into any scene and pressing play.
        //
        // Stages:
        //   A  tokenizer      — GPT2 BPE ids vs text_tokens.npy (exact match expected)
        //   B  T3 greedy      — step-0 logits diff + greedy speech-token prefix match
        //   C  S3Gen+vocoder  — injected z/NSF noise; mu / est_out / mel / wav tensor diffs
        //
        // fp16 weights vs the fp32 reference means small numeric drift is EXPECTED; what matters:
        // tokenizer exact, T3 top-1 stable for a long prefix, S3Gen correlation ~0.99+.
        public class ChatterboxParityProbe : MonoBehaviour
        {
            public const string DUMP_TEXT = "Hello world! This is a test of the DeepUnity port.";
            public string dumpDir = "Assets/DeepUnity/TTS/validation/dump";
            public string weightsDir = "Assets/Resources/DeepUnity/TTS/Chatterbox/weights_chatterbox_turbo_fp16";
            public string reportPath = "ProbeLogs/chatterbox_parity_report.md";
            public string doneMarker = "ProbeLogs/chatterbox_parity.done";

            readonly StringBuilder report = new StringBuilder();
            bool failed;

            void Start() => StartCoroutine(Run());

            // ---------------- minimal .npy loader (v1.0/2.0, little-endian, C-order) --------------
            static Array LoadNpy(string path, out int[] shape)
            {
                byte[] all = File.ReadAllBytes(path);
                if (all[0] != 0x93) throw new Exception($"not npy: {path}");
                int major = all[6];
                int headerLen = major >= 2 ? BitConverter.ToInt32(all, 8) : BitConverter.ToUInt16(all, 8);
                int dataStart = (major >= 2 ? 12 : 10) + headerLen;
                string header = Encoding.ASCII.GetString(all, major >= 2 ? 12 : 10, headerLen);

                string descr = ExtractField(header, "descr");
                string shapeStr = header.Substring(header.IndexOf("'shape':", StringComparison.Ordinal) + 8);
                shapeStr = shapeStr.Substring(shapeStr.IndexOf('(') + 1);
                shapeStr = shapeStr.Substring(0, shapeStr.IndexOf(')'));
                var dims = new List<int>();
                foreach (string s in shapeStr.Split(','))
                    if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                shape = dims.ToArray();
                long count = 1; foreach (int d in shape) count *= d;

                if (descr.Contains("f4"))
                {
                    float[] r = new float[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    return r;
                }
                if (descr.Contains("i8"))
                {
                    long[] r = new long[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8);
                    return r;
                }
                if (descr.Contains("i4"))
                {
                    int[] r = new int[count];
                    Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4);
                    return r;
                }
                throw new Exception($"unsupported npy dtype {descr} in {path}");
            }

            static string ExtractField(string header, string key)
            {
                int i = header.IndexOf("'" + key + "':", StringComparison.Ordinal);
                int q1 = header.IndexOf('\'', i + key.Length + 3);
                int q2 = header.IndexOf('\'', q1 + 1);
                return header.Substring(q1 + 1, q2 - q1 - 1);
            }

            float[] Floats(string name, out int[] shape) => (float[])LoadNpy(Path.Combine(dumpDir, name + ".npy"), out shape);
            int[] Ints(string name)
            {
                Array a = LoadNpy(Path.Combine(dumpDir, name + ".npy"), out _);
                if (a is int[] i) return i;
                long[] l = (long[])a;
                int[] r = new int[l.Length];
                for (int j = 0; j < l.Length; j++) r[j] = (int)l[j];
                return r;
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

            static float[] TransposeCT(float[] src, int C, int T)   // [C,T] -> [T,C]
            {
                float[] r = new float[src.Length];
                for (int c = 0; c < C; c++)
                    for (int t = 0; t < T; t++)
                        r[t * C + c] = src[c * T + t];
                return r;
            }

            void Log(string line)
            {
                report.AppendLine(line);
                Debug.Log("[ChatterboxParity] " + line);
            }

            IEnumerator Run()
            {
                Directory.CreateDirectory("ProbeLogs");
                Log($"# Chatterbox-Turbo parity report — {DateTime.Now:yyyy-MM-dd HH:mm}");
                Log("");

                if (!Directory.Exists(dumpDir) || !File.Exists(Path.Combine(dumpDir, "text_tokens.npy")))
                {
                    Log($"**DUMP MISSING** at {dumpDir} — run validation/dump_reference.py first.");
                    Finish(false);
                    yield break;
                }

                Log($"weights: {weightsDir}");
                var weights = new ChatterboxWeights(weightsDir);
                var tokenizer = new ChatterboxTokenizer("Assets/DeepUnity/TTS/Chatterbox/ChatterboxTokenizer");
                while (!weights.IsReady) yield return null;
                Log("weights streamed to GPU.");

                // ---------------- A: tokenizer ----------------
                int[] refText = Ints("text_tokens");
                int[] ourText = tokenizer.Encode(ChatterboxTokenizer.PuncNorm(DUMP_TEXT));
                bool tokOk = ourText.Length == refText.Length;
                if (tokOk)
                    for (int i = 0; i < ourText.Length; i++) tokOk &= ourText[i] == refText[i];
                Log($"## A. Tokenizer: {(tokOk ? "EXACT MATCH" : "MISMATCH")} " +
                    $"(ours {ourText.Length} vs ref {refText.Length} tokens)");
                if (!tokOk)
                {
                    failed = true;
                    Log($"   ours: [{string.Join(",", ourText)}]");
                    Log($"   ref:  [{string.Join(",", refText)}]");
                }

                // ---------------- B: T3 greedy ----------------
                var t3 = new T3Model(weights);
                float[] spk = weights.ReadFloats("conds/t3_speaker_emb");
                t3.SetSpeakerEmbedding(spk);
                int[] prompt = weights.ReadInts("conds/t3_prompt_tokens");

                t3.ResetCache();
                var swPrefill = System.Diagnostics.Stopwatch.StartNew();
                int L = t3.BuildPrefillEmbeds(prompt, refText);   // ref tokens: isolate T3 from tokenizer
                var pf = t3.PrefillYielding(L);
                while (pf.MoveNext()) yield return pf.Current;
                swPrefill.Stop();

                float[] refLogits = Floats("t3_logits_step0", out _);
                float[] ourLogits = t3.ReadLogits();
                var (lMax, lMae, lCorr) = Diff(ourLogits, refLogits);
                int refArg = ArgMax(refLogits), ourArg = ArgMax(ourLogits);
                Log($"## B. T3 step-0 logits: maxAbs {lMax:F4}  MAE {lMae:F5}  corr {lCorr:F6}");
                Log($"   argmax: ours {ourArg} vs ref {refArg} — {(ourArg == refArg ? "MATCH" : "MISMATCH")}");
                if (lCorr < 0.999f || ourArg != refArg) failed = true;

                int[] refSpeech = Ints("speech_tokens");
                var ourSpeech = new List<int>();
                int[] sampled = new int[1];
                var swDecode = System.Diagnostics.Stopwatch.StartNew();
                var s0 = t3.SampleYielding(null, 0, 0f, 0, 1f, 1f, sampled);
                while (s0.MoveNext()) yield return s0.Current;
                int tok = sampled[0];
                for (int step = 0; step < refSpeech.Length && tok != ChatterboxConfig.STOP_SPEECH_TOKEN; step++)
                {
                    ourSpeech.Add(tok);
                    var d = t3.DecodeStepYielding(tok);
                    while (d.MoveNext()) yield return d.Current;
                    var sm = t3.SampleYielding(null, 0, 0f, 0, 1f, 1f, sampled);
                    while (sm.MoveNext()) yield return sm.Current;
                    tok = sampled[0];
                }
                swDecode.Stop();
                float tokPerSec = ourSpeech.Count / Mathf.Max((float)swDecode.Elapsed.TotalSeconds, 1e-3f);
                Log($"   [perf] T3 prefill {L} tok: {swPrefill.Elapsed.TotalMilliseconds:F0} ms | " +
                    $"decode: {ourSpeech.Count} tok in {swDecode.Elapsed.TotalSeconds:F1}s = {tokPerSec:F1} tok/s " +
                    $"(real-time needs 25; {tokPerSec / 25f:F2}x RT)");
                int prefix = 0;
                while (prefix < ourSpeech.Count && prefix < refSpeech.Length && ourSpeech[prefix] == refSpeech[prefix])
                    prefix++;
                Log($"   greedy speech tokens: ours {ourSpeech.Count}, ref {refSpeech.Length}, " +
                    $"matching prefix {prefix} ({100f * prefix / Mathf.Max(refSpeech.Length, 1):0.0}%)");
                if (prefix < Mathf.Min(refSpeech.Length, 25)) failed = true;   // fp16 drift diverges eventually; early divergence = bug

                // ---------------- C: S3Gen with injected noise ----------------
                var s3gen = new S3GenModel(weights);
                int[] flowGen = Array.FindAll(refSpeech, v => v < ChatterboxConfig.FLOW_VOCAB);

                float[] z = Floats("z_noise", out int[] zs);       // [1, 80, Tmel] channel-major
                int Tmel = zs[2];
                s3gen.InjectFlowNoise = TransposeCT(z, 80, Tmel);
                float[] nsfNoise = Floats("nsf_noise", out int[] ns);  // [1, 9, S]
                s3gen.InjectNsfNoise = TransposeCT(nsfNoise, 9, ns[2]);
                s3gen.InjectNsfPhases = Floats("nsf_phases", out _);

                float[] wav = null;
                var syn = s3gen.SynthesizeYielding(flowGen, w => wav = w);
                while (syn.MoveNext()) yield return syn.Current;

                // mu: python [1, 2T, 80] time-major == ours [T,80]
                float[] refMu = Floats("mu", out int[] ms);
                float[] ourMu = new float[ms[1] * ms[2]];
                s3gen.DebugMu.GetData(ourMu, 0, 0, ourMu.Length);
                var (mMax, mMae, mCorr) = Diff(ourMu, refMu);
                Log($"## C. S3Gen mu: maxAbs {mMax:F4}  MAE {mMae:F5}  corr {mCorr:F6}");
                if (mCorr < 0.995f) failed = true;

                // est_out_1 (last dxdt): python [1, 80, Tmel] channel-major -> transpose
                float[] refDx = TransposeCT(Floats("est_out_1", out _), 80, Tmel);
                float[] ourDx = new float[Tmel * 80];
                s3gen.DebugDxdt.GetData(ourDx, 0, 0, ourDx.Length);
                var (dMax, dMae, dCorr) = Diff(ourDx, refDx);
                Log($"   estimator dxdt(step1): maxAbs {dMax:F4}  MAE {dMae:F5}  corr {dCorr:F6}");
                if (dCorr < 0.99f) failed = true;

                // mel: python [1, 80, Tg] -> transpose
                float[] refMelRaw = Floats("mel", out int[] mels);
                int Tg = mels[2];
                float[] refMel = TransposeCT(refMelRaw, 80, Tg);
                float[] ourMel = new float[Tg * 80];
                s3gen.DebugMel.GetData(ourMel, 0, 0, ourMel.Length);
                var (meMax, meMae, meCorr) = Diff(ourMel, refMel);
                Log($"   mel: maxAbs {meMax:F4}  MAE {meMae:F5}  corr {meCorr:F6}");
                if (meCorr < 0.99f) failed = true;

                // wav
                float[] refWav = Floats("wav", out _);
                if (wav != null)
                {
                    var (wMax, wMae, wCorr) = Diff(wav, refWav);
                    Log($"   wav: ours {wav.Length} vs ref {refWav.Length} samples; " +
                        $"maxAbs {wMax:F4}  MAE {wMae:F5}  corr {wCorr:F6}");
                    if (wCorr < 0.95f) failed = true;

                    // save the Unity-generated audio for listening
                    SaveWav("ProbeLogs/chatterbox_unity.wav", wav, ChatterboxConfig.SAMPLE_RATE);
                    Log("   Unity audio written to ProbeLogs/chatterbox_unity.wav");

                    float audioSec = wav.Length / (float)ChatterboxConfig.SAMPLE_RATE;
                    float t3Sec = (float)(swPrefill.Elapsed.TotalSeconds + swDecode.Elapsed.TotalSeconds);
                    float totalSec = t3Sec + s3gen.EndToEndMs / 1000f;
                    Log($"   [perf] S3Gen: encoder {s3gen.EncoderMs:F0} ms | meanflow(2 steps) {s3gen.EstimatorMs:F0} ms | " +
                        $"vocoder {s3gen.VocoderMs:F0} ms | readback {s3gen.ReadbackMs:F0} ms | total {s3gen.EndToEndMs:F0} ms");
                    Log($"   [perf] END-TO-END: {totalSec:F1}s for {audioSec:F1}s of audio -> RTF {totalSec / audioSec:F2} " +
                        $"(T3 {t3Sec:F1}s + S3Gen {s3gen.EndToEndMs / 1000f:F1}s; <1.0 = faster than real-time)");
                }
                else { Log("   wav: SYNTHESIS FAILED"); failed = true; }

                t3.Dispose(); s3gen.Dispose(); weights.Dispose();
                Finish(!failed);
            }

            static int ArgMax(float[] a)
            {
                int bi = 0; float bv = a[0];
                for (int i = 1; i < a.Length; i++) if (a[i] > bv) { bv = a[i]; bi = i; }
                return bi;
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

            void Finish(bool ok)
            {
                Log("");
                Log(ok ? "## RESULT: PASS" : "## RESULT: FAIL (see stages above)");
                File.WriteAllText(reportPath, report.ToString());
                File.WriteAllText(doneMarker, ok ? "PASS" : "FAIL");
#if UNITY_EDITOR
                UnityEditor.EditorApplication.isPlaying = false;
#endif
            }
        }
    }
}
