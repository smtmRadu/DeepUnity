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
        // P2/P3 parity probe — FlowLM front half vs the python dump (validation/dump_reference.py).
        // EDITOR-MODE + SYNCHRONOUS. Three isolated gates:
        //   text_embeddings  embed lookup            corr>0.99
        //   xformer_out_f0   6L causal transformer   corr>0.99   (input = [bos_before_voice ; voice ; text ; bos_latent])
        //   flow_latent_f0   SimpleMLPAdaLN velocity corr>0.99   (fed dumped c/s/t/x for isolation)
        public static class PocketTTSFlowLMProbe
        {
            const string DUMP = "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/dump";
            const string WEIGHTS_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            const string WEIGHTS_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
            const string REPORT = "ProbeLogs/pockettts_flowlm_parity.md";
            const string DONE = "ProbeLogs/pockettts_flowlm_parity.done";

            static string WEIGHTS = WEIGHTS_FP16;
            static readonly StringBuilder report = new StringBuilder();
            static void Log(string s) { report.AppendLine(s); Debug.Log("[PocketFlowLM] " + s); }

            [MenuItem("DeepUnity/PocketTTS/P2 FlowLM Parity")]
            public static void Run() { WEIGHTS = WEIGHTS_FP16; RunInner(); }

            [MenuItem("DeepUnity/PocketTTS/P2 FlowLM Parity (int8)")]
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
                PocketTTSFlowLM flm = null;
                try
                {
                    Log($"# pocket-tts P2/P3 — FlowLM parity — {DateTime.Now:yyyy-MM-dd HH:mm}");
                    weights = new PocketTTSWeights(WEIGHTS, beginLoad: false);
                    weights.LoadBlocking("flow_lm/");
                    weights.LoadBlocking("voices/");
                    Log("flow_lm/* + voices/* weights resident.");
                    flm = new PocketTTSFlowLM(weights);

                    // ---- gate 1: embed lookup ----
                    int[] ids = Ints("text_ids", out int[] idsh);      // [1,S]
                    int S = ids.Length;
                    float[] emb = flm.EmbedLookup(ids);                // [S*1024]
                    float[] embRef = Floats("text_embeddings", out _); // [1,S,1024]
                    failed |= Grade("text_embeddings", emb, embRef, 0.99f);

                    // ---- gate 2: transformer ----
                    // [bos_before_voice(1) ; voice_prompt(125) ; text_emb(S) ; input_linear(bos_emb)(1)]
                    // UNCERTAIN: whether bos_before_voice is in the predefined-voice context. If
                    // xformer_out is LOW, drop it (voice-only prefix) — one-line toggle.
                    float[] bbv = weights.ReadFloats("flow_lm.bos_before_voice");   // [1024] (top-level param → dot leaf)
                    float[] voice = weights.ReadFloats("voices/jean/audio_prompt"); // [125*1024]
                    float[] bosLat = flm.BosLatentEmbedding();                      // [1024]
                    int dim = PocketTTSConfig.DIM;
                    int voiceFrames = voice.Length / dim;
                    int L = 1 + voiceFrames + S + 1;
                    float[] seq = new float[L * dim];
                    int off = 0;
                    Array.Copy(bbv, 0, seq, off, dim); off += dim;
                    Array.Copy(voice, 0, seq, off, voice.Length); off += voice.Length;
                    Array.Copy(emb, 0, seq, off, emb.Length); off += emb.Length;
                    Array.Copy(bosLat, 0, seq, off, dim);
                    // construction gate: does our assembled sequence match the dump BEFORE the transformer?
                    // seq mismatch => construction bug (voice/bbv/bos_latent); seq match + xformer_out fail => math bug.
                    float[] seqRef = Floats("xformer_in", out int[] xish);   // [1,L,1024]
                    failed |= Grade("xformer_in", seq, seqRef, 0.999f);
                    var tfOut = flm.RunTransformer(seq, L);
                    float[] all = new float[L * dim];
                    tfOut.GetData(all, 0, 0, L * dim);
                    float[] lastRow = new float[dim];
                    Array.Copy(all, (L - 1) * dim, lastRow, 0, dim);
                    float[] xfRef = Floats("xformer_out_f0", out _);   // [1,1,1024]
                    failed |= Grade("xformer_out_f0", lastRow, xfRef, 0.99f);

                    // ---- gate 3: flow head (isolated: fed dumped c/s/t/x) ----
                    float[] c = Floats("flow_c_f0", out _);            // [1,1024]
                    float[] xNoise = Floats("flow_x_f0", out _);       // [1,32]
                    float[] sArr = Floats("flow_s_f0", out _);         // [1,1]
                    float[] tArr = Floats("flow_t_f0", out _);         // [1,1]
                    // intermediate gates (in order) to localize the flow-head divergence
                    flm.FlowTap = (nm, vals) =>
                    {
                        string refN = nm == "flow_final" ? "flow_latent_f0" : nm;
                        string pp = Path.Combine(DUMP, refN + ".npy");
                        if (!File.Exists(pp)) { Log($"   (no dump for {refN})"); return; }
                        float[] rf = Floats(refN, out _);
                        failed |= Grade(nm, vals, rf, nm == "flow_final" ? 0.99f : 0.99f);
                    };
                    float[] vel = flm.FlowHead(c, xNoise, sArr[0], tArr[0]);   // [32] velocity; taps fire in order
                }
                catch (Exception e)
                {
                    failed = true;
                    Log($"**EXCEPTION**: {e.GetType().Name}: {e.Message}\n{e.StackTrace}");
                }
                finally
                {
                    flm?.Dispose();
                    weights?.Dispose();
                    Log("");
                    Log(failed ? "## RESULT: FAIL" : "## RESULT: PASS");
                    File.WriteAllText(REPORT, report.ToString());
                    File.WriteAllText(DONE, failed ? "FAIL" : "PASS");
                }
            }

            static bool Grade(string name, float[] ours, float[] rf, float gate)
            {
                var (mx, mae, corr) = Diff(ours, rf);
                bool bad = corr < gate || ours.Length != rf.Length;
                Log($"## {name}: ours {ours.Length} vs ref {rf.Length}; maxAbs {mx:F4} MAE {mae:F5} corr {corr:F6}" + (bad ? "  <-- FAIL" : ""));
                return bad;
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
                foreach (string s in shapeStr.Split(',')) if (!string.IsNullOrWhiteSpace(s)) dims.Add(int.Parse(s.Trim()));
                if (dims.Count == 0) dims.Add(1);
                shape = dims.ToArray();
                long count = 1; foreach (int dd in shape) count *= dd;
                if (header.Contains("f4")) { float[] r = new float[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4); return r; }
                if (header.Contains("i8")) { long[] r = new long[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 8); return r; }
                if (header.Contains("i4")) { int[] r = new int[count]; Buffer.BlockCopy(all, dataStart, r, 0, (int)count * 4); return r; }
                throw new Exception($"unsupported npy dtype: {header}");
            }
            static float[] Floats(string name, out int[] sh) => (float[])LoadNpy(Path.Combine(DUMP, name + ".npy"), out sh);
            static int[] Ints(string name, out int[] sh)
            {
                Array a = LoadNpy(Path.Combine(DUMP, name + ".npy"), out sh);
                if (a is int[] ia) return ia;
                if (a is long[] la) { int[] r = new int[la.Length]; for (int i = 0; i < la.Length; i++) r[i] = (int)la[i]; return r; }
                // dump_reference.save() casts torch tensors to float32, so int64 token ids arrive as f4
                // (exact for ids < 2^24). Round-to-nearest back to int.
                if (a is float[] fa) { int[] r = new int[fa.Length]; for (int i = 0; i < fa.Length; i++) r[i] = (int)Math.Round(fa[i]); return r; }
                throw new Exception($"Ints: unexpected npy element type {a.GetType()} for {name}");
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
        }
    }
}
