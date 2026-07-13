#if UNITY_EDITOR
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // #31 — per-token decode cost breakdown for Qwen3.5 on the LOCAL GPU, edit mode, no play
        // mode and no reference data needed (perf only, greedy over a fixed dummy prompt).
        // Reports clean tok/s (unprofiled) then a serialized per-category share table
        // (Qwen3_5Model.StageProfile Mark() drains the queue per group — shares are the signal).
        public static class QwenDecodeProfileProbe
        {
            const string WEIGHTS = "Assets/Resources/Weights/weights_qwen3.5_0.8B_int8";
            const int PREFILL_LEN = 64, WARMUP = 4, TIMED = 32, PROFILED = 8;

            [MenuItem("DeepUnity/Qwen3.5/Decode Profile (int8)")]
            public static void Run()
            {
                Qwen3_5Model model = null;
                try
                {
                    EditorUtility.DisplayProgressBar("Qwen3.5 decode profile", "Loading weights (blocking)…", 0.1f);
                    var sw = System.Diagnostics.Stopwatch.StartNew();
                    model = new Qwen3_5Model(WEIGHTS, 2048, LLMQuant.INT8, KVQuant.FP16);
                    model.LoadBlockingForProbe();
                    float loadMs = (float)sw.Elapsed.TotalMilliseconds;

                    var ids = new float[PREFILL_LEN];
                    for (int i = 0; i < PREFILL_LEN; i++) ids[i] = 1000 + i;   // fixed dummy prompt

                    EditorUtility.DisplayProgressBar("Qwen3.5 decode profile", "Prefill + warmup…", 0.3f);
                    sw.Restart();
                    model.Forward(Tensor.Constant(ids), useCache: true, lastPosOnly: true);
                    int tok = model.SampleGreedy();
                    float prefillMs = (float)sw.Elapsed.TotalMilliseconds;
                    for (int i = 0; i < WARMUP; i++)
                    {
                        model.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
                        tok = model.SampleGreedy();
                    }

                    EditorUtility.DisplayProgressBar("Qwen3.5 decode profile", "Timed decode…", 0.5f);
                    sw.Restart();
                    var sampleSw = new System.Diagnostics.Stopwatch();
                    for (int i = 0; i < TIMED; i++)
                    {
                        model.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
                        sampleSw.Start();
                        tok = model.SampleGreedy();   // sampler kernels + SYNC token readback
                        sampleSw.Stop();
                    }
                    float decodeMs = (float)sw.Elapsed.TotalMilliseconds;
                    float msPerTok = decodeMs / TIMED;
                    Debug.Log($"[QwenProfile] int8 kvFP16, prefill {PREFILL_LEN} ids {prefillMs:F0} ms | " +
                              $"decode {msPerTok:F1} ms/tok = {1000f / msPerTok:F1} tok/s " +
                              $"(sample+readback {sampleSw.Elapsed.TotalMilliseconds / TIMED:F1} ms/tok of that) | " +
                              $"load {loadMs:F0} ms | GPU {SystemInfo.graphicsDeviceName}");

                    EditorUtility.DisplayProgressBar("Qwen3.5 decode profile", "Profiled decode (serialized)…", 0.7f);
                    Qwen3_5Model.StageProfile = new System.Collections.Generic.Dictionary<string, float>();
                    for (int i = 0; i < PROFILED; i++)
                    {
                        model.Forward(Tensor.Constant((float)tok), useCache: true, lastPosOnly: true);
                        tok = model.SampleGreedy();
                    }
                    var prof = Qwen3_5Model.StageProfile;
                    Qwen3_5Model.StageProfile = null;
                    float total = 0f; foreach (var kv in prof) total += kv.Value;
                    var sb = new StringBuilder();
                    sb.AppendLine($"[QwenProfile] DECODE STAGE SHARES over {PROFILED} tokens (serialized total {total / PROFILED:F1} ms/tok — shares are the signal):");
                    foreach (var kv in System.Linq.Enumerable.OrderByDescending(prof, p => p.Value))
                        sb.AppendLine($"    {kv.Key,-22} {kv.Value / PROFILED,8:F2} ms/tok   {100f * kv.Value / total,5:F1}%");
                    Debug.Log(sb.ToString());
                }
                finally
                {
                    Qwen3_5Model.StageProfile = null;
                    model?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }

            // #31 parity gate: logits of ONE decode step, coalesced vs legacy GEMV kernels, from an
            // IDENTICAL model state (reset + same prefill, whose kernels are unchanged). Lane-parallel
            // reduction reorders float sums, so the gate is tolerance + argmax agreement, not bit-exact.
            [MenuItem("DeepUnity/Qwen3.5/GEMV Parity (coal vs legacy)")]
            public static void RunParity()
            {
                Qwen3_5Model model = null;
                try
                {
                    EditorUtility.DisplayProgressBar("Qwen3.5 GEMV parity", "Loading weights (blocking)…", 0.1f);
                    model = new Qwen3_5Model(WEIGHTS, 2048, LLMQuant.INT8, KVQuant.FP16);
                    model.LoadBlockingForProbe();

                    var ids = new float[32];
                    for (int i = 0; i < 32; i++) ids[i] = 2000 + i * 3;

                    float[] Logits(bool legacy)
                    {
                        Qwen3_5Model.ForceLegacyGemv = legacy;
                        model.ResetCache();
                        model.Forward(Tensor.Constant(ids), useCache: true, lastPosOnly: true);   // prefill: same kernels both runs
                        model.Forward(Tensor.Constant(1234f), useCache: true, lastPosOnly: true); // ONE decode step
                        return model.ReadLogits(1).ToArray();
                    }

                    EditorUtility.DisplayProgressBar("Qwen3.5 GEMV parity", "Legacy pass…", 0.3f);
                    float[] la = Logits(true);
                    EditorUtility.DisplayProgressBar("Qwen3.5 GEMV parity", "Coalesced pass…", 0.6f);
                    float[] lb = Logits(false);

                    double maxAbs = 0, dot = 0, a2 = 0, b2 = 0;
                    int argA = 0, argB = 0;
                    for (int i = 0; i < la.Length; i++)
                    {
                        double d = System.Math.Abs(la[i] - (double)lb[i]);
                        if (d > maxAbs) maxAbs = d;
                        dot += (double)la[i] * lb[i]; a2 += (double)la[i] * la[i]; b2 += (double)lb[i] * lb[i];
                        if (la[i] > la[argA]) argA = i;
                        if (lb[i] > lb[argB]) argB = i;
                    }
                    double corr = dot / System.Math.Max(System.Math.Sqrt(a2 * b2), 1e-30);
                    bool pass = maxAbs < 5e-3 && corr > 0.9999999 && argA == argB;
                    Debug.Log($"[QwenParity] coal vs legacy logits ({la.Length} vocab): " +
                              (pass ? "PASS" : "FAIL") +
                              $" | maxAbs {maxAbs:E2} | corr {corr:F9} | argmax {argA} vs {argB} ({(argA == argB ? "match" : "MISMATCH")})");
                    if (!pass) Debug.LogError("[QwenParity] GEMV parity FAILED — do not ship the coalesced kernels.");
                }
                finally
                {
                    Qwen3_5Model.ForceLegacyGemv = false;
                    model?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }
        }
    }
}
#endif
