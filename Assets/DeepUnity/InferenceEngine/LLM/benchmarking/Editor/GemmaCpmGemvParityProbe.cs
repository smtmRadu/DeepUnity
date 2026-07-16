#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // #31 parity + A/B gate for the coalesced GEMV/GEMM port shared by Gemma3-270M and
    // MiniCPM5-1B (both dispatch Gemma3CS). Compares the logits of ONE decode step, coalesced vs
    // legacy kernels, from an IDENTICAL model state. The prefill of each arm runs its matching
    // kernels too (GemmCoal vs legacy batch), so the gate covers BOTH the decode GEMVs and the
    // prefill GEMMs (prefill feeds the decode step through the KV cache). Lane-parallel reduction
    // reorders float sums, so the gate is tolerance + argmax agreement, not bit-exact.
    // Also times WARM decode steps on each arm (one sync per arm) for a same-run speedup read.
    public static class GemmaCpmGemvParityProbe
    {
        const int PREFILL_LEN = 32, TIMED = 32;

        [MenuItem("DeepUnity/Gemma3/GEMV Parity + A-B (coal vs legacy, int8)")]
        public static void RunGemma() => RunGemmaQuant(LLMQuant.INT8);
        [MenuItem("DeepUnity/Gemma3/GEMV Parity + A-B (coal vs legacy, int4)")]
        public static void RunGemmaInt4() => RunGemmaQuant(LLMQuant.INT4);
        [MenuItem("DeepUnity/Gemma3/GEMV Parity + A-B (coal vs legacy, fp16)")]
        public static void RunGemmaFp16() => RunGemmaQuant(LLMQuant.FP16);

        static void RunGemmaQuant(LLMQuant quant)
        {
            string q = quant == LLMQuant.INT8 ? "int8" : quant == LLMQuant.INT4 ? "int4" : "fp16";
            Gemma3Modeling.Gemma3Model model = null;
            try
            {
                EditorUtility.DisplayProgressBar("Gemma3 GEMV parity", "Loading weights (blocking)…", 0.1f);
                model = new Gemma3Modeling.Gemma3Model(
                    $"Assets/Resources/Weights/weights_gemma3_270M_{q}", 2048, quant, KVQuant.FP16);
                model.LoadBlockingForProbe();
                Body($"Gemma3-270M-{q}",
                     legacy => Gemma3Modeling.Gemma3Model.ForceLegacyGemv = legacy,
                     () => model.ResetCache(),
                     (t, lastPos) => model.Forward(t, useCache: true, lastPosOnly: lastPos),
                     () => model.ReadLogits(1).ToArray());
            }
            finally
            {
                Gemma3Modeling.Gemma3Model.ForceLegacyGemv = false;
                model?.Dispose();
                EditorUtility.ClearProgressBar();
            }
        }

        [MenuItem("DeepUnity/MiniCPM5/GEMV Parity + A-B (coal vs legacy, int8)")]
        public static void RunMiniCPM() => RunMiniCPMQuant(LLMQuant.INT8);
        [MenuItem("DeepUnity/MiniCPM5/GEMV Parity + A-B (coal vs legacy, int4)")]
        public static void RunMiniCPMInt4() => RunMiniCPMQuant(LLMQuant.INT4);
        [MenuItem("DeepUnity/MiniCPM5/GEMV Parity + A-B (coal vs legacy, fp16)")]
        public static void RunMiniCPMFp16() => RunMiniCPMQuant(LLMQuant.FP16);

        static void RunMiniCPMQuant(LLMQuant quant)
        {
            string q = quant == LLMQuant.INT8 ? "int8" : quant == LLMQuant.INT4 ? "int4" : "fp16";
            MiniCPM5Modeling.MiniCPM5Model model = null;
            try
            {
                EditorUtility.DisplayProgressBar("MiniCPM5 GEMV parity", "Loading weights (blocking)…", 0.1f);
                model = new MiniCPM5Modeling.MiniCPM5Model(
                    $"Assets/Resources/Weights/weights_minicpm5_1B_{q}", 2048, quant, KVQuant.FP16);
                model.LoadBlockingForProbe();
                Body($"MiniCPM5-1B-{q}",
                     legacy => MiniCPM5Modeling.MiniCPM5Model.ForceLegacyGemv = legacy,
                     () => model.ResetCache(),
                     (t, lastPos) => model.Forward(t, useCache: true, lastPosOnly: lastPos),
                     () => model.ReadLogits(1).ToArray());
            }
            finally
            {
                MiniCPM5Modeling.MiniCPM5Model.ForceLegacyGemv = false;
                model?.Dispose();
                EditorUtility.ClearProgressBar();
            }
        }

        static void Body(string tag, System.Action<bool> setLegacy, System.Action resetCache,
                         System.Action<Tensor, bool> forward, System.Func<float[]> readLogits)
        {
            var ids = new float[PREFILL_LEN];
            for (int i = 0; i < PREFILL_LEN; i++) ids[i] = 2000 + i * 3;   // fixed dummy prompt

            // prefill and decode arms are switchable INDEPENDENTLY: the decode-only comparison
            // (identical legacy prefill -> identical KV cache) isolates the GEMV kernels from
            // prefill-diff propagation through the layers (gemma's sqrt(hidden) embed scale
            // amplifies tiny KV diffs into ~1e-2 logit diffs over 18 layers — not a kernel bug).
            float[] Logits(bool prefillLegacy, bool decodeLegacy)
            {
                setLegacy(prefillLegacy);
                resetCache();
                forward(Tensor.Constant(ids), true);    // prefill (GemmCoal vs legacy batch)
                setLegacy(decodeLegacy);
                forward(Tensor.Constant(1234f), true);  // ONE decode step (1VecCoal vs legacy 1Vec)
                return readLogits();
            }

            // decode wall-time over TIMED warm steps, ONE sync at the end (readback overhead paid once)
            double DecodeMs(bool legacy)
            {
                setLegacy(legacy);
                resetCache();
                forward(Tensor.Constant(ids), true);
                forward(Tensor.Constant(1000f), true); readLogits();   // warm + sync
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < TIMED; i++) forward(Tensor.Constant(1000f + i), true);
                readLogits();                                          // sync
                return sw.Elapsed.TotalMilliseconds / TIMED;
            }

            (double maxAbs, double rel, double corr, bool argMatch, int argA, int argB) Compare(float[] la, float[] lb)
            {
                double maxAbs = 0, dot = 0, a2 = 0, b2 = 0, maxL = 0;
                int argA = 0, argB = 0;
                for (int i = 0; i < la.Length; i++)
                {
                    double d = System.Math.Abs(la[i] - (double)lb[i]);
                    if (d > maxAbs) maxAbs = d;
                    double m = System.Math.Abs(la[i]);
                    if (m > maxL) maxL = m;
                    dot += (double)la[i] * lb[i]; a2 += (double)la[i] * la[i]; b2 += (double)lb[i] * lb[i];
                    if (la[i] > la[argA]) argA = i;
                    if (lb[i] > lb[argB]) argB = i;
                }
                double corr = dot / System.Math.Max(System.Math.Sqrt(a2 * b2), 1e-30);
                return (maxAbs, maxAbs / System.Math.Max(maxL, 1e-30), corr, argA == argB, argA, argB);
            }

            EditorUtility.DisplayProgressBar($"{tag} GEMV parity", "Reference pass (all legacy)…", 0.25f);
            float[] lRef  = Logits(prefillLegacy: true,  decodeLegacy: true);
            EditorUtility.DisplayProgressBar($"{tag} GEMV parity", "Decode-only coal pass…", 0.4f);
            float[] lDec  = Logits(prefillLegacy: true,  decodeLegacy: false);
            EditorUtility.DisplayProgressBar($"{tag} GEMV parity", "Full coal pass…", 0.55f);
            float[] lFull = Logits(prefillLegacy: false, decodeLegacy: false);

            var dec  = Compare(lRef, lDec);    // GEMV kernels only, identical KV state -> tight gate
            var full = Compare(lRef, lFull);   // + GEMM prefill; propagation-amplified -> relative gate

            bool pass = dec.maxAbs < 5e-3 && dec.corr > 0.9999999 && dec.argMatch
                     && full.rel < 2e-3 && full.corr > 0.999999 && full.argMatch;

            EditorUtility.DisplayProgressBar($"{tag} GEMV parity", "Timing legacy…", 0.7f);
            double msLegacy = DecodeMs(true);
            EditorUtility.DisplayProgressBar($"{tag} GEMV parity", "Timing coalesced…", 0.9f);
            double msCoal = DecodeMs(false);

            Debug.Log($"[{tag}Parity] coal vs legacy logits ({lRef.Length} vocab): " +
                      (pass ? "PASS" : "FAIL") +
                      $" | decode-only maxAbs {dec.maxAbs:E2} corr {dec.corr:F9} argmax {(dec.argMatch ? "match" : $"MISMATCH {dec.argA} vs {dec.argB}")}" +
                      $" | full-path maxAbs {full.maxAbs:E2} (rel {full.rel:E2}) corr {full.corr:F9} argmax {(full.argMatch ? "match" : $"MISMATCH {full.argA} vs {full.argB}")}" +
                      $" | decode legacy {msLegacy:F1} ms/tok vs coal {msCoal:F1} ms/tok = {msLegacy / System.Math.Max(msCoal, 1e-9):F2}x");
            if (!pass) Debug.LogError($"[{tag}Parity] GEMV parity FAILED — do not ship the coalesced kernels.");
        }
    }
}
#endif
