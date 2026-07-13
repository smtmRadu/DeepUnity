#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // #30 kernel-parity probe — tiled vs legacy Mimi decode kernels, NO reference dumps
        // (the .npy dumps live on the main dev box only). Feeds DETERMINISTIC pseudo-random
        // latents through the full decode twice — once with PocketTTSMimi.ForceLegacyKernels —
        // and gates on bit-exactness: every #30 rewrite preserves the accumulation order
        // (bias first, k-outer/ic-inner, j-ascending attention), so the only tolerated
        // difference is +-0.0 sign, i.e. maxAbs must be EXACTLY 0. Also reports the speedup.
        public static class PocketTTSKernelParityProbe
        {
            [MenuItem("DeepUnity/PocketTTS/Kernel Parity (tiled vs legacy)")]
            public static void Run()
            {
                PocketTTSWeights weights = null;
                PocketTTSMimi mimi = null;
                try
                {
                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Loading mimi weights…", 0.1f);
                    weights = new PocketTTSWeights(PocketTTSConfig.WEIGHTS_DIR_FP16, beginLoad: false);
                    weights.LoadBlocking("mimi/");
                    mimi = new PocketTTSMimi(weights);

                    const int T = 41;                       // one streaming window (CTX 40 + 1): odd, exercises every stage shape
                    var latents = new float[T * PocketTTSConfig.LDIM];
                    uint rng = 0x9E3779B9u;                 // xorshift32 — machine-independent input
                    for (int i = 0; i < latents.Length; i++)
                    {
                        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                        latents[i] = (rng / 4294967295f) * 4f - 2f;
                    }

                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Legacy decode…", 0.3f);
                    PocketTTSMimi.ForceLegacyKernels = true;
                    mimi.Decode(latents, T);                              // warmup: kernel compiles
                    float[] wavLegacy = mimi.Decode(latents, T);
                    float legacyMs = mimi.DecodeMs;

                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Tiled decode…", 0.6f);
                    PocketTTSMimi.ForceLegacyKernels = false;
                    mimi.Decode(latents, T);                              // warmup
                    float[] wavTiled = mimi.Decode(latents, T);
                    float tiledMs = mimi.DecodeMs;

                    double maxAbs = 0, sumSq = 0, dotAB = 0, sumA2 = 0, sumB2 = 0;
                    int worstIdx = -1;
                    for (int i = 0; i < wavLegacy.Length; i++)
                    {
                        double d = System.Math.Abs(wavLegacy[i] - (double)wavTiled[i]);
                        if (d > maxAbs) { maxAbs = d; worstIdx = i; }
                        sumSq += d * d;
                        dotAB += (double)wavLegacy[i] * wavTiled[i];
                        sumA2 += (double)wavLegacy[i] * wavLegacy[i];
                        sumB2 += (double)wavTiled[i] * wavTiled[i];
                    }
                    double corr = dotAB / System.Math.Max(System.Math.Sqrt(sumA2 * sumB2), 1e-30);
                    bool pass = maxAbs == 0.0;
                    string verdict = pass ? "PASS (bit-exact)"
                                   : maxAbs < 1e-5 ? $"PASS-ish (maxAbs {maxAbs:E2} @ {worstIdx} — expected 0, investigate)"
                                   : $"FAIL (maxAbs {maxAbs:E2} @ sample {worstIdx})";
                    Debug.Log($"[PocketParity] tiled vs legacy, T={T} ({wavLegacy.Length} samples): {verdict} | " +
                              $"corr {corr:F9} | legacy {legacyMs:F0} ms -> tiled {tiledMs:F0} ms ({legacyMs / Mathf.Max(tiledMs, 0.01f):F2}x) | " +
                              $"GPU {SystemInfo.graphicsDeviceName}");
                    if (!pass && maxAbs >= 1e-5) Debug.LogError("[PocketParity] kernel parity FAILED — do not ship the tiled kernels.");

                    // ---- gate 2 (#30): tail-restricted window decode vs full decode of the SAME
                    // window. The kept tail must be BIT-exact; the context region is garbage by
                    // contract and is not compared. Mirrors the streaming flush (CTX 40 + chunk 12).
                    EditorUtility.DisplayProgressBar("pocket-tts kernel parity", "Tail-restricted decode…", 0.8f);
                    const int T2 = 52, TAIL = 12;
                    var lat2 = new float[T2 * PocketTTSConfig.LDIM];
                    rng = 0x2545F491u;
                    for (int i = 0; i < lat2.Length; i++)
                    {
                        rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5;
                        lat2[i] = (rng / 4294967295f) * 4f - 2f;
                    }
                    float[] wavFull = mimi.Decode(lat2, T2);
                    float fullMs = mimi.DecodeMs;
                    float[] wavTail = mimi.Decode(lat2, T2, tailLatents: TAIL);
                    float tailMs = mimi.DecodeMs;
                    int tailN = TAIL * PocketTTSConfig.SAMPLES_PER_LATENT;
                    int off = wavFull.Length - tailN;
                    double maxAbs2 = 0; int worst2 = -1;
                    for (int i = 0; i < tailN; i++)
                    {
                        double d = System.Math.Abs(wavFull[off + i] - (double)wavTail[off + i]);
                        if (d > maxAbs2) { maxAbs2 = d; worst2 = i; }
                    }
                    bool pass2 = maxAbs2 == 0.0;
                    Debug.Log($"[PocketParity] tail-restricted (T={T2}, tail={TAIL}): " +
                              (pass2 ? "PASS (bit-exact tail)" : $"FAIL (maxAbs {maxAbs2:E2} @ tail sample {worst2})") +
                              $" | full {fullMs:F0} ms -> tail {tailMs:F0} ms ({fullMs / Mathf.Max(tailMs, 0.01f):F2}x)");
                    if (!pass2) Debug.LogError("[PocketParity] tail-restriction parity FAILED — do not ship.");
                }
                finally
                {
                    PocketTTSMimi.ForceLegacyKernels = false;
                    mimi?.Dispose();
                    weights?.Dispose();
                    EditorUtility.ClearProgressBar();
                }
            }
        }
    }
}
#endif
