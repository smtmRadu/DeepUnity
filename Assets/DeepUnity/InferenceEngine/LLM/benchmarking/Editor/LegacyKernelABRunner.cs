#if UNITY_EDITOR
using UnityEditor;

namespace DeepUnity
{
    // #30/#31 A/B on ANY machine: same probes, legacy kernels forced — run right after the
    // coalesced/tiled variants to measure the local GPU's real win (the 1650 numbers in the
    // commit log don't transfer; e.g. the 4060 was dispatch-bound, not bandwidth-bound).
    public static class LegacyKernelABRunner
    {
        [MenuItem("DeepUnity/Qwen3.5/Decode Profile (int8, LEGACY GEMV)")]
        public static void QwenLegacy()
        {
            Qwen3_5Modeling.Qwen3_5Model.ForceLegacyGemv = true;
            try { Qwen3_5Modeling.QwenDecodeProfileProbe.Run(); }
            finally { Qwen3_5Modeling.Qwen3_5Model.ForceLegacyGemv = false; }
        }

        [MenuItem("DeepUnity/PocketTTS/RTF Benchmark (int8, LEGACY kernels)")]
        public static void PocketLegacy()
        {
            PocketTTSModeling.PocketTTSMimi.ForceLegacyKernels = true;
            try { PocketTTSModeling.PocketTTSRtfProbe.RunInt8(); }
            finally { PocketTTSModeling.PocketTTSMimi.ForceLegacyKernels = false; }
        }
    }
}
#endif
