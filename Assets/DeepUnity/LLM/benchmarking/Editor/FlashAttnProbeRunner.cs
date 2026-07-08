#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Bridge/batch entry points for the FlashAttention A/B probe:
    //   bridge: { "action": "invoke", "method": "DeepUnity.FlashAttnProbeRunner.RunGemma" }
    //   bridge: { "action": "invoke", "method": "DeepUnity.FlashAttnProbeRunner.RunQwen" }
    // Builds a temp scene with the probe and enters play mode; progress streams to
    // ClaudeBridge/flash_probe_status.txt, the report to ProbeLogs/flash_attn_<model>_<ts>/.
    public static class FlashAttnProbeRunner
    {
        const string TempScenePath = "Assets/__flash_attn_probe_tmp.unity";

        public static string RunGemma() => Run(FlashAttnProbe.LMKind.Gemma3);
        public static string RunQwen() => Run(FlashAttnProbe.LMKind.Qwen3_5);

        // Quantization-vs-FP16 probes (QuantProbe = Qwen3.5, GemmaQuantProbe = Gemma3).
        public static string RunQwenInt8() => RunQuant(LLMQuant.INT8, Qwen3_5Size.B0_8);
        public static string RunQwenInt4() => RunQuant(LLMQuant.INT4, Qwen3_5Size.B0_8);
        public static string RunQwen2BInt8() => RunQuant(LLMQuant.INT8, Qwen3_5Size.B2);
        public static string RunQwen2BInt4() => RunQuant(LLMQuant.INT4, Qwen3_5Size.B2);
        public static string RunGemmaInt8() => RunGemmaQuant(LLMQuant.INT8);
        public static string RunGemmaInt4() => RunGemmaQuant(LLMQuant.INT4);
        public static string RunMiniCPMInt8() => RunMiniCPMQuant(LLMQuant.INT8);
        public static string RunMiniCPMInt4() => RunMiniCPMQuant(LLMQuant.INT4);

        static string RunQuant(LLMQuant quant, Qwen3_5Size size)
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string tag = size == Qwen3_5Size.B2 ? "qwen2b" : "qwen";
            string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"{tag}_{quant}_{runId}".ToLowerInvariant());
            Directory.CreateDirectory(dir);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            new GameObject("Camera").AddComponent<Camera>();
            var probe = new GameObject("QuantProbe").AddComponent<QuantProbe>();
            probe.quant = quant;
            probe.size = size;
            probe.reportDirectory = dir;
            EditorSceneManager.SaveScene(scene, TempScenePath);

            EditorApplication.isPlaying = true;
            return $"{tag} {quant} probe scene built, entering play mode; report dir: " + dir;
        }

        static string RunGemmaQuant(LLMQuant quant)
        {
            try
            {
                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"gemma_{quant}_{runId}".ToLowerInvariant());
                Directory.CreateDirectory(dir);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                new GameObject("Camera").AddComponent<Camera>();
                var probe = new GameObject("GemmaQuantProbe").AddComponent<GemmaQuantProbe>();
                probe.quant = quant;
                probe.reportDirectory = dir;
                EditorSceneManager.SaveScene(scene, TempScenePath);

                Debug.Log($"[FlashAttnProbeRunner] gemma {quant} report dir: {dir}");
                EditorApplication.isPlaying = true;
                return $"gemma {quant} probe scene built, entering play mode; report dir: " + dir;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);   // batch safety: never hang on a build failure
                return "error: " + ex.Message;
            }
        }

        static string RunMiniCPMQuant(LLMQuant quant)
        {
            try
            {
                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"minicpm5_{quant}_{runId}".ToLowerInvariant());
                Directory.CreateDirectory(dir);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                new GameObject("Camera").AddComponent<Camera>();
                var probe = new GameObject("MiniCPM5QuantProbe").AddComponent<MiniCPM5QuantProbe>();
                probe.quant = quant;
                probe.reportDirectory = dir;
                EditorSceneManager.SaveScene(scene, TempScenePath);

                Debug.Log($"[FlashAttnProbeRunner] minicpm5 {quant} report dir: {dir}");
                EditorApplication.isPlaying = true;
                return $"minicpm5 {quant} probe scene built, entering play mode; report dir: " + dir;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);   // batch safety: never hang on a build failure
                return "error: " + ex.Message;
            }
        }

        static string Run(FlashAttnProbe.LMKind kind)
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"flash_attn_{kind}_{runId}".ToLowerInvariant());
            Directory.CreateDirectory(dir);

            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            new GameObject("Camera").AddComponent<Camera>();
            var probe = new GameObject("FlashAttnProbe").AddComponent<FlashAttnProbe>();
            probe.lmKind = kind;
            probe.reportDirectory = dir;
            EditorSceneManager.SaveScene(scene, TempScenePath);

            EditorApplication.isPlaying = true;
            return $"probe scene built ({kind}), entering play mode; report dir: " + dir;
        }
    }
}
#endif
