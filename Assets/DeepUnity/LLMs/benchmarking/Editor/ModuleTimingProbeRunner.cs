#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Bridge/batch entry points for the per-module timing probe (ModuleTimingProbe — a scaffold;
    // the timing body is not implemented yet, so these currently just write a stub report).
    //   bridge: { "action": "invoke", "method": "DeepUnity.ModuleTimingProbeRunner.RunGemmaFP16" }
    //   batch : Unity.exe -batchmode -executeMethod DeepUnity.ModuleTimingProbeRunner.RunGemmaFP16
    //           (NO -nographics: compute shaders need a graphics device.)
    public static class ModuleTimingProbeRunner
    {
        const string TempScenePath = "Assets/__module_timing_probe_tmp.unity";

        public static string RunGemmaFP16() => Run(FlashAttnProbe.LMKind.Gemma3, LLMQuant.FP16);
        public static string RunGemmaInt8() => Run(FlashAttnProbe.LMKind.Gemma3, LLMQuant.INT8);
        public static string RunGemmaInt4() => Run(FlashAttnProbe.LMKind.Gemma3, LLMQuant.INT4);
        public static string RunQwenFP16()  => Run(FlashAttnProbe.LMKind.Qwen3_5, LLMQuant.FP16);
        public static string RunQwenInt8()  => Run(FlashAttnProbe.LMKind.Qwen3_5, LLMQuant.INT8);
        public static string RunQwenInt4()  => Run(FlashAttnProbe.LMKind.Qwen3_5, LLMQuant.INT4);

        static string Run(FlashAttnProbe.LMKind kind, LLMQuant quant)
        {
            try
            {
                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs",
                                          $"module_timing_{kind}_{quant}_{runId}".ToLowerInvariant());
                Directory.CreateDirectory(dir);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                new GameObject("Camera").AddComponent<Camera>();
                var probe = new GameObject("ModuleTimingProbe").AddComponent<ModuleTimingProbe>();
                probe.lmKind = kind;
                probe.quant = quant;
                probe.reportDirectory = dir;
                EditorSceneManager.SaveScene(scene, TempScenePath);

                Debug.Log($"[ModuleTimingProbeRunner] {kind} {quant} report dir: {dir}");
                EditorApplication.isPlaying = true;
                return $"module-timing probe scene built ({kind} {quant}), entering play mode; report dir: " + dir;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);   // batch safety: never hang on a build failure
                return "error: " + ex.Message;
            }
        }
    }
}
#endif
