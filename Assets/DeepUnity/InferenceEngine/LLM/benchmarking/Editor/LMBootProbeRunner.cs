#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Batch entry point for the Qwen3.5 boot smoothness probe:
    //   Unity.exe -projectPath <proj> -batchmode -executeMethod DeepUnity.LMBootProbeRunner.RunQwenBootProbe
    // (NO -nographics: compute shaders need a graphics device.)
    public static class LMBootProbeRunner
    {
        const string TempScenePath = "Assets/__lm_boot_probe_tmp.unity";

        public static void RunQwenBootProbe()
        {
            try
            {
                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"qwen_boot_probe_{runId}");
                Directory.CreateDirectory(dir);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var camGo = new GameObject("Camera");
                camGo.AddComponent<Camera>();
                var go = new GameObject("LMBootProbe");
                var probe = go.AddComponent<LMBootProbe>();
                probe.reportDirectory = dir;
                EditorSceneManager.SaveScene(scene, TempScenePath);

                Debug.Log($"[LMBootProbeRunner] Report dir: {dir}");
                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }
    }
}
#endif
