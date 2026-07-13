#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

namespace DeepUnity
{
    // Launcher for LMFramePacingProbe (task #20). Remembers the currently open scene, spins up a
    // temp empty scene with the probe, and enters play mode; when the probe self-exits play mode,
    // the [InitializeOnLoadMethod] hook below reopens the remembered scene and deletes the temp
    // scene — fully self-driving from ClaudeBridge (menu → poll status file → editor restored).
    public static class LMFramePacingRunner
    {
        const string TempScenePath = "Assets/__lm_framepacing_tmp.unity";
        const string RestoreKey = "lmfp_restore_scene";   // SessionState survives domain reloads

        [MenuItem("Tools/DeepUnity/Benchmarks/Frame Pacing Probe/Qwen3.5 INT8 (kv INT8)")]
        static void FP_Q_INT8() => Launch(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT8, KVQuant.INT8);

        [MenuItem("Tools/DeepUnity/Benchmarks/Frame Pacing Probe/Qwen3.5 FP16 (kv FP16)")]
        static void FP_Q_FP16() => Launch(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.FP16, KVQuant.FP16);

        // batch entry point (Unity closed): -executeMethod DeepUnity.LMFramePacingRunner.RunFramePacingProbe
        public static void RunFramePacingProbe() => Launch(ProbeModelKind.Qwen3_5_0_8B, LLMQuant.INT8, KVQuant.INT8);

        static void Launch(ProbeModelKind kind, LLMQuant quant, KVQuant kv)
        {
            try
            {
                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs",
                                          $"framepacing_{LMProbeCommon.ModelLabel(kind)}_{quant}_{runId}");
                Directory.CreateDirectory(dir);

                SessionState.SetString(RestoreKey, SceneManager.GetActiveScene().path ?? "");

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                new GameObject("Camera").AddComponent<Camera>();
                var probe = new GameObject("LMFramePacingProbe").AddComponent<LMFramePacingProbe>();
                probe.model = kind;
                probe.quant = quant;
                probe.kvQuant = kv;
                probe.reportDirectory = dir;
                EditorSceneManager.SaveScene(scene, TempScenePath);

                Debug.Log($"[LMFramePacingRunner] report dir: {dir}");
                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
            }
        }

        [InitializeOnLoadMethod]
        static void RegisterRestore()
        {
            EditorApplication.playModeStateChanged += c =>
            {
                if (c != PlayModeStateChange.EnteredEditMode) return;
                string prev = SessionState.GetString(RestoreKey, "");
                if (string.IsNullOrEmpty(prev)) return;
                SessionState.EraseString(RestoreKey);
                try
                {
                    if (File.Exists(prev)) EditorSceneManager.OpenScene(prev);
                    AssetDatabase.DeleteAsset(TempScenePath);
                    Debug.Log($"[LMFramePacingRunner] restored scene {prev}");
                }
                catch (Exception ex)
                {
                    Debug.LogException(ex);
                }
            };
        }
    }
}
#endif
