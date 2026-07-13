using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    namespace STTValidation
    {
        // Bridge-orchestrated STT GPU end-to-end run (play mode, temp scene):
        //   Run() -> temp scene + SttGpuProbe -> play ON
        //   (bridge polls ProbeLogs/stt_gpu.done) -> Finish() -> play OFF -> Restore()
        // The probe needs real play mode (AsyncGPUReadback + background Tasks + weight streaming).
        public static class SttGpuProbeRunner
        {
            const string TmpScene = "Assets/__stt_gpu_tmp.unity";
            const string Marker = "ProbeLogs/stt_gpu.done";

            [MenuItem("DeepUnity/STT/Run STT GPU Probe")]
            public static void Run()
            {
                Directory.CreateDirectory("ProbeLogs");
                if (File.Exists(Marker)) File.Delete(Marker);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var go = new GameObject(nameof(SttGpuProbe));
                go.AddComponent<SttGpuProbe>();
                EditorSceneManager.SaveScene(scene, TmpScene);
                EditorApplication.isPlaying = true;
            }

            public static void Finish() => EditorApplication.isPlaying = false;

            public static void Restore()
            {
                if (File.Exists(TmpScene)) AssetDatabase.DeleteAsset(TmpScene);
                var s = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            }
        }
    }
}
