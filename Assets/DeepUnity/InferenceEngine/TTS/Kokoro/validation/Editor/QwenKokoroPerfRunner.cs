using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Frame-pacing run for the demo voice stack, orchestrated over the ClaudeBridge:
        //   1. Run()     — temp scene with QwenKokoroPerfProbe, enters play mode
        //   2. (bridge polls ProbeLogs/qwen_kokoro_perf.done)
        //   3. Finish()  — leaves play mode
        //   4. Restore() — reopens ChatDemo3D, deletes the temp scene
        // No EditorApplication.update watcher: play-mode domain reload wipes static state, so
        // the bridge drives each step explicitly.
        public static class QwenKokoroPerfRunner
        {
            const string TmpScene = "Assets/__qwen_kokoro_perf_tmp.unity";
            const string DemoScene = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
            const string Marker = "ProbeLogs/qwen_kokoro_perf.done";

            [MenuItem("DeepUnity/TTS/Run Qwen+Kokoro Perf Probe")]
            public static void Run()
            {
                Directory.CreateDirectory("ProbeLogs");
                if (File.Exists(Marker)) File.Delete(Marker);
                var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
                var go = new GameObject(nameof(QwenKokoroPerfProbe));
                go.AddComponent<QwenKokoroPerfProbe>();
                EditorSceneManager.SaveScene(scene, TmpScene);
                EditorApplication.isPlaying = true;
            }

            public static void Finish() => EditorApplication.isPlaying = false;

            public static void Restore()
            {
                EditorSceneManager.OpenScene(DemoScene);
                AssetDatabase.DeleteAsset(TmpScene);
            }
        }
    }
}
