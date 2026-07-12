using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // Batch entry for the Chatterbox parity probe (Unity closed):
        //   Unity.exe -batchmode -projectPath C:\dev\DeepUnity ^
        //     -executeMethod DeepUnity.ChatterboxModeling.ChatterboxParityBatchRunner.Run ^
        //     -logFile ProbeLogs/chatterbox_parity.log
        // NO -nographics (compute shaders need a GPU). Creates a temp scene with the probe,
        // enters play mode, waits for the done marker, exits 0 on PASS / 1 on FAIL.
        // Also usable with Unity open: menu DeepUnity/TTS/Run Chatterbox Parity Probe.
        public static class ChatterboxParityBatchRunner
        {
            const string TmpScene = "Assets/__chatterbox_parity_tmp.unity";
            const string Marker = "ProbeLogs/chatterbox_parity.done";

            [MenuItem("DeepUnity/TTS/Run Chatterbox Parity Probe")]
            public static void RunInteractive() => Setup<ChatterboxParityProbe>(exitWhenDone: false, Marker);

            public static void Run() => Setup<ChatterboxParityProbe>(exitWhenDone: true, Marker);

            const string Int8Dir = "Assets/Resources/Weights/weights_chatterbox_turbo_int8";

            [MenuItem("DeepUnity/TTS/Run Chatterbox Parity Probe (int8)")]
            public static void RunInteractiveInt8() => Setup<ChatterboxParityProbe>(exitWhenDone: false, Marker,
                go => go.GetComponent<ChatterboxParityProbe>().weightsDir = Int8Dir);

            public static void RunInt8() => Setup<ChatterboxParityProbe>(exitWhenDone: true, Marker,
                go => go.GetComponent<ChatterboxParityProbe>().weightsDir = Int8Dir);

            [MenuItem("DeepUnity/TTS/Run Chatterbox Listen Probe (sampled audio)")]
            public static void RunListenInteractive() => Setup<ChatterboxListenProbe>(exitWhenDone: false, "ProbeLogs/chatterbox_listen.done");

            public static void RunListen() => Setup<ChatterboxListenProbe>(exitWhenDone: true, "ProbeLogs/chatterbox_listen.done");

            static string _marker;
            static void Setup<T>(bool exitWhenDone, string marker, System.Action<GameObject> configure = null) where T : Component
            {
                _marker = marker;
                Directory.CreateDirectory("ProbeLogs");
                if (File.Exists(marker)) File.Delete(marker);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var go = new GameObject(typeof(T).Name);
                go.AddComponent<T>();
                configure?.Invoke(go);
                EditorSceneManager.SaveScene(scene, TmpScene);

                if (exitWhenDone)
                    EditorApplication.update += WatchForDone;

                EditorApplication.isPlaying = true;
            }

            static double _t0;
            static bool _stopping;
            static void WatchForDone()
            {
                if (_t0 == 0) _t0 = EditorApplication.timeSinceStartup;

                if (File.Exists(_marker))
                {
                    // Exit() while play mode is still tearing down hangs batch Unity (the known
                    // stuck-Unity gotcha) — leave play mode first, Exit once it's fully stopped.
                    if (!_stopping)
                    {
                        _stopping = true;
                        Debug.Log($"[ChatterboxParityBatchRunner] probe finished: {File.ReadAllText(_marker).Trim()}");
                        EditorApplication.isPlaying = false;
                        return;
                    }
                    if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                        return;
                    string verdict = File.ReadAllText(_marker).Trim();
                    EditorApplication.update -= WatchForDone;
                    EditorApplication.Exit(verdict == "PASS" ? 0 : 1);
                }
                else if (EditorApplication.timeSinceStartup - _t0 > 1800)   // 30 min hard cap
                {
                    Debug.LogError("[ChatterboxParityBatchRunner] TIMEOUT");
                    EditorApplication.update -= WatchForDone;
                    EditorApplication.Exit(2);
                }
            }
        }
    }
}
