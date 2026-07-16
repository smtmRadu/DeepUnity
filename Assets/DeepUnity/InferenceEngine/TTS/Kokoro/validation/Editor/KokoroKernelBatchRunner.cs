using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Batch entry for the Kokoro kernel + stage probe (Unity closed):
        //   Unity.exe -batchmode -projectPath C:\dev\DeepUnity ^
        //     -executeMethod DeepUnity.KokoroModeling.KokoroKernelBatchRunner.Run ^
        //     -logFile ProbeLogs/kokoro_kernel.log
        // NO -nographics (compute shaders need a GPU). Creates a temp scene with the probe,
        // enters play mode, waits for ProbeLogs/kokoro_kernel.done, exits 0 on PASS / 1 on FAIL
        // / 2 on timeout (ChatterboxParityBatchRunner pattern, incl. the stuck-Unity guard:
        // leave play mode fully before EditorApplication.Exit).
        // Also usable with Unity open: menu DeepUnity/TTS/Run Kokoro Kernel Probe.
        public static class KokoroKernelBatchRunner
        {
            const string TmpScene = "Assets/__kokoro_kernel_tmp.unity";
            const string Marker = "ProbeLogs/kokoro_kernel.done";

            [MenuItem("DeepUnity/TTS/Run Kokoro Kernel Probe")]
            public static void RunInteractive() => Setup(exitWhenDone: false);

            // Bisect via the probe's serialized flags (statics don't survive the play-mode
            // domain reload): v3 (GPU-LSTM, default) / v2 (#33 convs) / v1 (#26) / legacy.
            [MenuItem("DeepUnity/TTS/Run Kokoro Kernel Probe V2 (FastKernels3 off)")]
            public static void RunInteractiveV2() => Setup(exitWhenDone: false, fastKernels3: false);

            [MenuItem("DeepUnity/TTS/Run Kokoro Kernel Probe V1 (FastKernels2 off)")]
            public static void RunInteractiveV1()
                => Setup(exitWhenDone: false, fastKernels2: false, fastKernels3: false);

            [MenuItem("DeepUnity/TTS/Run Kokoro Kernel Probe LEGACY (FastKernels off)")]
            public static void RunInteractiveLegacy()
                => Setup(exitWhenDone: false, fastKernels: false, fastKernels2: false, fastKernels3: false);

            public static void Run() => Setup(exitWhenDone: true);

            static void Setup(bool exitWhenDone, bool fastKernels = true, bool fastKernels2 = true,
                              bool fastKernels3 = true)
            {
                Directory.CreateDirectory("ProbeLogs");
                if (File.Exists(Marker)) File.Delete(Marker);

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var go = new GameObject(nameof(KokoroKernelProbe));
                var probe = go.AddComponent<KokoroKernelProbe>();
                probe.fastKernels = fastKernels;
                probe.fastKernels2 = fastKernels2;
                probe.fastKernels3 = fastKernels3;
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

                if (File.Exists(Marker))
                {
                    // Exit() while play mode is still tearing down hangs batch Unity (the known
                    // stuck-Unity gotcha) — leave play mode first, Exit once it's fully stopped.
                    if (!_stopping)
                    {
                        _stopping = true;
                        Debug.Log($"[KokoroKernelBatchRunner] probe finished: {File.ReadAllText(Marker).Trim()}");
                        EditorApplication.isPlaying = false;
                        return;
                    }
                    if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                        return;
                    string verdict = File.ReadAllText(Marker).Trim();
                    EditorApplication.update -= WatchForDone;
                    EditorApplication.Exit(verdict == "PASS" ? 0 : 1);
                }
                else if (EditorApplication.timeSinceStartup - _t0 > 1800)   // 30 min hard cap
                {
                    Debug.LogError("[KokoroKernelBatchRunner] TIMEOUT");
                    EditorApplication.update -= WatchForDone;
                    EditorApplication.Exit(2);
                }
            }
        }
    }
}
