#if UNITY_EDITOR
using System;
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Batch entry for the Kokoro RTF benchmark (Unity closed):
        //   Unity.exe -batchmode -projectPath C:\dev\DeepUnity ^
        //     -executeMethod DeepUnity.KokoroModeling.KokoroRtfBatchRunner.RunFp16 ^
        //     -logFile ProbeLogs/_run_kokoro_rtf.log
        // NO -nographics. Report -> ProbeLogs/kokorortf_<QUANT>_<ts>/summary.json.
        // Exit/timeout mechanics mirror KokoroKernelBatchRunner (leave play mode fully before
        // EditorApplication.Exit — the stuck-Unity gotcha).
        public static class KokoroRtfBatchRunner
        {
            const string TmpScene = "Assets/__kokoro_rtf_tmp.unity";
            const string Marker = "ProbeLogs/kokoro_rtf.done";

            [MenuItem("DeepUnity/TTS/Kokoro RTF Benchmark (fp16)")]
            public static void RunFp16Interactive() => Setup("fp16", exitWhenDone: false);
            [MenuItem("DeepUnity/TTS/Kokoro RTF Benchmark (int8)")]
            public static void RunInt8Interactive() => Setup("int8", exitWhenDone: false);

            public static void RunFp16() => Setup("fp16", exitWhenDone: true);
            public static void RunInt8() => Setup("int8", exitWhenDone: true);

            static void Setup(string quant, bool exitWhenDone)
            {
                Directory.CreateDirectory("ProbeLogs");
                if (File.Exists(Marker)) File.Delete(Marker);

                string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
                string dir = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs",
                                          $"kokorortf_{quant.ToUpperInvariant()}_{runId}");

                var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
                var probe = new GameObject(nameof(KokoroRtfProbe)).AddComponent<KokoroRtfProbe>();
                probe.quant = quant;
                probe.reportDirectory = dir;
                EditorSceneManager.SaveScene(scene, TmpScene);

                Debug.Log($"[KokoroRtfBatchRunner] {quant} report dir: {dir}");
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
                    if (!_stopping)
                    {
                        _stopping = true;
                        Debug.Log($"[KokoroRtfBatchRunner] probe finished: {File.ReadAllText(Marker).Trim()}");
                        EditorApplication.isPlaying = false;
                        return;
                    }
                    if (EditorApplication.isPlaying || EditorApplication.isPlayingOrWillChangePlaymode)
                        return;
                    string verdict = File.ReadAllText(Marker).Trim();
                    EditorApplication.update -= WatchForDone;
                    EditorApplication.Exit(verdict == "PASS" ? 0 : 1);
                }
                else if (EditorApplication.timeSinceStartup - _t0 > 900)   // 15 min hard cap
                {
                    Debug.LogError("[KokoroRtfBatchRunner] TIMEOUT");
                    EditorApplication.update -= WatchForDone;
                    EditorApplication.Exit(2);
                }
            }
        }
    }
}
#endif
