#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Batch entry for the Velmire-on-Kokoro demo probe (Unity closed):
    //   Unity.exe -batchmode -projectPath C:\dev\DeepUnity ^
    //     -executeMethod DeepUnity.NpcKokoroDemoRunner.Run -logFile ProbeLogs/_run_npc_kokoro.log
    // Opens the REAL ChatDemo3D scene, injects the probe, enters play mode, exits 0 on PASS /
    // 1 on FAIL / 2 on timeout (leave play mode fully before Exit — the stuck-Unity gotcha).
    // Interactive: menu DeepUnity/NPC/Run Velmire-on-Kokoro Demo Probe.
    public static class NpcKokoroDemoRunner
    {
        const string DemoScene = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
        const string Marker = "ProbeLogs/npc_kokoro_demo.done";

        [MenuItem("DeepUnity/NPC/Run Velmire-on-Kokoro Demo Probe")]
        public static void RunInteractive() => Setup(exitWhenDone: false);

        public static void Run() => Setup(exitWhenDone: true);

        static void Setup(bool exitWhenDone)
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);

            EditorSceneManager.OpenScene(DemoScene);
            new GameObject(nameof(NpcKokoroDemoProbe)).AddComponent<NpcKokoroDemoProbe>();

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
                    Debug.Log($"[NpcKokoroDemoRunner] probe finished: {File.ReadAllText(Marker).Trim()}");
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
                Debug.LogError("[NpcKokoroDemoRunner] TIMEOUT");
                EditorApplication.update -= WatchForDone;
                EditorApplication.Exit(2);
            }
        }
    }
}
#endif
