#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Scene-true manual-play repro run (see NpcManualReproProbe). Same SessionState re-arm
    // pattern as the other probe runners (play-mode entry can domain-reload).
    public static class NpcManualReproProbeRunner
    {
        const string Marker = "ProbeLogs/npc_manual_repro_probe.done";
        const string ArmedKey = "DeepUnity.NpcManualReproProbe.Armed";

        [MenuItem("DeepUnity/NPC/Run Manual Repro Probe (ChatDemo3D)")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            SessionState.SetBool(ArmedKey, true);
            EditorApplication.playModeStateChanged += SpawnOnEnter;
            EditorApplication.isPlaying = true;
        }

        [InitializeOnLoadMethod]
        static void Rearm()
        {
            if (!SessionState.GetBool(ArmedKey, false)) return;
            EditorApplication.delayCall += () =>
            {
                if (!SessionState.GetBool(ArmedKey, false)) return;
                if (Application.isPlaying) Spawn();
                else EditorApplication.playModeStateChanged += SpawnOnEnter;
            };
        }

        static void SpawnOnEnter(PlayModeStateChange s)
        {
            if (s != PlayModeStateChange.EnteredPlayMode) return;
            EditorApplication.playModeStateChanged -= SpawnOnEnter;
            Spawn();
        }

        static void Spawn()
        {
            SessionState.SetBool(ArmedKey, false);
            if (Object.FindObjectOfType<NpcManualReproProbe>() != null) return;
            new GameObject(nameof(NpcManualReproProbe)).AddComponent<NpcManualReproProbe>();
        }
    }
}
#endif
