#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated #31 compact-mechanics run inside the real ChatDemo3D scene.
    // The probe GO is created only INSIDE play mode (play-mode objects are discarded on exit),
    // so the editor scene is never dirtied by the probe. Entering play mode can DOMAIN-RELOAD
    // (always when scripts just recompiled, and whenever reload-on-play is enabled) which wipes
    // a naive playModeStateChanged subscription — the armed state therefore lives in
    // SessionState (survives reloads) and re-arms from [InitializeOnLoadMethod].
    public static class NpcCompactProbeRunner
    {
        const string Marker = "ProbeLogs/npc_compact_probe.done";
        const string ArmedKey = "DeepUnity.NpcCompactProbe.Armed";

        [MenuItem("DeepUnity/NPC/Run Compact Probe (ChatDemo3D)")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            SessionState.SetBool(ArmedKey, true);
            EditorApplication.playModeStateChanged += SpawnOnEnter;   // no-reload path
            EditorApplication.isPlaying = true;
        }

        // After ANY domain reload (including the one play-mode entry itself triggers): if a run
        // is armed, spawn now when we came up inside play mode, else re-subscribe and wait.
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
            if (Object.FindObjectOfType<NpcCompactProbe>() != null) return;
            new GameObject(nameof(NpcCompactProbe)).AddComponent<NpcCompactProbe>();
        }
    }
}
#endif
