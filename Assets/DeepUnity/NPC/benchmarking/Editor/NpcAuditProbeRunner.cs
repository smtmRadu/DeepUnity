#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated audit-fix run inside the real ChatDemo3D scene.
    // Same armed-state pattern as NpcInterruptProbeRunner: the probe GO exists only inside play
    // mode (scene never dirtied) and the armed flag lives in SessionState so it survives the
    // domain reload that play-mode entry can trigger.
    public static class NpcAuditProbeRunner
    {
        const string Marker = "ProbeLogs/npc_audit_probe.done";
        const string ArmedKey = "DeepUnity.NpcAuditProbe.Armed";

        [MenuItem("DeepUnity/NPC/Run Audit Probe (ChatDemo3D)")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            SessionState.SetBool(ArmedKey, true);
            EditorApplication.playModeStateChanged += SpawnOnEnter;   // no-reload path
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
            if (Object.FindObjectOfType<NpcAuditProbe>() != null) return;
            new GameObject(nameof(NpcAuditProbe)).AddComponent<NpcAuditProbe>();
        }
    }
}
#endif
