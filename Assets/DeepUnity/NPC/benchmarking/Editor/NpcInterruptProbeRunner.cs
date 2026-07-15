#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated mid-reply-interruption run inside the real ChatDemo3D scene.
    // Same armed-state pattern as NpcCompactProbeRunner: the probe GO exists only inside play
    // mode (scene never dirtied) and the armed flag lives in SessionState so it survives the
    // domain reload that play-mode entry can trigger.
    public static class NpcInterruptProbeRunner
    {
        const string Marker = "ProbeLogs/npc_interrupt_probe.done";
        const string ArmedKey = "DeepUnity.NpcInterruptProbe.Armed";

        [MenuItem("DeepUnity/NPC/Run Interrupt Probe (ChatDemo3D)")]
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
            if (Object.FindObjectOfType<NpcInterruptProbe>() != null) return;
            new GameObject(nameof(NpcInterruptProbe)).AddComponent<NpcInterruptProbe>();
        }
    }
}
#endif
