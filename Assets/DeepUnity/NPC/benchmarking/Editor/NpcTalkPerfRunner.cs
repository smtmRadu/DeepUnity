#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated #29 talk-perf run inside the real ChatDemo3D scene:
    //   Run() -> opens ChatDemo3D, injects NpcTalkPerfProbe, play mode ON;
    //   the probe self-exits play mode after writing ProbeLogs/npc_talkperf.md + .done
    //   (the probe GO is an unsaved scene edit — leaving play mode discards it).
    public static class NpcTalkPerfRunner
    {
        const string DemoScene = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
        const string Marker = "ProbeLogs/npc_talkperf.done";

        [MenuItem("DeepUnity/NPC/Run Talk-Perf Probe (ChatDemo3D)")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            EditorSceneManager.OpenScene(DemoScene);
            new GameObject(nameof(NpcTalkPerfProbe)).AddComponent<NpcTalkPerfProbe>();
            EditorApplication.isPlaying = true;
        }
    }
}
#endif
