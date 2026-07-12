using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated E2E run INSIDE the real ChatDemo3D scene:
    //   Run() -> opens ChatDemo3D, injects NpcE2EProbe, play mode ON
    //   (bridge polls ProbeLogs/npc_e2e.done) -> Finish() -> play OFF (scene reloads clean)
    public static class NpcE2ERunner
    {
        const string DemoScene = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
        const string Marker = "ProbeLogs/npc_e2e.done";

        [MenuItem("DeepUnity/NPC/Run NPC E2E Probe (ChatDemo3D)")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            EditorSceneManager.OpenScene(DemoScene);
            var go = new GameObject(nameof(NpcE2EProbe));
            go.AddComponent<NpcE2EProbe>();
            EditorApplication.isPlaying = true;
            // the probe GO is a scene edit that never gets saved — leaving play mode discards it
        }

        public static void Finish() => EditorApplication.isPlaying = false;

        public static void Restore()
        {
            EditorSceneManager.OpenScene(DemoScene);   // reload pristine (drops the probe GO)
        }
    }
}
