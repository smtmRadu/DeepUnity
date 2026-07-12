using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated E2E run inside the real ChatDemo2D scene (same pattern as NpcE2ERunner).
    public static class NpcE2E2DRunner
    {
        const string DemoScene = "Assets/DeepUnity/Tutorials/ChatDemo2D/ChatDemo2D.unity";
        const string Marker = "ProbeLogs/npc_e2e_2d.done";

        [MenuItem("DeepUnity/NPC/Run NPC E2E Probe (ChatDemo2D)")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            EditorSceneManager.OpenScene(DemoScene);
            var go = new GameObject(nameof(NpcE2E2DProbe));
            go.AddComponent<NpcE2E2DProbe>();
            EditorApplication.isPlaying = true;
        }

        public static void Finish() => EditorApplication.isPlaying = false;

        public static void Restore()
        {
            EditorSceneManager.OpenScene("Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity");
        }
    }
}
