using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity
{
    // Bridge-orchestrated play-mode run for ZoneEntryProbe (same pattern as
    // QwenKokoroPerfRunner: Run -> poll .done -> Finish -> Restore).
    public static class ZoneEntryRunner
    {
        const string TmpScene = "Assets/__zone_entry_tmp.unity";
        const string DemoScene = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
        const string Marker = "ProbeLogs/zone_entry_probe.done";

        [MenuItem("DeepUnity/LLM/Run Zone-Entry Freeze Probe")]
        public static void Run()
        {
            Directory.CreateDirectory("ProbeLogs");
            if (File.Exists(Marker)) File.Delete(Marker);
            var scene = EditorSceneManager.NewScene(NewSceneSetup.DefaultGameObjects, NewSceneMode.Single);
            var go = new GameObject(nameof(ZoneEntryProbe));
            go.AddComponent<ZoneEntryProbe>();
            EditorSceneManager.SaveScene(scene, TmpScene);
            EditorApplication.isPlaying = true;
        }

        public static void Finish() => EditorApplication.isPlaying = false;

        public static void Restore()
        {
            EditorSceneManager.OpenScene(DemoScene);
            AssetDatabase.DeleteAsset(TmpScene);
        }
    }
}
