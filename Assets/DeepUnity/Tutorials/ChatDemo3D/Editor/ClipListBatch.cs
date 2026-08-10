#if UNITY_EDITOR
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    // batch-mode cousin of AnimClipListProbe: lists the embedded clips of every character rig
    // (the generic ones carry their own clip sets and each pack ships a different list), so a
    // scene builder can pick real state names instead of guessing.
    public static class ClipListBatch
    {
        public static void ListBatch()
        {
            var sb = new StringBuilder();
            foreach (var fbx in new[]
            {
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Characters/Warrior.fbx",
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Characters/Monk.fbx",
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Characters/Rogue.fbx",
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Characters/Wizard.fbx",
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Characters/Witch.fbx",
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Animations/UAL1.fbx",
                "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Animations/UAL2.fbx",
            })
            {
                sb.AppendLine($"--- {fbx} ---");
                foreach (var a in AssetDatabase.LoadAllAssetsAtPath(fbx))
                    if (a is AnimationClip c && !c.name.StartsWith("__preview"))
                        sb.AppendLine($"  CLIP {c.name}  ({c.length:F2}s)");
            }
            Debug.Log("[ClipListBatch]\n" + sb);
            EditorApplication.Exit(0);
        }
    }
}
#endif
