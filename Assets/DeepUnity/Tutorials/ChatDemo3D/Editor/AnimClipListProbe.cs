#if UNITY_EDITOR
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    // one-off: list every AnimationClip inside the UAL packs so we know whether a real
    // sheath/unsheath animation exists before falling back to a procedural weapon tween
    public static class AnimClipListProbe
    {
        [MenuItem("DeepUnity/ChatDemo3D/List Animation Clips")]
        public static void ListClips()
        {
            var sb = new StringBuilder();
            foreach (var fbx in new[] { "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Animations/UAL1.fbx",
                                        "Assets/DeepUnity/Tutorials/ChatDemo3D/Art/Animations/UAL2.fbx" })
            {
                sb.AppendLine($"--- {fbx} ---");
                foreach (var a in AssetDatabase.LoadAllAssetsAtPath(fbx))
                    if (a is AnimationClip c && !c.name.StartsWith("__preview"))
                        sb.AppendLine($"  {c.name}  ({c.length:F2}s)");
            }
            Debug.Log(sb.ToString());
        }
    }
}
#endif
