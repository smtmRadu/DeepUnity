#if UNITY_EDITOR
using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    // One-off diagnostic: confirms the Rocketbox facial FBX imported with usable blendshapes,
    // sane scale, and that the textures resolved — BEFORE we invest in the scene builder.
    public static class AnyaModelProbe
    {
        const string FBX = "Assets/DeepUnity/Tutorials/AnyaChatDemo/Art/Female_Adult_01/Export/Female_Adult_01_facial.fbx";

        [MenuItem("DeepUnity/Anya/Probe Model")]
        public static void Probe()
        {
            var go = AssetDatabase.LoadAssetAtPath<GameObject>(FBX);
            if (go == null) { Debug.LogError($"[AnyaProbe] FBX not found/imported at {FBX}"); return; }

            var sb = new StringBuilder();
            sb.AppendLine($"[AnyaProbe] loaded {go.name}");
            var smrs = go.GetComponentsInChildren<SkinnedMeshRenderer>(true);
            sb.AppendLine($"[AnyaProbe] SkinnedMeshRenderers: {smrs.Length}");
            foreach (var smr in smrs)
            {
                var m = smr.sharedMesh;
                sb.AppendLine($"  - '{smr.name}' mesh='{(m ? m.name : "null")}' verts={(m ? m.vertexCount : 0)} " +
                              $"blendshapes={(m ? m.blendShapeCount : 0)} boundsY={(m ? m.bounds.size.y.ToString("F3") : "?")} " +
                              $"mats={smr.sharedMaterials.Length}");
                if (m != null && m.blendShapeCount > 0)
                {
                    var names = new StringBuilder("      visemes/morphs: ");
                    for (int i = 0; i < Mathf.Min(m.blendShapeCount, 12); i++) names.Append(m.GetBlendShapeName(i)).Append(" | ");
                    sb.AppendLine(names.ToString());
                    // hunt for the mouth/jaw morphs lip-sync needs
                    var mouth = new StringBuilder("      mouth/jaw: ");
                    for (int i = 0; i < m.blendShapeCount; i++)
                    {
                        string n = m.GetBlendShapeName(i).ToLowerInvariant();
                        if (n.Contains("jaw") || n.Contains("mouth") || n.Contains("lip") || n.Contains("funnel") || n.Contains("pucker"))
                            mouth.Append(m.GetBlendShapeName(i)).Append(' ');
                    }
                    sb.AppendLine(mouth.ToString());
                }
            }
            // texture import sanity
            string texDir = "Assets/DeepUnity/Tutorials/AnyaChatDemo/Art/Female_Adult_01/Textures/";
            foreach (var t in new[] { "f001_head_color.tga", "f001_head_normal.tga", "f001_head_specular.tga" })
            {
                var tex = AssetDatabase.LoadAssetAtPath<Texture2D>(texDir + t);
                sb.AppendLine($"  tex {t}: {(tex ? tex.width + "x" + tex.height : "NOT IMPORTED")}");
            }
            Debug.Log(sb.ToString());
        }

        // Full dump for the procedural "life" layer: every blendshape name (so I can grab the ARKit
        // eye-look / brow / mouth morphs by exact name) + the bones that matter for head/gaze motion.
        [MenuItem("DeepUnity/Anya/Dump Face + Skeleton")]
        public static void DumpFace()
        {
            var go = AssetDatabase.LoadAssetAtPath<GameObject>(FBX);
            if (go == null) { Debug.LogError($"[AnyaDump] FBX not found at {FBX}"); return; }
            var smr = go.GetComponentInChildren<SkinnedMeshRenderer>(true);
            if (smr == null || smr.sharedMesh == null) { Debug.LogError("[AnyaDump] no SMR/mesh on FBX"); return; }
            var m = smr.sharedMesh;

            var sb = new StringBuilder();
            sb.AppendLine($"[AnyaDump] {m.blendShapeCount} blendshapes on '{smr.name}':");
            for (int i = 0; i < m.blendShapeCount; i++) sb.AppendLine($"  [{i:D3}] {m.GetBlendShapeName(i)}");

            // every transform (bone) name so I can pick the head/neck bone for sway+nod and confirm eyes
            sb.AppendLine("[AnyaDump] all transforms (bone hierarchy):");
            foreach (var t in go.GetComponentsInChildren<Transform>(true))
                sb.AppendLine($"  xform '{t.name}'");
            Debug.Log(sb.ToString());
        }
    }
}
#endif
