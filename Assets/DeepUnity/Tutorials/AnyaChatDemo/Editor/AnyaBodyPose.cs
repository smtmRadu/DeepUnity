#if UNITY_EDITOR
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// The Rocketbox FBX imports in its A-pose (arms out to the sides) and nothing animates the body,
    /// so Anya is frozen mid-jumping-jack. This lowers the arms to a natural rest by swinging each
    /// upper-arm bone so it points down (with a hint of forward + outward). It's axis-agnostic — it
    /// rotates from the arm's current shoulder→elbow direction to the target, so it works for both
    /// sides without guessing local-bone axes. Callable standalone AND folded into the scene builder.
    /// </summary>
    public static class AnyaBodyPose
    {
        [MenuItem("DeepUnity/Anya/Lower Arms")]
        public static void LowerArmsMenu()
        {
            var smr = Object.FindObjectOfType<SkinnedMeshRenderer>();
            if (smr == null) { Debug.LogError("[AnyaBody] no SMR in scene"); return; }
            LowerArms(smr.transform.root);
            smr.forceMatrixRecalculationPerRender = true;
            smr.updateWhenOffscreen = true;
            var bake = new Mesh(); smr.BakeMesh(bake); Object.DestroyImmediate(bake);
            EditorUtility.SetDirty(smr.transform.root);
            UnityEditor.SceneManagement.EditorSceneManager.MarkSceneDirty(smr.gameObject.scene);
            Debug.Log("[AnyaBody] arms lowered");
        }

        // rest a whole arm chain: upper arm points down+slightly forward/out, elbow softened a touch
        public static void LowerArms(Transform root)
        {
            Pose(root, "Bip01 L UpperArm", "Bip01 L Forearm", new Vector3(-0.16f, -1f, 0.10f));
            Pose(root, "Bip01 R UpperArm", "Bip01 R Forearm", new Vector3(0.16f, -1f, 0.10f));
        }

        static void Pose(Transform root, string upperName, string elbowName, Vector3 targetDir)
        {
            var upper = Find(root, upperName);
            var elbow = Find(root, elbowName);
            if (upper == null || elbow == null) { Debug.LogWarning($"[AnyaBody] missing {upperName}/{elbowName}"); return; }

            Vector3 curDir = (elbow.position - upper.position).normalized;
            Quaternion swing = Quaternion.FromToRotation(curDir, targetDir.normalized);
            upper.rotation = swing * upper.rotation;
        }

        static Transform Find(Transform root, string name)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == name) return t;
            return null;
        }
    }
}
#endif
