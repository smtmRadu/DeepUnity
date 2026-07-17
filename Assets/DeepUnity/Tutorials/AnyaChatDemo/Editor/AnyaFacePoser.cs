#if UNITY_EDITOR
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Edit-mode expression preview: poses Anya's face by setting ARKit blendshape weights on the
    /// scene's SkinnedMeshRenderer — no LLM, no TTS, just morphs — so we can screenshot each emotion
    /// and judge whether the rig reads convincingly. Weights are matched by name substring (the
    /// runtime names carry a "blendShape1." prefix; the first match per key is the base morph).
    /// </summary>
    public static class AnyaFacePoser
    {
        static readonly (string, float)[] Neutral = { };

        static readonly (string, float)[] Happy =
        {
            ("MouthSmileLeft", 88), ("MouthSmileRight", 88),
            ("CheekSquintLeft", 55), ("CheekSquintRight", 55),
            ("EyeSquintLeft", 28), ("EyeSquintRight", 28), ("BrowInnerUp", 12),
        };

        static readonly (string, float)[] Surprised =
        {
            ("BrowInnerUp", 60), ("BrowOuterUpLeft", 72), ("BrowOuterUpRight", 72),
            ("EyeWideLeft", 58), ("EyeWideRight", 58), ("JawOpen", 34),
        };

        static readonly (string, float)[] Sad =
        {
            ("BrowInnerUp", 78), ("MouthFrownLeft", 62), ("MouthFrownRight", 62),
            ("MouthLowerDownLeft", 14), ("MouthLowerDownRight", 14), ("EyeSquintLeft", 12), ("EyeSquintRight", 12),
        };

        static readonly (string, float)[] Angry =
        {
            ("BrowDownLeft", 82), ("BrowDownRight", 82),
            ("MouthPressLeft", 42), ("MouthPressRight", 42), ("MouthFrownLeft", 22), ("MouthFrownRight", 22),
            ("NoseSneerLeft", 22), ("NoseSneerRight", 22),
        };

        // a natural mid-speech mouth (an open "ah" with a hint of smile) — fake talking, no audio
        static readonly (string, float)[] Talk =
        {
            ("JawOpen", 42), ("MouthSmileLeft", 20), ("MouthSmileRight", 20), ("BrowInnerUp", 8),
        };

        // pose EACH expression + force the skin to re-bake + render a still, all in one call — so
        // edit-mode blendshape changes actually appear (a manual Camera.Render() alone renders the
        // stale, un-reskinned mesh).
        [MenuItem("DeepUnity/Anya/Shoot All Expressions")]
        public static void ShootAll()
        {
            var smr = Object.FindObjectOfType<SkinnedMeshRenderer>();
            if (smr == null) { Debug.LogError("[AnyaPose] no SMR"); return; }
            smr.updateWhenOffscreen = true;
            smr.forceMatrixRecalculationPerRender = true;

            var camGO = new GameObject("__poseCam", typeof(Camera));
            var cam = camGO.GetComponent<Camera>();
            cam.clearFlags = CameraClearFlags.SolidColor;
            cam.backgroundColor = new Color(0.06f, 0.07f, 0.09f);
            cam.fieldOfView = 22f;
            var smrB = smr.bounds;
            float faceY = Mathf.Lerp(smrB.center.y, smrB.max.y, 0.72f);
            camGO.transform.position = new Vector3(0f, faceY + 0.01f, 0.9f);
            camGO.transform.LookAt(new Vector3(0f, faceY, 0f));

            var exprs = new (string, (string, float)[])[]
            { ("neutral", Neutral), ("happy", Happy), ("surprised", Surprised), ("sad", Sad), ("angry", Angry), ("talk", Talk) };
            var bake = new Mesh();
            foreach (var (name, e) in exprs)
            {
                ApplyTo(smr, e);
                smr.BakeMesh(bake);   // FORCES the skin+blendshape compute synchronously
                Shoot(cam, $"ProbeLogs/anya_expr_{name}.png");
            }
            Apply(Neutral);
            Object.DestroyImmediate(camGO);
            Object.DestroyImmediate(bake);
            Debug.Log("[AnyaPose] shot 6 expressions -> ProbeLogs/anya_expr_*.png");
        }

        static void Shoot(Camera cam, string path)
        {
            var rt = new RenderTexture(820, 820, 24);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            System.IO.File.WriteAllBytes(path, tex.EncodeToPNG());
            cam.targetTexture = null;
            RenderTexture.active = null;
            Object.DestroyImmediate(tex);
            Object.DestroyImmediate(rt);
        }

        static void ApplyTo(SkinnedMeshRenderer smr, (string, float)[] expr)
        {
            var mesh = smr.sharedMesh;
            for (int i = 0; i < mesh.blendShapeCount; i++) smr.SetBlendShapeWeight(i, 0f);
            foreach (var (key, w) in expr)
                for (int i = 0; i < mesh.blendShapeCount; i++)
                    if (mesh.GetBlendShapeName(i).Contains(key)) smr.SetBlendShapeWeight(i, w);
        }

        [MenuItem("DeepUnity/Anya/Pose - Neutral")]   public static void PoseNeutral()   => Apply(Neutral);
        [MenuItem("DeepUnity/Anya/Pose - Happy")]     public static void PoseHappy()     => Apply(Happy);
        [MenuItem("DeepUnity/Anya/Pose - Surprised")] public static void PoseSurprised() => Apply(Surprised);
        [MenuItem("DeepUnity/Anya/Pose - Sad")]       public static void PoseSad()       => Apply(Sad);
        [MenuItem("DeepUnity/Anya/Pose - Angry")]     public static void PoseAngry()     => Apply(Angry);
        [MenuItem("DeepUnity/Anya/Pose - Talk")]      public static void PoseTalk()      => Apply(Talk);

        static void Apply((string, float)[] expr)
        {
            var smr = Object.FindObjectOfType<SkinnedMeshRenderer>();
            if (smr == null || smr.sharedMesh == null) { Debug.LogError("[AnyaPose] no SkinnedMeshRenderer in scene"); return; }
            var mesh = smr.sharedMesh;
            for (int i = 0; i < mesh.blendShapeCount; i++) smr.SetBlendShapeWeight(i, 0f);   // reset

            var applied = new List<string>();
            foreach (var (key, w) in expr)
            {
                int hits = 0;
                for (int i = 0; i < mesh.blendShapeCount; i++)
                    if (mesh.GetBlendShapeName(i).Contains(key)) { smr.SetBlendShapeWeight(i, w); hits++; }
                if (hits > 0) applied.Add($"{key}={w}x{hits}");
                else Debug.LogWarning($"[AnyaPose] blendshape not found: {key}");
            }
            EditorUtility.SetDirty(smr);
            EditorApplication.QueuePlayerLoopUpdate();   // force the skinned mesh to re-bake in edit mode
            Debug.Log($"[AnyaPose] applied: {string.Join(", ", applied)} | mesh has {mesh.blendShapeCount} shapes");
        }
    }
}
#endif
