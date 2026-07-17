#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D.EditorTools
{
    /// <summary>
    /// Screenshot lineup for the IN-HAND sword orientation (the StowTuneProbe covers the stowed
    /// poses; this covers the held one). Samples the idle clip so the hand is in the real game
    /// pose, then applies each candidate local pre-rotation to the actual sword inside its
    /// Weapon.R mount and renders two views per candidate (3/4 front + top-down, with a floor
    /// bar marking the character's forward). Pick the frame whose blade points forward, copy its
    /// euler into ChatDemo3DBuilder's sword attach. Nothing is saved — the scene is left dirty
    /// in-memory only (batch exits without saving; interactive users just don't save).
    /// Batch: -executeMethod DeepUnity.Tutorials.ChatDemo3D.EditorTools.SwordHoldTuneProbe.Run -quit
    /// </summary>
    public static class SwordHoldTuneProbe
    {
        const string Scene = "Assets/DeepUnity/Tutorials/ChatDemo3D/ChatDemo3D.unity";
        const string OutDir = "ProbeLogs/swordhold";

        static readonly Vector3[] Cands =
        {
            new Vector3(0f, 0f, 0f),
            new Vector3(0f, 45f, 0f),
            new Vector3(0f, 90f, 0f),
            new Vector3(0f, 135f, 0f),
            new Vector3(0f, 180f, 0f),
            new Vector3(0f, -135f, 0f),
            new Vector3(0f, -90f, 0f),
            new Vector3(0f, -45f, 0f),
        };

        [MenuItem("DeepUnity/ChatDemo3D/SwordHoldTune - Lineup Screenshots")]
        public static void Run()
        {
            EditorSceneManager.OpenScene(Scene);
            var pc = Object.FindObjectOfType<SoulsPlayerController>();
            if (pc == null) { Debug.LogError("[SwordHoldTune] no SoulsPlayerController"); return; }
            Transform root = pc.transform;
            var anim = pc.GetComponentInChildren<Animator>();
            Transform sword = FindDeep(root, "Sword");
            if (anim == null || sword == null) { Debug.LogError("[SwordHoldTune] animator/sword missing"); return; }

            // hand must be in the game idle pose, not the bind pose
            AnimationClip idle = null;
            foreach (var c in anim.runtimeAnimatorController.animationClips)
                if (c.name == "Idle" || c.name.Contains("Sword_Idle")) { idle = c; break; }
            if (idle != null) idle.SampleAnimation(anim.gameObject, 0f);

            // floor bar marking root.forward so the top view reads unambiguously
            var fwdBar = GameObject.CreatePrimitive(PrimitiveType.Cube).transform;
            fwdBar.name = "__fwdBar";
            fwdBar.position = root.position + root.forward * 0.9f + Vector3.up * 0.02f;
            fwdBar.rotation = root.rotation;
            fwdBar.localScale = new Vector3(0.06f, 0.04f, 1.8f);

            Directory.CreateDirectory(OutDir);
            Quaternion orig = sword.localRotation;
            for (int i = 0; i < Cands.Length; i++)
            {
                sword.localRotation = Quaternion.Euler(Cands[i]) * orig;
                string tag = $"{i}_y{Cands[i].y:0}";
                Shot(root, root.position + root.forward * 2.3f + root.right * 1.3f + Vector3.up * 1.5f,
                     root.position + Vector3.up * 1.0f, Vector3.up, $"{tag}_front");
                Shot(root, root.position + Vector3.up * 3.6f,
                     root.position + Vector3.up * 0.8f, root.forward, $"{tag}_top");
            }
            sword.localRotation = orig;
            Object.DestroyImmediate(fwdBar.gameObject);
            Debug.Log($"[SwordHoldTune] {Cands.Length * 2} shots -> {OutDir} (candidates: local pre-rot, Y sweep)");
        }

        static void Shot(Transform root, Vector3 camPos, Vector3 lookAt, Vector3 up, string name)
        {
            var go = new GameObject("__shotCam");
            var cam = go.AddComponent<Camera>();
            cam.transform.position = camPos;
            cam.transform.rotation = Quaternion.LookRotation((lookAt - camPos).normalized, up);
            cam.fieldOfView = 45f;
            cam.clearFlags = CameraClearFlags.Color;
            cam.backgroundColor = new Color(0.16f, 0.17f, 0.20f);
            var rt = new RenderTexture(900, 700, 24);
            cam.targetTexture = rt;
            cam.Render();
            RenderTexture.active = rt;
            var tex = new Texture2D(rt.width, rt.height, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, rt.width, rt.height), 0, 0);
            tex.Apply();
            File.WriteAllBytes(Path.Combine(OutDir, name + ".png"), tex.EncodeToPNG());
            RenderTexture.active = null;
            cam.targetTexture = null;
            Object.DestroyImmediate(rt);
            Object.DestroyImmediate(tex);
            Object.DestroyImmediate(go);
        }

        static Transform FindDeep(Transform t, string name)
        {
            if (t.name == name) return t;
            foreach (Transform c in t) { var r = FindDeep(c, name); if (r != null) return r; }
            return null;
        }
    }
}
#endif
