#if UNITY_EDITOR
using System.IO;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Renders the procedural idle (<see cref="AnyaLifeLayer"/>) to a sequence of stills in EDIT mode
    /// — no play mode, no LLM/TTS — so the motion can be judged as a flipbook. Anti-aliased (MSAA 8)
    /// to kill the jagged edges, and the clip window is auto-centred on a genuine smile so the proof
    /// always contains one. Frames land in ProbeLogs/anya_life/frame_####.jpg.
    /// </summary>
    public static class AnyaFilmStrip
    {
        const int FPS = 24;
        const float SECS = 8f;
        const int W = 460, H = 540;      // portrait; MSAA-resolved so it stays crisp when shown ~same size

        [MenuItem("DeepUnity/Anya/Life — Preview Stills (5)")]
        public static void Stills()
        {
            if (!Setup(out var life, out var cam, out var smr, out var head, out var rest)) return;
            float t0 = SmileCentredStart();
            Directory.CreateDirectory("ProbeLogs");
            float[] ts = { t0 + 0.5f, t0 + 2f, t0 + 4f, t0 + 5f, t0 + 7f };
            for (int i = 0; i < ts.Length; i++) RenderFrame(life, cam, smr, ts[i], $"ProbeLogs/anya_life_still_{i}.jpg");
            Teardown(smr, head, rest, cam);
            Debug.Log($"[AnyaLife] 5 stills -> ProbeLogs/anya_life_still_*.jpg (t0={t0:F2}, smile peak at still #2)");
        }

        const string TRACK = "Assets/DeepUnity/Tutorials/AnyaChatDemo/Art/anya_idle_mocap.bytes";

        [MenuItem("DeepUnity/Anya/Mocap — Preview Stills (5)")]
        public static void MocapStills()
        {
            if (!SetupMocap(out var mocap, out var cam, out var smr, out var head, out var rest)) return;
            mocap.Smooth = 0f;   // stills jump in time; EMA would lag
            Directory.CreateDirectory("ProbeLogs");
            float[] ts = { 2f, 9f, 16f, 23f, 30f };
            for (int i = 0; i < ts.Length; i++)
            { mocap.Evaluate(ts[i]); Bake(smr); Shoot(cam, $"ProbeLogs/anya_mocap_still_{i}.jpg"); }
            Teardown(smr, head, rest, cam);
            Debug.Log("[AnyaMocap] 5 stills -> ProbeLogs/anya_mocap_still_*.jpg");
        }

        [MenuItem("DeepUnity/Anya/Mocap — Render Filmstrip")]
        public static void MocapFilmstrip()
        {
            if (!SetupMocap(out var mocap, out var cam, out var smr, out var head, out var rest)) return;
            string dir = "ProbeLogs/anya_mocap";
            if (Directory.Exists(dir)) foreach (var f in Directory.GetFiles(dir, "*.jpg")) File.Delete(f);
            Directory.CreateDirectory(dir);
            float t0 = 1.0f;   // skip the very start of the capture
            int n = Mathf.RoundToInt(SECS * FPS);
            for (int f = 0; f < n; f++)
            {
                mocap.Evaluate(t0 + f / (float)FPS);
                Bake(smr);
                Shoot(cam, $"{dir}/frame_{f:D4}.jpg");
                if ((f & 31) == 0) EditorUtility.DisplayProgressBar("Anya mocap", $"frame {f}/{n}", f / (float)n);
            }
            EditorUtility.ClearProgressBar();
            Teardown(smr, head, rest, cam);
            Debug.Log($"[AnyaMocap] rendered {n} frames @ {FPS}fps -> {dir}/frame_####.jpg");
        }

        static bool SetupMocap(out AnyaMocapTrack mocap, out Camera cam, out SkinnedMeshRenderer smr, out Transform head, out Quaternion rest)
        {
            mocap = null;
            if (!Setup(out _, out cam, out smr, out head, out rest)) return false;
            var track = AssetDatabase.LoadAssetAtPath<TextAsset>(TRACK);
            if (track == null) { Debug.LogError($"[AnyaMocap] no track at {TRACK} — run the extraction first"); return false; }
            mocap = new AnyaMocapTrack();
            mocap.Init(smr, track.bytes);
            return mocap.Ready;
        }

        static void Bake(SkinnedMeshRenderer smr)
        {
            var m = new Mesh();
            smr.BakeMesh(m);   // force synchronous skin+blendshape compute in edit mode
            Object.DestroyImmediate(m);
        }

        static void Shoot(Camera cam, string path)
        {
            var msaa = new RenderTexture(W, H, 24) { antiAliasing = 8 };
            msaa.Create();
            var resolve = new RenderTexture(W, H, 0);
            resolve.Create();
            var prevRT = cam.targetTexture;
            cam.targetTexture = msaa;
            cam.Render();
            Graphics.Blit(msaa, resolve);
            RenderTexture.active = resolve;
            var tex = new Texture2D(W, H, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToJPG(82));
            cam.targetTexture = prevRT;
            RenderTexture.active = null;
            Object.DestroyImmediate(tex);
            msaa.Release(); Object.DestroyImmediate(msaa);
            resolve.Release(); Object.DestroyImmediate(resolve);
        }

        [MenuItem("DeepUnity/Anya/Life — Render Idle Filmstrip")]
        public static void Filmstrip()
        {
            if (!Setup(out var life, out var cam, out var smr, out var head, out var rest)) return;
            float t0 = SmileCentredStart();
            string dir = "ProbeLogs/anya_life";
            if (Directory.Exists(dir)) foreach (var f in Directory.GetFiles(dir, "*.jpg")) File.Delete(f);
            Directory.CreateDirectory(dir);

            int n = Mathf.RoundToInt(SECS * FPS);
            for (int f = 0; f < n; f++)
            {
                float t = t0 + f / (float)FPS;
                RenderFrame(life, cam, smr, t, $"{dir}/frame_{f:D4}.jpg");
                if ((f & 31) == 0) EditorUtility.DisplayProgressBar("Anya life", $"frame {f}/{n}", f / (float)n);
            }
            EditorUtility.ClearProgressBar();
            Teardown(smr, head, rest, cam);
            Debug.Log($"[AnyaLife] rendered {n} frames @ {FPS}fps -> {dir}/frame_####.jpg (t0={t0:F2})");
        }

        // ---- pick a start time so a Duchenne smile PEAKS ~4 s into the clip. The smile envelope peaks
        // ~1.2 s after its event start, so offset the clip by (4 - 1.2) = 2.8 s before that start.
        static float SmileCentredStart()
        {
            for (float tt = 0.1f; tt < 80f; tt += 0.2f)
            {
                AnyaLifeLayer.EventAt(6101, tt, 6f, 13f, out float ts, out int idx);
                if (ts < 0.2f && ts < 900f && idx >= 0) return Mathf.Max(0f, (tt - ts) - 2.8f);
            }
            return 0f;
        }

        static bool Setup(out AnyaLifeLayer life, out Camera cam, out SkinnedMeshRenderer smr, out Transform head, out Quaternion rest)
        {
            life = null; cam = null; head = null; rest = Quaternion.identity;
            smr = Object.FindObjectOfType<SkinnedMeshRenderer>();
            if (smr == null || smr.sharedMesh == null)
            { Debug.LogError("[AnyaLife] no SkinnedMeshRenderer in the open scene — open AnyaFacePreview first"); return false; }

            smr.updateWhenOffscreen = true;
            smr.forceMatrixRecalculationPerRender = true;   // recompute bone matrices each render (head bone)
            // find the head bone only to restore it afterwards; the life layer owns it during render
            foreach (var t in smr.transform.root.GetComponentsInChildren<Transform>(true))
                if (t.name == "Bip01 Head") { head = t; break; }
            if (head != null) rest = head.rotation;

            life = new AnyaLifeLayer();
            life.Init(smr);

            cam = Camera.main;
            if (cam == null) cam = Object.FindObjectOfType<Camera>();
            if (cam == null)
            {
                var b = smr.bounds;
                float faceY = Mathf.Lerp(b.center.y, b.max.y, 0.72f);
                var go = new GameObject("__lifeCam", typeof(Camera));
                cam = go.GetComponent<Camera>();
                cam.transform.position = new Vector3(0f, faceY + 0.02f, 0.72f);
                cam.transform.LookAt(new Vector3(0f, faceY, 0f));
                cam.fieldOfView = 34f;
            }
            cam.clearFlags = CameraClearFlags.SolidColor;
            if (cam.backgroundColor.maxColorComponent > 0.4f) cam.backgroundColor = new Color(0.05f, 0.055f, 0.07f);
            cam.nearClipPlane = 0.03f;
            cam.allowMSAA = true;
            return true;
        }

        static void RenderFrame(AnyaLifeLayer life, Camera cam, SkinnedMeshRenderer smr, float t, string path)
        {
            life.Evaluate(t);
            var forceBake = new Mesh();
            smr.BakeMesh(forceBake);        // force synchronous skin+blendshape compute in edit mode
            Object.DestroyImmediate(forceBake);

            var msaa = new RenderTexture(W, H, 24) { antiAliasing = 8 };
            msaa.Create();
            var resolve = new RenderTexture(W, H, 0);
            resolve.Create();

            var prevRT = cam.targetTexture;
            cam.targetTexture = msaa;
            cam.Render();
            Graphics.Blit(msaa, resolve);
            RenderTexture.active = resolve;

            var tex = new Texture2D(W, H, TextureFormat.RGB24, false);
            tex.ReadPixels(new Rect(0, 0, W, H), 0, 0);
            tex.Apply();
            File.WriteAllBytes(path, tex.EncodeToJPG(82));

            cam.targetTexture = prevRT;
            RenderTexture.active = null;
            Object.DestroyImmediate(tex);
            msaa.Release(); Object.DestroyImmediate(msaa);
            resolve.Release(); Object.DestroyImmediate(resolve);
        }

        static void Teardown(SkinnedMeshRenderer smr, Transform head, Quaternion rest, Camera cam)
        {
            var m = smr.sharedMesh;
            for (int i = 0; i < m.blendShapeCount; i++) smr.SetBlendShapeWeight(i, 0f);
            if (head != null) head.rotation = rest;
            if (cam != null && cam.name == "__lifeCam") Object.DestroyImmediate(cam.gameObject);
        }
    }
}
#endif
