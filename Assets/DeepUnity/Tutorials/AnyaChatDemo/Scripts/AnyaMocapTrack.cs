using System.IO;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// REAL facial mocap replayed on Anya's ARKit rig. The track is extracted from video of an actual
    /// person (MediaPipe FaceLandmarker → the 52 ARKit blendshape scores + head pose per frame), so
    /// blinks, gaze, micro-expressions and head motion are genuine human motion — not synthesized.
    /// Binary layout: see convert_mocap_to_bytes.py. Evaluation is a pure function of time t
    /// (frame lerp + looping), same contract as <see cref="AnyaLifeLayer"/>, so the edit-mode
    /// filmstrip and the play-mode component show identical motion.
    /// </summary>
    public class AnyaMocapTrack
    {
        public float WeightScale = 1f;    // <1 tames "exaggerated" source performances
        public float HeadScale = 0.6f;    // head motion gain (1 = as captured)
        public float Smooth = 0.35f;      // 0..1 exponential smoothing against tracker jitter

        SkinnedMeshRenderer smr;
        Transform head;
        Quaternion headRestWorld;

        float fps;
        int nShapes, nFrames;
        float[] weights;    // nFrames * nShapes, 0..1
        float[] heads;      // nFrames * 3 (pitch, yaw, roll deg), rest-relative after Init
        int[] map;          // track shape index -> mesh blendshape index (-1 = unmapped)
        float[] cur;        // smoothed current weights per mesh blendshape
        Vector3 curHead;

        public bool Ready { get; private set; }
        public float Duration => nFrames / Mathf.Max(1f, fps);

        public void Init(SkinnedMeshRenderer renderer, byte[] track)
        {
            Ready = false;
            smr = renderer;
            if (smr == null || smr.sharedMesh == null || track == null) return;
            var mesh = smr.sharedMesh;
            smr.updateWhenOffscreen = true;

            using (var r = new BinaryReader(new MemoryStream(track)))
            {
                if (r.ReadInt32() != 0x4D594E41) { Debug.LogError("[AnyaMocap] bad magic"); return; }
                r.ReadInt32();   // version
                fps = r.ReadSingle();
                nShapes = r.ReadInt32();
                map = new int[nShapes];
                for (int i = 0; i < nShapes; i++)
                {
                    string name = System.Text.Encoding.UTF8.GetString(r.ReadBytes(r.ReadInt32()));
                    map[i] = FindShape(mesh, name);
                }
                nFrames = r.ReadInt32();
                weights = new float[nFrames * nShapes];
                for (int i = 0; i < weights.Length; i++) weights[i] = r.ReadSingle();
                heads = new float[nFrames * 3];
                for (int i = 0; i < heads.Length; i++) heads[i] = r.ReadSingle();
            }

            // make head angles rest-relative: subtract the track average (the person's neutral pose
            // relative to their camera is arbitrary; we only want the MOTION)
            for (int a = 0; a < 3; a++)
            {
                float mean = 0f;
                for (int f = 0; f < nFrames; f++) mean += heads[f * 3 + a];
                mean /= nFrames;
                for (int f = 0; f < nFrames; f++) heads[f * 3 + a] -= mean;
            }

            head = FindBone(smr.transform.root, "Bip01 Head");
            if (head != null) headRestWorld = head.rotation;
            cur = new float[mesh.blendShapeCount];
            curHead = Vector3.zero;
            for (int i = 0; i < mesh.blendShapeCount; i++) smr.SetBlendShapeWeight(i, 0f);   // clear leftovers

            int mapped = 0; foreach (int m in map) if (m >= 0) mapped++;
            Debug.Log($"[AnyaMocap] track: {nFrames} frames @ {fps:F1}fps ({Duration:F1}s), {mapped}/{nShapes} shapes mapped");
            Ready = true;
        }

        // pose face + head for absolute time t; loops over the track
        public void Evaluate(float t)
        {
            if (!Ready) return;
            float ft = Mathf.Repeat(t * fps, nFrames - 1.001f);
            int f0 = (int)ft;
            float frac = ft - f0;

            // frame-lerped raw targets, then exponential smoothing (tracker de-jitter)
            float k = 1f - Mathf.Clamp01(Smooth);
            for (int s = 0; s < nShapes; s++)
            {
                int mi = map[s];
                if (mi < 0) continue;
                float w = Mathf.Lerp(weights[f0 * nShapes + s], weights[(f0 + 1) * nShapes + s], frac) * 100f * WeightScale;
                cur[mi] += (w - cur[mi]) * k;
                smr.SetBlendShapeWeight(mi, Mathf.Clamp(cur[mi], 0f, 100f));
            }

            if (head != null)
            {
                var raw = new Vector3(
                    Mathf.Lerp(heads[f0 * 3 + 0], heads[(f0 + 1) * 3 + 0], frac),
                    Mathf.Lerp(heads[f0 * 3 + 1], heads[(f0 + 1) * 3 + 1], frac),
                    Mathf.Lerp(heads[f0 * 3 + 2], heads[(f0 + 1) * 3 + 2], frac)) * HeadScale;
                curHead += (raw - curHead) * k;
                // world-axis deltas (character faces +Z): pitch=nod, yaw=turn, roll=tilt
                Quaternion delta = Quaternion.AngleAxis(-curHead.y, Vector3.up)
                                 * Quaternion.AngleAxis(-curHead.x, Vector3.right)
                                 * Quaternion.AngleAxis(-curHead.z, Vector3.forward);
                head.rotation = delta * headRestWorld;
            }
        }

        // mediapipe names are ARKit camelCase ("browDownLeft"); mesh names are "blendShape1.AK_01_BrowDownLeft"
        static int FindShape(Mesh mesh, string track)
        {
            string want = track.ToLowerInvariant();
            for (int i = 0; i < mesh.blendShapeCount; i++)
            {
                string n = mesh.GetBlendShapeName(i);
                int us = n.LastIndexOf('_');                       // strip "blendShape1.AK_01_" prefix
                string tail = (us >= 0 && us + 1 < n.Length ? n.Substring(us + 1) : n).ToLowerInvariant();
                // ARKit L/R names end with "Left"/"Right" INSIDE the tail (e.g. BrowDownLeft) — the
                // underscore strip above only removes the numeric prefix, so compare full tails
                if (tail == want) return i;
            }
            return -1;
        }

        static Transform FindBone(Transform root, string name)
        {
            foreach (var tr in root.GetComponentsInChildren<Transform>(true))
                if (tr.name == name) return tr;
            return null;
        }
    }
}
