using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Procedural "life" for an ARKit-blendshape face + head bone — the thing that makes a static
    /// mesh read as a living person instead of a mannequin. It is a pure, DETERMINISTIC function of a
    /// single time value <c>t</c> (so an edit-mode filmstrip and the runtime <see cref="AnyaFaceDemo"/>
    /// produce the exact same motion) and it is NON-REPEATING: the continuous layers (head sway,
    /// breathing, eye tremor) are sums of sines/Perlin at incommensurate frequencies, and every
    /// discrete event (blink, saccade, nod, genuine smile, brow flash) is scheduled from a hash of an
    /// event index, so no interval ever repeats.
    ///
    /// Layers, roughly in order of how much "life" each contributes:
    ///   1. eyes  — saccadic gaze (hold-then-jump fixations) + micro-tremor. Eyes are ~80% of life.
    ///   2. blinks — irregular, occasional doubles, faster-close-than-open, slight L/R offset.
    ///   3. head  — continuous sway + breathing + occasional NODS and curious TILTS; follows the eyes.
    ///   4. smile — periodic *Duchenne* smiles that bloom and fade (mouth + cheek raise + eye squint).
    ///   5. brows — occasional attentive brow flashes.
    /// All layers are additive and clamped 0..100.
    /// </summary>
    public class AnyaLifeLayer
    {
        SkinnedMeshRenderer smr;
        Mesh mesh;
        Transform head;
        Quaternion headRestWorld;
        float[] w;   // scratch blendshape weights, rebuilt + applied every Evaluate

        // resolved ARKit blendshape indices (-1 when absent)
        int blinkL, blinkR;
        int lookInL, lookInR, lookOutL, lookOutR, lookUpL, lookUpR, lookDownL, lookDownR;
        int smileL, smileR, cheekL, cheekR, squintL, squintR, wideL, wideR;
        int browInner, browOutL, browOutR, browDnL, browDnR;
        int jawOpen, lipsPart, pressL, pressR, funnel, dimpL, dimpR;

        public bool Ready => smr != null && mesh != null;

        public void Init(SkinnedMeshRenderer renderer)
        {
            smr = renderer;
            if (smr == null) return;
            mesh = smr.sharedMesh;
            if (mesh == null) { smr = null; return; }
            w = new float[mesh.blendShapeCount];
            smr.updateWhenOffscreen = true;

            head = FindBone(smr.transform.root, "Bip01 Head");
            if (head != null) headRestWorld = head.rotation;

            blinkL = Idx("EyeBlinkLeft"); blinkR = Idx("EyeBlinkRight");
            lookInL = Idx("EyeLookInLeft"); lookInR = Idx("EyeLookInRight");
            lookOutL = Idx("EyeLookOutLeft"); lookOutR = Idx("EyeLookOutRight");
            lookUpL = Idx("EyeLookUpLeft"); lookUpR = Idx("EyeLookUpRight");
            lookDownL = Idx("EyeLookDownLeft"); lookDownR = Idx("EyeLookDownRight");
            smileL = Idx("MouthSmileLeft"); smileR = Idx("MouthSmileRight");
            cheekL = Idx("CheekSquintLeft"); cheekR = Idx("CheekSquintRight");
            squintL = Idx("EyeSquintLeft"); squintR = Idx("EyeSquintRight");
            wideL = Idx("EyeWideLeft"); wideR = Idx("EyeWideRight");
            browInner = Idx("BrowInnerUp"); browOutL = Idx("BrowOuterUpLeft"); browOutR = Idx("BrowOuterUpRight");
            browDnL = Idx("BrowDownLeft"); browDnR = Idx("BrowDownRight");
            jawOpen = Idx("JawOpen"); lipsPart = Idx("LipsPart");
            pressL = Idx("MouthPressLeft"); pressR = Idx("MouthPressRight"); funnel = Idx("MouthFunnel");
            dimpL = Idx("MouthDimpleLeft"); dimpR = Idx("MouthDimpleRight");
        }

        // ---- the one entry point: pose the face + head for absolute time t (seconds) ------------
        public void Evaluate(float t)
        {
            if (!Ready) return;
            System.Array.Clear(w, 0, w.Length);

            Gaze(t, out float gx, out float gy);   // normalized -1..1 (also used to lean the head)
            Blinks(t, gx, gy);
            Smile(t);
            BrowFlash(t);
            RestMouth(t);

            for (int i = 0; i < w.Length; i++)
                smr.SetBlendShapeWeight(i, Mathf.Clamp(w[i], 0f, 100f));

            Head(t, gx, gy);
        }

        // -------------------------------------------------------------------- eyes: saccadic gaze
        void Gaze(float t, out float gx, out float gy)
        {
            // hold-then-jump fixations; a saccade is a ~55 ms fast move between held targets
            EventAt(1301, t, 0.55f, 2.4f, out float ts, out int idx);
            Vector2 cur = GazeTarget(idx);
            Vector2 prev = GazeTarget(idx - 1);
            float move = Smooth01(ts / 0.055f);
            Vector2 g = Vector2.Lerp(prev, cur, move);

            // micro-tremor during fixation so the eyes are never glassy-still
            g.x += (Mathf.PerlinNoise(t * 11.3f, 4.1f) - 0.5f) * 0.06f;
            g.y += (Mathf.PerlinNoise(t * 9.7f, 8.6f) - 0.5f) * 0.05f;

            gx = Mathf.Clamp(g.x, -1.2f, 1.2f);
            gy = Mathf.Clamp(g.y, -1.2f, 1.2f);

            const float H = 24f, V = 20f;   // max eye-look weight horizontal / vertical
            if (gx > 0) { Add(lookOutL, gx * H); Add(lookInR, gx * H); }   // gaze to her left
            else { Add(lookInL, -gx * H); Add(lookOutR, -gx * H); }        // gaze to her right
            if (gy > 0) { Add(lookUpL, gy * V); Add(lookUpR, gy * V); }
            else { Add(lookDownL, -gy * V); Add(lookDownR, -gy * V); }
        }

        // a fixation point (normalized), pure function of its index; occasional larger look-away
        Vector2 GazeTarget(int idx)
        {
            if (idx < 0) return Vector2.zero;
            float x = Hash01(9001 + idx * 13) * 2f - 1f;
            float y = (Hash01(9002 + idx * 13) * 2f - 1f) * 0.7f;   // less vertical range than horizontal
            float big = Hash01(9003 + idx * 13) < 0.16f ? 1.7f : 0.75f;
            return new Vector2(x, y) * big;
        }

        // -------------------------------------------------------------------- blinks
        void Blinks(float t, float gx, float gy)
        {
            EventAt(2207, t, 2.4f, 5.6f, out float ts, out int idx);
            float amt = BlinkCurve(ts);
            if (Hash01(3300 + idx) < 0.28f) amt = Mathf.Max(amt, BlinkCurve(ts - 0.20f));   // double-blink

            // ~30% of big gaze shifts carry a blink (natural gaze-evoked blink)
            EventAt(1301, t, 0.55f, 2.4f, out float sts, out int sidx);
            if (GazeTarget(sidx).sqrMagnitude > 1.1f && Hash01(4400 + sidx) < 0.3f)
                amt = Mathf.Max(amt, BlinkCurve(sts - 0.02f));

            float lead = Hash01(5500 + idx) * 0.02f;               // tiny L/R timing asymmetry
            Add(blinkL, BlinkCurve(ts - lead) < amt ? amt * 100f : BlinkCurve(ts - lead) * 100f);
            Add(blinkR, amt * 100f);
        }

        // faster close than open; 0 outside the blink window
        static float BlinkCurve(float ts)
        {
            const float dur = 0.14f;
            if (ts < 0f || ts > dur) return 0f;
            float x = ts / dur;
            float p = x < 0.4f ? (x / 0.4f) : 1f - (x - 0.4f) / 0.6f;
            return Smooth01(p);
        }

        // -------------------------------------------------------------------- genuine (Duchenne) smile
        void Smile(float t)
        {
            EventAt(6101, t, 6f, 13f, out float ts, out int idx);
            if (ts > 3.2f || ts >= 900f) return;
            // envelope: ramp 0.55s -> hold -> decay 1.15s
            float env;
            if (ts < 0.55f) env = Smooth01(ts / 0.55f);
            else if (ts < 2.05f) env = 1f;
            else env = 1f - Smooth01((ts - 2.05f) / 1.15f);
            float warmth = 0.7f + 0.3f * Hash01(6200 + idx);       // vary intensity per smile
            env *= warmth;
            float asym = 0.88f + 0.12f * Hash01(6300 + idx);       // one corner leads slightly

            Add(smileL, env * 62f);
            Add(smileR, env * 62f * asym);
            Add(cheekL, env * 46f);                                 // cheek raise ...
            Add(cheekR, env * 46f * asym);
            Add(squintL, env * 24f);                                // ... and eye squint = the Duchenne marker
            Add(squintR, env * 24f * asym);
            Add(browInner, env * 8f);
            if (dimpL >= 0) Add(dimpL, env * 12f);
            if (dimpR >= 0) Add(dimpR, env * 12f);
            Add(jawOpen, env * 6f * Hash01(6400 + idx));            // occasional warm open-mouth smile
        }

        // -------------------------------------------------------------------- attentive brow flash
        void BrowFlash(float t)
        {
            EventAt(7103, t, 3.6f, 8.5f, out float ts, out int idx);
            if (ts > 0.7f || ts >= 900f) return;
            float e = Mathf.Sin(Mathf.Clamp01(ts / 0.7f) * Mathf.PI);   // quick up-down
            float amp = 12f + 10f * Hash01(7200 + idx);
            Add(browInner, e * amp);
            Add(browOutL, e * amp * 0.8f);
            Add(browOutR, e * amp * 0.8f);
            Add(wideL, e * amp * 0.35f);
            Add(wideR, e * amp * 0.35f);
        }

        // -------------------------------------------------------------------- lips at rest
        void RestMouth(float t)
        {
            // lips slightly parted + breathing-slow variation so the mouth is never a hard dead seam
            float part = 2.2f + 1.4f * Mathf.PerlinNoise(t * 0.35f, 2.7f);
            if (lipsPart >= 0) Add(lipsPart, part); else Add(jawOpen, part * 0.6f);
            // very occasional lip-press / swallow micro-gesture
            EventAt(8104, t, 7f, 16f, out float ts, out int idx);
            if (ts < 0.5f && ts < 900f)
            {
                float e = Mathf.Sin(Mathf.Clamp01(ts / 0.5f) * Mathf.PI) * (10f + 8f * Hash01(8200 + idx));
                Add(pressL, e); Add(pressR, e);
            }
        }

        // -------------------------------------------------------------------- head: sway + breath + nods
        void Head(float t, float gx, float gy)
        {
            if (head == null) return;

            // continuous idle sway (incommensurate Perlin) — degrees
            float pitch = (Mathf.PerlinNoise(t * 0.13f, 0.0f) - 0.5f) * 3.0f;
            float yaw = (Mathf.PerlinNoise(t * 0.11f, 5.0f) - 0.5f) * 3.4f;
            float roll = (Mathf.PerlinNoise(t * 0.09f, 9.0f) - 0.5f) * 2.2f;

            // breathing ~0.25 Hz: a gentle vertical head bob expressed as a small pitch oscillation
            pitch += Mathf.Sin(t * 2f * Mathf.PI * 0.25f) * 0.5f;

            // the head leans a little toward where the eyes are looking (VOR-like)
            yaw += -gx * 3.2f;
            pitch += -gy * 2.0f;

            // occasional NOD (chin dip + slight overshoot) or curious TILT
            EventAt(9107, t, 4.5f, 10f, out float ts, out int idx);
            if (ts < 900f)
            {
                bool tilt = Hash01(9300 + idx) < 0.4f;
                if (!tilt && ts < 1.0f)
                {
                    float e = Mathf.Sin(Mathf.Clamp01(ts / 1.0f) * Mathf.PI);
                    float overshoot = ts > 0.7f ? -0.25f * Mathf.Sin((ts - 0.7f) / 0.3f * Mathf.PI) : 0f;
                    pitch += (e + overshoot) * (5.5f + 2.5f * Hash01(9400 + idx));
                }
                else if (tilt && ts < 1.6f)
                {
                    float e = Mathf.Sin(Mathf.Clamp01(ts / 1.6f) * Mathf.PI);
                    float dir = Hash01(9500 + idx) < 0.5f ? 1f : -1f;
                    roll += e * dir * (4.5f + 2.5f * Hash01(9600 + idx));
                }
            }

            // apply about WORLD axes (character faces +Z, upright) so pitch=nod / yaw=turn / roll=tilt
            // regardless of the bone's exported local axes
            Quaternion delta = Quaternion.AngleAxis(yaw, Vector3.up)
                             * Quaternion.AngleAxis(pitch, Vector3.right)
                             * Quaternion.AngleAxis(roll, Vector3.forward);
            head.rotation = delta * headRestWorld;
        }

        // -------------------------------------------------------------------- helpers
        void Add(int idx, float v) { if (idx >= 0) w[idx] += v; }

        int Idx(string token)
        {
            for (int i = 0; i < mesh.blendShapeCount; i++)
                if (mesh.GetBlendShapeName(i).Contains(token)) return i;
            return -1;
        }

        static Transform FindBone(Transform root, string name)
        {
            foreach (var t in root.GetComponentsInChildren<Transform>(true))
                if (t.name == name) return t;
            return null;
        }

        // deterministic hash -> [0,1)
        static float Hash01(int n)
        {
            uint x = (uint)n * 2654435761u;
            x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
            return (x & 0xFFFFFF) / 16777216f;
        }

        static float Smooth01(float x) { x = Mathf.Clamp01(x); return x * x * (3f - 2f * x); }

        // the discrete-event scheduler: which event of stream `seed` is active at time t, and how long
        // ago it started (tSince). Gaps are hash-drawn from [minGap,maxGap] so intervals never repeat.
        // Before the first event, tSince = 999 (inactive), idx = -1.
        public static void EventAt(int seed, float t, float minGap, float maxGap, out float tSince, out int idx)
        {
            float clock = 0f; int k = 0; float lastStart = -1f; int lastIdx = -1;
            while (k < 100000)
            {
                float gap = Mathf.Lerp(minGap, maxGap, Hash01(seed + k * 2749));
                clock += gap;
                if (clock <= t) { lastStart = clock; lastIdx = k; k++; }
                else break;
            }
            tSince = lastIdx >= 0 ? t - lastStart : 999f;
            idx = lastIdx;
        }
    }
}
