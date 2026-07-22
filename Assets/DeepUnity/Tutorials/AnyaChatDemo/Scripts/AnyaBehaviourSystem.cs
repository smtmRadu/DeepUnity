using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Per-frame context handed to every <see cref="AnyaBehaviour"/>.
    /// <c>t</c> is the deterministic idle clock, <c>speak</c> is the runtime speech-lock blend
    /// (0 = idle, 1 = she is audibly talking: gaze and head must stay on the camera), and
    /// <c>camYaw</c>/<c>camPitch</c> are the head angles (degrees, character-space) that would aim
    /// the head exactly at the camera — the anchor of the whole "looking point" model.
    /// </summary>
    public struct AnyaIdleFrame
    {
        public float t;          // deterministic idle clock (seconds since enable)
        public float speak;      // 0..1 smoothed "she is speaking" blend — scales look-aways/gestures to 0
        public float think;      // 0..1 smoothed "the LLM is composing a reply" blend — recall gaze up-left
        public float camYaw;     // deg of yaw that points the head at the camera (+ = her right / world +X)
        public float camPitch;   // deg of pitch that points the head at the camera (+ = chin down)
    }

    /// <summary>
    /// One modular idle action (gaze, blink, a head gesture, a lip micro-movement, ...).
    /// To add a NEW random behaviour: subclass this, resolve blendshape indices in
    /// <see cref="Init"/> via <see cref="AnyaFaceRig.Shape"/>, schedule discrete events in
    /// <see cref="Evaluate"/> with <see cref="AnyaFaceRig.EventAt"/> (deterministic, non-repeating),
    /// write blendshape weights with <see cref="AnyaFaceRig.Add"/> and/or head degrees into
    /// <see cref="AnyaFaceRig.Pitch"/>/<see cref="AnyaFaceRig.Yaw"/>/<see cref="AnyaFaceRig.Roll"/>,
    /// then drop an instance into <see cref="AnyaBehaviourIdle"/> (AddBehaviour or AddDefaults).
    /// Scale anything that must not happen during speech by <c>(1 - f.speak)</c>.
    /// </summary>
    public abstract class AnyaBehaviour
    {
        /// <summary>Cache blendshape indices etc. Called once when the rig is ready.</summary>
        public virtual void Init(AnyaFaceRig rig) { }

        /// <summary>Contribute this behaviour's weights / head degrees for frame <paramref name="f"/>.</summary>
        public abstract void Evaluate(AnyaFaceRig rig, in AnyaIdleFrame f);
    }

    /// <summary>
    /// Shared face rig the behaviours compose onto each frame: an additive blendshape scratch buffer,
    /// additive head-rotation channels (degrees about the character's up/right/forward), and a couple
    /// of shared cross-behaviour channels (the look-away glance, the final gaze). BeginFrame clears,
    /// behaviours accumulate, Apply clamps + writes to the SkinnedMeshRenderer and head bone.
    /// Also hosts the deterministic scheduling helpers (Hash01 / Smooth01 / EventAt) every
    /// behaviour uses, so the whole idle stays a pure function of time.
    /// </summary>
    public class AnyaFaceRig
    {
        public SkinnedMeshRenderer Smr { get; private set; }
        public Mesh Mesh { get; private set; }
        public Transform Root { get; private set; }
        public Transform Head { get; private set; }
        public Quaternion HeadRestWorld { get; private set; }

        float[] w;   // additive blendshape scratch, cleared each BeginFrame

        // ---- shared per-frame channels (cleared each BeginFrame) --------------------------------
        /// <summary>Head rotation contributions in degrees. + pitch = chin down, + yaw = her right (world +X when facing +Z), + roll = tilt.</summary>
        public float Pitch, Yaw, Roll;
        /// <summary>Look-away glance offset in normalized eye units on the FAST saccade envelope (for the EYES only).</summary>
        public float GlanceX, GlanceY;
        /// <summary>The same glance on a SLOW eased envelope (for the head share — real head reorientation takes ~0.4-0.6 s, never saccade speed).</summary>
        public float HeadGlanceX, HeadGlanceY;
        /// <summary>Final normalized gaze the eye behaviour used this frame (for followers).</summary>
        public float GazeX, GazeY;
        /// <summary>0..1 amplitude of the strongest text-conditioned talking smile this frame
        /// (written by <see cref="AnyaTalkingExpressionsBehaviour"/>, which runs earlier in the
        /// list). The idle Duchenne smile scales by (1 - TalkSmile) so smiles never stack.</summary>
        public float TalkSmile;
        /// <summary>Fore/aft head TRANSLATION target in meters along the character's forward
        /// (+ = toward the camera). Applied around the cached rest position (never accumulates)
        /// through its own critically-damped spring, same time-constant as the rotation.</summary>
        public float Lean;

        // ---- head smoother (presentation-side state, persists across frames) ---------------------
        /// <summary>Time-constant (s) of the critically-damped spring the final head channels pass
        /// through in <see cref="Apply"/>. Globally rounds off EVERY source of head motion (glances,
        /// speech-lock glide, nods, sway) and caps angular acceleration so nothing can snap.
        /// Set per-frame by <see cref="AnyaBehaviourIdle"/> from its inspector knob.</summary>
        public float HeadSmoothTime = 0.18f;
        float sPitch, sYaw, sRoll, sLean;   // smoothed head channels
        float vPitch, vYaw, vRoll, vLean;   // spring velocities (deg/s, m/s)
        bool smootherInit;
        Quaternion headRestLocal;           // head LOCAL rest rotation (base rides the animated parent)
        Vector3 headRestLocalPos;           // head LOCAL rest position (reset each frame; lean never accumulates)
        float bodyYaw, bodyPitch;           // measured torso contribution last frame (deg, character space)

        /// <summary>Head pitch/yaw actually APPLIED to the bone last frame (post-spring), INCLUDING
        /// what the animated torso contributed. The eye behaviour compensates against these — the
        /// eyes track where the head really is, so during a glance/nod/body-sway the eyes hold the
        /// camera while the head moves (eyes lead, head follows).</summary>
        public float AppliedPitch => sPitch + bodyPitch;
        public float AppliedYaw => sYaw + bodyYaw;
        /// <summary>The torso's own contribution last frame (deg, character space). The head-aim
        /// behaviour SUBTRACTS these from its camera aim so the HEAD stays on the lens no matter
        /// what stance the body clip holds — the eyes alone cannot hide a constant clip bias.</summary>
        public float BodyYaw => bodyYaw;
        public float BodyPitch => bodyPitch;

        public bool Ready => Smr != null && Mesh != null;

        public void Init(SkinnedMeshRenderer renderer)
        {
            Smr = renderer;
            if (Smr == null) return;
            Mesh = Smr.sharedMesh;
            if (Mesh == null) { Smr = null; return; }
            w = new float[Mesh.blendShapeCount];
            Smr.updateWhenOffscreen = true;
            Root = Smr.transform.root;
            Head = FindBone(Root, "Bip01 Head");
            if (Head != null)
            {
                HeadRestWorld = Head.rotation;         // reference facing for the body-delta measure
                headRestLocal = Head.localRotation;    // rest pose RELATIVE to the (possibly animated) parent
                headRestLocalPos = Head.localPosition;
            }
        }

        /// <summary>Resolve a blendshape index by ARKit name token (Contains match). -1 when absent.</summary>
        public int Shape(string token)
        {
            for (int i = 0; i < Mesh.blendShapeCount; i++)
                if (Mesh.GetBlendShapeName(i).Contains(token)) return i;
            return -1;
        }

        public void Add(int idx, float v) { if (idx >= 0) w[idx] += v; }

        public void BeginFrame()
        {
            System.Array.Clear(w, 0, w.Length);
            Pitch = Yaw = Roll = 0f;
            GlanceX = GlanceY = 0f;
            HeadGlanceX = HeadGlanceY = 0f;
            GazeX = GazeY = 0f;
            TalkSmile = 0f;
            Lean = 0f;
        }

        /// <summary>
        /// Clamp + write the blendshape weights, then pose the head. The accumulated Pitch/Yaw/Roll
        /// are TARGETS: the bone follows them through a critically-damped spring (SmoothDamp per
        /// channel, time-constant <see cref="HeadSmoothTime"/>) so every head transition eases in
        /// and out — no source of motion can snap the neck, however fast its own envelope is.
        ///
        /// The head is posed RELATIVE to the skeleton the Animator evaluated this frame (we run in
        /// LateUpdate, after the animation pass): base = animated-parent rotation * the head's REST
        /// local rotation. The head therefore stays attached to the swaying/gesturing torso, while
        /// the clip's own head-bone track is discarded — this stack owns the head. With no Animator
        /// the parent never moves, so base == the cached rest world pose (static scenes unchanged).
        /// </summary>
        public void Apply(float dt)
        {
            for (int i = 0; i < w.Length; i++)
                Smr.SetBlendShapeWeight(i, Mathf.Clamp(w[i], 0f, 100f));
            if (Head == null) return;

            if (!smootherInit || HeadSmoothTime <= 0.001f)
            {
                sPitch = Pitch; sYaw = Yaw; sRoll = Roll; sLean = Lean;
                vPitch = vYaw = vRoll = vLean = 0f;
                smootherInit = true;
            }
            else if (dt > 0f)
            {
                sPitch = Mathf.SmoothDamp(sPitch, Pitch, ref vPitch, HeadSmoothTime, Mathf.Infinity, dt);
                sYaw = Mathf.SmoothDamp(sYaw, Yaw, ref vYaw, HeadSmoothTime, Mathf.Infinity, dt);
                sRoll = Mathf.SmoothDamp(sRoll, Roll, ref vRoll, HeadSmoothTime, Mathf.Infinity, dt);
                sLean = Mathf.SmoothDamp(sLean, Lean, ref vLean, HeadSmoothTime, Mathf.Infinity, dt);
            }

            // reset the head's LOCAL offset to rest first: undoes last frame's lean (humanoid
            // animators re-write rotations, not bone local positions — without this the lean
            // would feed back and accumulate) and yields the clean animated-chain base position.
            Head.localPosition = headRestLocalPos;
            Quaternion baseRot = Head.parent != null ? Head.parent.rotation * headRestLocal : HeadRestWorld;

            // measure the torso's contribution in character space (deg). Exposed via AppliedYaw/
            // AppliedPitch so the eyes compensate the BODY sway too and stay camera-locked.
            Quaternion bodyDelta = baseRot * Quaternion.Inverse(HeadRestWorld);
            Vector3 lf = Quaternion.Inverse(Root.rotation) * (bodyDelta * Root.forward);
            bodyYaw = Mathf.Atan2(lf.x, lf.z) * Mathf.Rad2Deg;
            bodyPitch = -Mathf.Atan2(lf.y, Mathf.Sqrt(lf.x * lf.x + lf.z * lf.z)) * Mathf.Rad2Deg;

            // deltas about the CHARACTER's axes (upright, faces +Z at rest) so pitch=nod /
            // yaw=turn / roll=tilt regardless of the bone's exported local axes
            Quaternion delta = Quaternion.AngleAxis(sYaw, Root.up)
                             * Quaternion.AngleAxis(sPitch, Root.right)
                             * Quaternion.AngleAxis(sRoll, Root.forward);
            Head.rotation = delta * baseRot;
            // fore/aft lean on top of the animated base position (local offset was reset above,
            // so this can never accumulate or drift)
            Head.position += Root.forward * sLean;
        }

        // ---- deterministic helpers ---------------------------------------------------------------
        /// <summary>Deterministic hash of an int to [0,1).</summary>
        public static float Hash01(int n)
        {
            uint x = (uint)n * 2654435761u;
            x ^= x >> 15; x *= 2246822519u; x ^= x >> 13; x *= 3266489917u; x ^= x >> 16;
            return (x & 0xFFFFFF) / 16777216f;
        }

        public static float Smooth01(float x) { x = Mathf.Clamp01(x); return x * x * (3f - 2f * x); }

        /// <summary>
        /// The discrete-event scheduler: which event of stream <paramref name="seed"/> is active at
        /// time t, and how long ago it started (tSince). Gaps are hash-drawn from [minGap,maxGap] so
        /// intervals never repeat. Before the first event: tSince = 999, idx = -1.
        /// </summary>
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

        static Transform FindBone(Transform root, string name)
        {
            foreach (var tr in root.GetComponentsInChildren<Transform>(true))
                if (tr.name == name) return tr;
            return null;
        }
    }
}
