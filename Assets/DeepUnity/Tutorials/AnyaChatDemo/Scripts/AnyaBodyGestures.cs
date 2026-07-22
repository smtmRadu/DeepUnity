using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// Drives Anya's BODY through the Animator the scene builder wires up, layered UNDER the
    /// face/head stack: the Animator poses the skeleton in the animation pass, then this component
    /// re-owns the ARMS in LateUpdate (pins them to the classic lowered rest — the user turned the
    /// hand/gesture animations off), <see cref="AnyaBehaviourIdle"/> (order 10) re-owns the head
    /// relative to the animated torso, and FaceSync (order 100) blends the mouth last.
    ///
    /// BREATHING: the Animator's single "Idle" state is a 1D blend (param "Breath") between a
    /// frozen neutral pose and the full Idle_Loop — <see cref="breathingAmount"/> scales the whole
    /// body-motion amplitude, so the shoulders move a LITTLE (user: keep breathing, much smaller).
    ///
    /// GESTURES (currently disabled — user: "turn off the hands animation"): the crossfade logic
    /// for Talking/Think/LookAtHands states is kept behind <see cref="enableBodyGestures"/> for a
    /// later re-enable; it also needs GESTURE_STATES=true in AnyaChatDemoBuilder so the states
    /// exist in the controller. The up-left "recalling" GAZE during LLM thinking is face-side
    /// (AnyaLookAwayBehaviour) and stays on regardless.
    /// </summary>
    [DefaultExecutionOrder(5)]
    public class AnyaBodyGestures : MonoBehaviour
    {
        [Header("Breathing")]
        [Tooltip("0..1 amplitude of the idle body motion: blend between a frozen neutral pose (0) and the full captured Idle_Loop (1). ~0.25-0.3 = subtle, visible shoulder breathing.")]
        [SerializeField, Range(0f, 1f)] float breathingAmount = 0.28f;

        [Header("Arms")]
        [Tooltip("Re-own the arms every frame AFTER the Animator: swing each upper arm so it points down in the classic rest pose (same target as AnyaBodyPose.LowerArms). Shoulders still carry the breathing; forearms/hands hang still.")]
        [SerializeField] bool pinArmsDown = true;

        [Header("Gestures (user-disabled: hand animations off)")]
        [Tooltip("Master switch for the body gesture crossfades (Talking/Think/LookAtHands). Requires GESTURE_STATES=true in AnyaChatDemoBuilder so the states exist. Kept for later re-enable.")]
        [SerializeField] bool enableBodyGestures = false;
        [SerializeField] float intoTalking = 0.45f;
        [SerializeField] float intoThink = 0.5f;
        [SerializeField] float backToIdle = 0.6f;
        [Tooltip("SpeakBlend above this starts the Talking gestures.")]
        [SerializeField] float talkOn = 0.30f;
        [Tooltip("SpeakBlend below this (after talking) returns the body to Idle — the gap prevents flicker on clause pauses.")]
        [SerializeField] float talkOff = 0.10f;
        [Tooltip("Deterministic (hash-scheduled) gap range in seconds between look-at-hands fidgets.")]
        [SerializeField] float gestureMinGap = 25f;
        [SerializeField] float gestureMaxGap = 60f;
        [Tooltip("Only fidget after being quietly idle (no speech/thought) at least this long.")]
        [SerializeField] float quietBefore = 6f;

        public const string StIdle = "Idle", StTalking = "Talking", StThink = "Think", StLookHands = "LookAtHands";
        static readonly int BreathParam = Animator.StringToHash("Breath");

        Animator anim;
        AnyaBehaviourIdle face;
        string current = StIdle;
        float t0, idleSince;
        bool talking;
        bool hasThink, hasLookHands;
        Transform lUpper, lFore, rUpper, rFore;

        void Start()
        {
            anim = GetComponent<Animator>();
            face = GetComponent<AnyaBehaviourIdle>();
            if (anim == null || anim.runtimeAnimatorController == null) { enabled = false; return; }
            anim.applyRootMotion = false;   // fixed portrait framing — never walk out of frame
            hasThink = anim.HasState(0, Animator.StringToHash(StThink));
            hasLookHands = anim.HasState(0, Animator.StringToHash(StLookHands));
            lUpper = FindBone("Bip01 L UpperArm"); lFore = FindBone("Bip01 L Forearm");
            rUpper = FindBone("Bip01 R UpperArm"); rFore = FindBone("Bip01 R Forearm");
            t0 = Time.time;
            idleSince = Time.time;
        }

        void Update()
        {
            // the breathing dial — scales the Idle blend every frame so it's live-tunable
            anim.SetFloat(BreathParam, breathingAmount);

            if (!enableBodyGestures) return;

            float speak = face != null ? face.SpeakBlend : 0f;
            float think = face != null ? face.ThinkBlend : 0f;

            // hysteresis so the body doesn't flicker Idle<->Talking across clause gaps
            if (talking) { if (speak < talkOff) talking = false; }
            else if (speak > talkOn) talking = true;

            string want;
            if (hasThink && think > 0.5f) want = StThink;         // LLM composing -> thinking pose
            else if (talking) want = StTalking;                   // audible speech -> explaining hands
            else want = StIdle;

            if (want != StIdle) idleSince = Time.time;

            // a look-at-hands one-shot in progress: let it play out (speech/thought interrupts it)
            if (current == StLookHands && want == StIdle)
            {
                var st = anim.GetCurrentAnimatorStateInfo(0);
                if (!st.IsName(StLookHands) || st.normalizedTime < 0.85f) return;
            }

            // rare deterministic fidget: glance toward her own hands — never during/near speech
            if (want == StIdle && current == StIdle && hasLookHands)
            {
                AnyaFaceRig.EventAt(11207, Time.time - t0, gestureMinGap, gestureMaxGap, out float ts, out int gi);
                if (gi >= 0 && ts < 0.12f && Time.time - idleSince > quietBefore)
                {
                    Fade(StLookHands, 0.5f);
                    return;
                }
            }

            if (want != current)
                Fade(want, want == StTalking ? intoTalking : want == StThink ? intoThink : backToIdle);
        }

        // Pin the arms AFTER the Animator wrote the skeleton (any LateUpdate runs post-animation;
        // order 5 = before the face stack, which never touches arms). Same axis-agnostic swing as
        // the old edit-time AnyaBodyPose.LowerArms: rotate each upper arm so shoulder->elbow points
        // down (a hint forward + outward). The shoulder/clavicle keeps the (reduced) breathing —
        // the arm DIRECTION stays constant, so forearms/hands hang down and still.
        void LateUpdate()
        {
            if (!pinArmsDown) return;
            Pin(lUpper, lFore, transform.rotation * new Vector3(-0.16f, -1f, 0.10f));
            Pin(rUpper, rFore, transform.rotation * new Vector3(0.16f, -1f, 0.10f));
        }

        static void Pin(Transform upper, Transform elbow, Vector3 targetDir)
        {
            if (upper == null || elbow == null) return;
            Vector3 cur = (elbow.position - upper.position).normalized;
            upper.rotation = Quaternion.FromToRotation(cur, targetDir.normalized) * upper.rotation;
        }

        void Fade(string state, float seconds)
        {
            anim.CrossFadeInFixedTime(state, seconds, 0);
            current = state;
        }

        Transform FindBone(string name)
        {
            foreach (var t in GetComponentsInChildren<Transform>(true))
                if (t.name == name) return t;
            return null;
        }
    }
}
