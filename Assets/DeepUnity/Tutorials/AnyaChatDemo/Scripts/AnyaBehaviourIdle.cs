using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// The generalized idle "brain": composes modular <see cref="AnyaBehaviour"/> units onto the
    /// shared <see cref="AnyaFaceRig"/> every LateUpdate. Replaces the mocap replay in the chat
    /// scene so gaze and head are camera-anchored:
    ///
    ///  - "looking point" model: eyes + head aim at Camera.main; every several seconds a
    ///    deterministic look-away glances at a nearby point (more eyes than head) and returns.
    ///  - speech lock: while the AudioSource on this GameObject (pocket-tts) is audibly playing —
    ///    the same signal <see cref="FaceSync"/> uses — a smoothed <c>speak</c> blend ramps to 1
    ///    and every behaviour scaled by (1 - speak) collapses: no look-aways, no nods/tilts;
    ///    she looks straight into the camera the whole time she talks.
    ///  - head incline is damped in <see cref="AnyaHeadMotionBehaviour"/> (small sway, subtle tilts).
    ///  - transition feel: the final head channels run through a critically-damped spring in
    ///    <see cref="AnyaFaceRig.Apply"/> (time-constant <c>headSmoothTime</c>) and the head share
    ///    of glances uses its own slow eased envelope — eyes saccade fast, the head never snaps.
    ///  - talking expressions: pocket-tts' OnClauseSpoken(text, duration) is mined for lexical
    ///    cues (see <see cref="AnyaSpeechCueDetector"/>) rendered by
    ///    <see cref="AnyaTalkingExpressionsBehaviour"/> — her face reacts to WHAT she says,
    ///    scaled by the <c>talkingExpressiveness</c> dial.
    ///
    /// Runs at execution order 10, BEFORE <see cref="FaceSync"/> (100), which re-reads the mouth
    /// weights this idle wrote and blends the viseme on top. To add a new random action, subclass
    /// <see cref="AnyaBehaviour"/> and register it in <see cref="AddDefaults"/> (or via
    /// <see cref="AddBehaviour"/> from other code).
    /// </summary>
    [DefaultExecutionOrder(10)]   // before AnyaLipSync (100): the mouth blends on top of this idle
    public class AnyaBehaviourIdle : MonoBehaviour
    {
        [Header("Speech lock (gaze+head stay on camera while she talks)")]
        [Tooltip("Output RMS above this counts as audible speech (same scale AnyaLipSync gates on).")]
        [SerializeField] float speechRms = 0.0035f;
        [Tooltip("Seconds speech stays 'on' after audio dips below the gate (bridges word gaps).")]
        [SerializeField] float speechHold = 0.4f;
        [Tooltip("Seconds to fully lock onto the camera when speech starts (smoothstep-shaped, so it eases in and out).")]
        [SerializeField] float lockRamp = 0.45f;
        [Tooltip("Seconds to release the lock after speech ends.")]
        [SerializeField] float unlockRamp = 0.9f;

        [Header("Camera aim")]
        [Tooltip("Clamp on the head aim toward the camera, deg (a stray camera can't wrench her neck).")]
        [SerializeField] float maxAimDeg = 25f;

        [Header("Head transition feel")]
        [Tooltip("Time-constant (s) of the critically-damped spring the head runs through. Rounds off every head transition (glances, nods, speech lock). Higher = lazier, lower = snappier; ~0.15-0.25 feels organic. 0 disables.")]
        [SerializeField, Range(0f, 0.5f)] float headSmoothTime = 0.18f;

        [Header("Talking expressions (text-conditioned, via pocket-tts OnClauseSpoken)")]
        [Tooltip("Global 0..1 multiplier on ALL speech-cue expressions (talking smiles, question brows, emphasis nods, sympathy...). 0 = expression-dead during speech, 1 = full.")]
        [SerializeField, Range(0f, 1f)] float talkingExpressiveness = 0.8f;

        [Header("Breathing (procedural, on the static body)")]
        [Tooltip("0..1 amplitude of the clavicle/chest breathing sinusoid. 1 = full clip-like breathing (clavicle ±4°, chest ±1.5°). 0.35 = the user-tuned default (play-tested 2026-07-22).")]
        [SerializeField, Range(0f, 1f)] float breathingAmount = 0.35f;

        readonly AnyaFaceRig rig = new AnyaFaceRig();
        readonly List<AnyaBehaviour> behaviours = new List<AnyaBehaviour>();

        /// <summary>The speech-cue renderer — external code may queue extra cues into its
        /// <see cref="AnyaTalkingExpressionsBehaviour.Cues"/> list.</summary>
        public AnyaTalkingExpressionsBehaviour TalkingExpressions { get; private set; }

        AnyaBreathingBehaviour breathing;   // procedural chest/clavicle unit; Amount = the dial above

        AudioSource src;   // added at runtime by the TTS voice component — grabbed lazily
        PocketTTSModeling.PocketTTSVoice tts;   // ditto — subscribed lazily for OnClauseSpoken
        NPCChatBase npc;   // the NPC brain on this GameObject (optional) — source of the think signal
        Camera cam;
        float t0;
        float speakRaw;    // linear 0..1 speech-lock ramp (smoothstep-shaped before use)
        float thinkRaw;    // linear 0..1 "LLM composing a reply" ramp
        float hold;        // audible-speech hold timer
        readonly float[] samp = new float[256];

        /// <summary>Smoothed 0..1 "she is audibly speaking" blend — the same signal the face uses.
        /// Read by <see cref="AnyaBodyGestures"/> to crossfade the body Idle/Talking states.</summary>
        public float SpeakBlend => AnyaFaceRig.Smooth01(speakRaw);
        /// <summary>Smoothed 0..1 "the LLM is composing, nothing audible yet" blend (the reply-
        /// latency window). Drives the recall gaze up-left and the body Think pose; mutually
        /// exclusive with <see cref="SpeakBlend"/> by construction.</summary>
        public float ThinkBlend => AnyaFaceRig.Smooth01(thinkRaw) * (1f - AnyaFaceRig.Smooth01(speakRaw));

        /// <summary>Drop in a new behaviour at runtime (initialized immediately if the rig is ready).</summary>
        public void AddBehaviour(AnyaBehaviour b)
        {
            behaviours.Add(b);
            if (rig.Ready) b.Init(rig);
        }

        void Start()
        {
            var smr = GetComponentInChildren<SkinnedMeshRenderer>();
            if (smr == null || smr.sharedMesh == null) { enabled = false; return; }
            rig.Init(smr);
            if (!rig.Ready) { enabled = false; return; }
            if (behaviours.Count == 0) AddDefaults();
            foreach (var b in behaviours) b.Init(rig);
            t0 = Time.time;
        }

        // ORDER MATTERS for the gaze chain: LookAway writes the shared glance, HeadMotion consumes
        // it (head share + camera aim), CameraGaze compensates the final head pose and adds the
        // eye share. TalkingExpressions runs BEFORE Smile so rig.TalkSmile can suppress the idle
        // Duchenne (no double-smile stacking). Everything else is order-independent additive.
        void AddDefaults()
        {
            behaviours.Add(breathing = new AnyaBreathingBehaviour());   // bones first: head/eyes then compensate on top
            behaviours.Add(new AnyaLookAwayBehaviour());
            behaviours.Add(new AnyaHeadMotionBehaviour());
            behaviours.Add(new AnyaCameraGazeBehaviour());
            behaviours.Add(TalkingExpressions = new AnyaTalkingExpressionsBehaviour());
            behaviours.Add(new AnyaBlinkBehaviour());
            behaviours.Add(new AnyaSmileBehaviour());
            behaviours.Add(new AnyaBrowFlashBehaviour());
            behaviours.Add(new AnyaRestMouthBehaviour());
        }

        void LateUpdate()
        {
            HookTts();
            UpdateSpeak();
            if (TalkingExpressions != null) TalkingExpressions.Expressiveness = talkingExpressiveness;
            if (breathing != null) breathing.Amount = breathingAmount;
            // smoothstep-shape the linear ramps so the camera-lock / recall-gaze glides ease in
            // AND out (a raw linear blend starts/stops with a visible velocity kink)
            var f = new AnyaIdleFrame { t = Time.time - t0, speak = AnyaFaceRig.Smooth01(speakRaw), think = ThinkBlend };
            ComputeCameraAim(ref f);
            rig.BeginFrame();
            for (int i = 0; i < behaviours.Count; i++)
                behaviours[i].Evaluate(rig, in f);
            rig.HeadSmoothTime = headSmoothTime;
            rig.Apply(Time.deltaTime);
        }

        // the PocketTTSVoice is added at runtime by NPCChatBase on approach — subscribe as soon as
        // it exists. NPCChatBase has its own OnClauseSpoken handler (text reveal); ours is additive.
        void HookTts()
        {
            if (tts != null) return;
            tts = GetComponent<PocketTTSModeling.PocketTTSVoice>();
            if (tts == null) return;
            tts.OnClauseSpoken -= OnClauseSpoken;   // defensive: never double-subscribe
            tts.OnClauseSpoken += OnClauseSpoken;
        }

        void OnDestroy()
        {
            if (tts != null) tts.OnClauseSpoken -= OnClauseSpoken;
        }

        // a spoken chunk's audio just STARTED playing (text, audible seconds): mine the text for
        // expression cues and schedule them along the clause on the idle clock
        void OnClauseSpoken(string text, float duration)
        {
            if (TalkingExpressions == null) return;
            AnyaSpeechCueDetector.Detect(text, Time.time - t0, duration, TalkingExpressions.Cues);
        }

        // speaking = the AudioSource is playing AND actually audible (RMS gate) — the exact signal
        // AnyaLipSync animates from, so mouth and gaze-lock always agree. A short hold bridges
        // inter-word silence; the blend ramps so the gaze glides (never snaps) to/from the camera.
        void UpdateSpeak()
        {
            if (src == null) src = GetComponent<AudioSource>();
            bool audible = false;
            if (src != null && src.isPlaying)
            {
                src.GetOutputData(samp, 0);
                float rms = 0f;
                for (int i = 0; i < samp.Length; i++) rms += samp[i] * samp[i];
                rms = Mathf.Sqrt(rms / samp.Length);
                audible = rms > speechRms;
            }
            hold = audible ? speechHold : Mathf.Max(0f, hold - Time.deltaTime);
            float target = (audible || hold > 0f) ? 1f : 0f;
            float ramp = target > speakRaw ? lockRamp : unlockRamp;
            speakRaw = Mathf.MoveTowards(speakRaw, target, Time.deltaTime / Mathf.Max(0.01f, ramp));

            // "thinking" = the NPC brain is generating the reply (TalkingInInteraction) but nothing
            // is audible yet — the reply-latency window the recall gaze / Think pose should fill.
            // Once generation finishes the state leaves TalkingInInteraction, so this never
            // lingers after the answer; while audio plays, ThinkBlend is zeroed by SpeakBlend.
            if (npc == null) npc = GetComponent<NPCChatBase>();
            bool composing = npc != null && npc.State == NPCChatBase.NPCState.TalkingInInteraction
                          && !audible && hold <= 0f;
            thinkRaw = Mathf.MoveTowards(thinkRaw, composing ? 1f : 0f, Time.deltaTime / 0.5f);
        }

        // head angles (deg, character space) that aim the head exactly at the camera. Camera dead
        // ahead (the portrait framing) => ~0/0, i.e. neutral head; robust to a moved camera.
        void ComputeCameraAim(ref AnyaIdleFrame f)
        {
            if (cam == null) cam = Camera.main;   // cheap since 2020.2; re-resolves if it appears late
            if (cam == null || rig.Head == null) { f.camYaw = 0f; f.camPitch = 0f; return; }
            Vector3 d = Quaternion.Inverse(rig.Root.rotation) * (cam.transform.position - rig.Head.position);
            if (d.sqrMagnitude < 1e-6f) { f.camYaw = 0f; f.camPitch = 0f; return; }
            f.camYaw = Mathf.Clamp(Mathf.Atan2(d.x, d.z) * Mathf.Rad2Deg, -maxAimDeg, maxAimDeg);
            f.camPitch = Mathf.Clamp(-Mathf.Atan2(d.y, Mathf.Sqrt(d.x * d.x + d.z * d.z)) * Mathf.Rad2Deg,
                                     -maxAimDeg, maxAimDeg);
        }
    }
}
