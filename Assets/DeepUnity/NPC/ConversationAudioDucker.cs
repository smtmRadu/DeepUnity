using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Pulls the whole game's audio down while an NPC is talking, so the voice sits on top of it:
    /// music, ambience, footsteps, combat, UI clicks — everything EXCEPT the conversation itself.
    /// How far down is the NPC's call (<c>worldAudioWhileInteracting</c> on <see cref="NPCChatBase"/>,
    /// 1 = untouched, 0.5 = half); this component only decides how the move is made.
    /// <para>One per scene, on any always-active object. Zero wiring: it finds the audio itself and
    /// works out what to leave alone.</para>
    /// <para>The ramp is EXPONENTIAL — <c>scale = target^k</c> with k travelling 0→1 over
    /// <see cref="fadeSeconds"/>. Exponential in amplitude is LINEAR in decibels, the only shape that
    /// reads as an even fade to the ear (a linear amplitude ramp audibly rushes its second half), and
    /// it lands exactly on the target at the end of the window whatever the target is — the same
    /// reasoning as the voice's own exponential cut-off when the player interrupts.</para>
    /// </summary>
    [AddComponentMenu("DeepUnity/NPC/Conversation Audio Ducker")]
    [DisallowMultipleComponent]
    public class ConversationAudioDucker : MonoBehaviour
    {
        [Tooltip("Seconds for a full move from untouched to the NPC's target, and the same coming back. The ramp is exponential (linear in dB) and lands exactly on the target at the end of it.")]
        [Min(0.01f)] [SerializeField] private float fadeSeconds = 3f;

        [Tooltip("Extra roots to leave at full volume, on top of the NPCs and their dialogue windows (excluded automatically). Nothing under one of these transforms is ever touched.")]
        [SerializeField] private Transform[] alsoExclude;

        [Tooltip("How often the scene is re-scanned for AudioSources while ducked, in seconds. Sources that appear mid-conversation are picked up on the next scan.")]
        [Min(0.05f)] [SerializeField] private float rescanInterval = 0.5f;

        [Tooltip("Read-only: the multiplier currently applied to world audio. 1 while nobody is talking.")]
        [SerializeField] [ViewOnly] private float currentScale = 1f;

        /// <summary>The multiplier world audio is sitting at right now — 1 when nobody is talking.
        /// A script that drives its own AudioSource's volume every frame must multiply by this and
        /// opt out via <see cref="SelfDucking"/> instead of being ducked from the outside.</summary>
        public static float WorldScale { get; private set; } = 1f;

        // Sources whose volume is written every frame by their own script (a music cross-fade, a
        // combat swell). Ducking those from out here is a tug-of-war neither side wins: our write
        // lands, their next MoveTowards starts from the ducked value and climbs back, and the volume
        // settles wherever the two rates happen to meet. So they opt OUT and multiply by WorldScale
        // on their own line — one honest multiply beats a fight over the same field.
        static readonly HashSet<AudioSource> selfDucking = new HashSet<AudioSource>();

        /// <summary>Declare that <paramref name="src"/>'s volume is driven by its own script, which
        /// multiplies by <see cref="WorldScale"/> itself. The ducker then never writes it.</summary>
        public static void SelfDucking(AudioSource src, bool on = true)
        {
            if (src == null) return;
            if (on) selfDucking.Add(src);
            else selfDucking.Remove(src);
        }

        // One entry per ducked source. `full` is the volume it wants at full loudness and `applied`
        // is what we last wrote, which is how an external write is told apart from our own.
        class Ducked
        {
            public AudioSource src;
            public float full;
            public float applied;
        }

        readonly List<Ducked> ducked = new List<Ducked>();
        readonly HashSet<AudioSource> known = new HashSet<AudioSource>();
        float k;                    // 0 = untouched, 1 = fully at activeTarget
        float activeTarget = 1f;    // the target the current ramp is aimed at — HELD while easing back
        float nextScanAt;

        void OnEnable()
        {
            k = 0f;
            activeTarget = 1f;
            WorldScale = currentScale = 1f;
        }

        void OnDisable()
        {
            RestoreAll();
            WorldScale = currentScale = 1f;
        }

        void Update()
        {
            float target = NPCChatBase.WorldAudioTarget;   // 1 when nobody is in conversation
            if (target < 1f) activeTarget = target;
            // else: KEEP activeTarget. The conversation closing snaps the target back to 1, and
            // reading it here would snap the volume back with it — the fade back up is k travelling
            // to 0 against the target it was ducked to, not the target vanishing.

            k = Mathf.MoveTowards(k, target < 1f ? 1f : 0f, Time.unscaledDeltaTime / fadeSeconds);
            // Unscaled on purpose: a paused or slow-motion game still moves the world out from under
            // the voice at wall-clock speed.
            WorldScale = currentScale = k <= 0f ? 1f : Mathf.Pow(activeTarget, k);

            if (k <= 0f)
            {
                if (ducked.Count > 0) RestoreAll();   // all the way back up: hand the volumes over and let go
                activeTarget = 1f;
                return;
            }

            if (Time.unscaledTime >= nextScanAt)
            {
                nextScanAt = Time.unscaledTime + rescanInterval;
                Rescan();
            }
            Apply();
        }

        void Apply()
        {
            for (int i = ducked.Count - 1; i >= 0; i--)
            {
                var d = ducked[i];
                if (d.src == null)                                  // destroyed under us
                {
                    ducked.RemoveAt(i);
                    continue;
                }
                // Volume differs from what WE wrote => its own script set a fresh full-loudness
                // value this frame. Take that as the new base rather than overwriting it, so a
                // one-shot's volume change survives being ducked (and the base never drifts).
                if (!Mathf.Approximately(d.src.volume, d.applied))
                    d.full = d.src.volume;
                d.applied = d.full * WorldScale;
                d.src.volume = d.applied;
            }
        }

        void RestoreAll()
        {
            for (int i = 0; i < ducked.Count; i++)
                if (ducked[i].src != null) ducked[i].src.volume = ducked[i].full;
            ducked.Clear();
            known.Clear();
        }

        void Rescan()
        {
            var all = FindObjectsOfType<AudioSource>();
            for (int i = 0; i < all.Length; i++)
            {
                var src = all[i];
                if (!known.Add(src)) continue;
                if (selfDucking.Contains(src) || IsExcluded(src.transform))
                    continue;                                       // known, deliberately not ducked
                ducked.Add(new Ducked { src = src, full = src.volume, applied = src.volume });
            }
        }

        // The conversation is: the NPC (its voice AudioSource lives on the NPC's own object, added by
        // whichever TTS component is in use) and the dialogue window (typing ticks, button clicks).
        // Found by COMPONENT rather than by name or a wired list, so a new environment gets the right
        // exclusions for free and a renamed object cannot silently start being ducked.
        bool IsExcluded(Transform t)
        {
            if (t.GetComponentInParent<NPCChatBase>() != null) return true;
            if (t.GetComponentInParent<NPCDialogueWindow>() != null) return true;
            // ABOVE the window too, not just under it: the typing ticks and the button click live on
            // the UI canvas ROOT (deliberately — a click must outlive the panel deactivating), so the
            // parent walk alone would have ducked the conversation's own sounds along with the world.
            if (t.GetComponentInChildren<NPCDialogueWindow>(true) != null) return true;
            if (alsoExclude != null)
                for (int i = 0; i < alsoExclude.Length; i++)
                    if (alsoExclude[i] != null && t.IsChildOf(alsoExclude[i])) return true;
            return false;
        }
    }
}
