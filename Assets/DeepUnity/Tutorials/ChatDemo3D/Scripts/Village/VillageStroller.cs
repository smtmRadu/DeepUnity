using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// A walking, gossiping villager: a full dialogue NPC (VillageInteractor → NPCChatBase, so
    /// E opens a real LLM conversation) that also strolls a loop with a partner and speaks
    /// AMBIENT lines through its own pocket-tts voice while walking. The ambient path uses the
    /// voice component directly (<see cref="PocketTTSModeling.PocketTTSVoice.Say(string)"/>) and
    /// never touches the conversation machinery — NPCChatBase's own clause handler already
    /// ignores clauses while the NPC is Idle, so banter and dialogue coexist on one voice.
    /// </summary>
    public class VillageStroller : VillageInteractor
    {
        [Tooltip("The stroll group this villager walks with (pauses the whole group on interaction).")]
        [SerializeField] private VillageStrollGroup group;

        Animator modelAnimator;
        bool walkAnimOn;

        protected override void Start()
        {
            modelAnimator = GetComponentInChildren<Animator>();
            base.Start();
        }

        /// <summary>The pocket voice, once NPCChatBase has built it (Start). Null before that.</summary>
        public PocketTTSModeling.PocketTTSVoice Voice => pkVoice;

        public bool VoiceReady => pkVoice != null && pkVoice.IsReady;

        public bool VoiceBusy => pkVoice != null && (pkVoice.HasPendingSpeech || pkVoice.IsAudioPlaying);

        /// <summary>Speak a scripted line out loud (no window, no LLM). No-op mid-conversation.</summary>
        public void SpeakAmbient(string line)
        {
            if (state != NPCState.Idle || pkVoice == null) return;
            pkVoice.FeedText(line);
            pkVoice.FlushText();
        }

        /// <summary>Cut whatever ambient line is still sounding (dialogue is about to start).</summary>
        public void CutAmbientSpeech()
        {
            if (pkVoice != null && (pkVoice.HasPendingSpeech || pkVoice.IsAudioPlaying))
                pkVoice.FadeOutAndStop(0.25f);
        }

        /// <summary>Walk-cycle control for the stroll group. Idempotent per state.</summary>
        public void PlayWalkAnim(bool walking)
        {
            if (walking == walkAnimOn) return;
            walkAnimOn = walking;
            if (modelAnimator != null)
                modelAnimator.CrossFadeInFixedTime(walking ? "Walk" : "Idle", 0.25f, 0);
        }

        // While strolling, the talking gesture must NOT interrupt the walk cycle — the base
        // implementation crossfades to a standing "Talking" state the moment the voice becomes
        // audible, which would plant the pair mid-street every time a banter line starts. The
        // gesture still runs whenever the group is stopped (dialogue, or the partner finishing
        // a line while paused).
        protected override void OnTalkingChanged(bool talking)
        {
            if (state == NPCState.Idle && group != null && group.IsWalking)
            {
                walkAnimOn = true;   // stay in Walk; keep the flag honest for PlayWalkAnim
                return;
            }
            walkAnimOn = false;      // base is about to crossfade to Talking/Idle
            base.OnTalkingChanged(talking);
        }

        protected override void OnInteractionStarted()
        {
            base.OnInteractionStarted();          // camera framing + mutual facing
            group?.OnMemberInteractionStarted(this);
        }

        protected override void OnInteractionClosed(bool interrupted)
        {
            base.OnInteractionClosed(interrupted);
            group?.OnMemberInteractionClosed(this);
        }
    }
}
