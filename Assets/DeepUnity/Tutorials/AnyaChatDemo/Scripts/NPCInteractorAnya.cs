using UnityEngine;
using DeepUnity.Tutorials.ChatDemo2D;   // reuses ChatWindow2D (the bottom chat panel)

namespace DeepUnity.Tutorials.AnyaChatDemo
{
    /// <summary>
    /// A talking-head NPC: no player, no movement, no interaction zones. The chat auto-opens the
    /// moment the scene plays and stays open — you type, Anya answers with a local full-GPU LLM
    /// (Qwen3.5) and speaks the reply with pocket-tts; uLipSync (added separately) drives her viseme
    /// blendshapes from that audio. Everything about the LLM/TTS/streaming lives in
    /// <see cref="NPCChatBase"/>; this subclass only supplies the trivial presentation hooks and the
    /// auto-open. Model / quant / persona / TTS are set on the inherited inspector fields by the
    /// scene builder.
    /// </summary>
    public class NPCInteractorAnya : NPCChatBase
    {
        [SerializeField] private ChatWindow2D chatWindow;
        private bool opened;

        protected override INPCChatWindow Window => chatWindow;
        protected override KeyCode InteractKey => KeyCode.None;   // no walk-up; we auto-open
        protected override bool PlayerReady => true;
        protected override float DialogueOpenDelay => 0f;

        // she speaks through pocket-tts regardless of the inherited ttsModel field
        protected override TtsModel EffectiveTtsModel => TtsModel.PocketTTS;

        protected override void Update()
        {
            // open the conversation once, on the first idle frame after load
            if (!opened && state == NPCState.Idle && chatWindow != null)
            {
                opened = true;
                StartInteraction();
            }
            base.Update();
        }

        // face-cam: the voice is always "in front of" the camera → flat 2D audio, full volume
        protected override void ConfigureVoiceAudioSource(AudioSource src)
        {
            src.spatialBlend = 0f;
            src.playOnAwake = false;
        }

        // static portrait camera — nothing to move on open/close. BUT: this always-on talking head has
        // no walk-up zone, and the zone is normally what streams the pocket-tts weights. So kick the
        // voice load (weights + kernels) ourselves the moment the chat opens, or she never gets a voice.
        protected override void OnInteractionStarted()
        {
            pkVoice?.PrewarmKernels();
            pkVoice?.PrefetchNow();
        }

        protected override void OnInteractionClosed(bool interrupted)
        {
            // she is the whole demo — if the chat ever closes, re-arm so it reopens next frame
            opened = false;
        }
    }
}
