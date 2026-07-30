using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Souls-style dialogue NPC backed by a local full-GPU LLM (Qwen3.5-0.8B or Gemma3-270M,
    /// selectable in the inspector together with the weight quantization).
    /// Walk into the trigger and press I: the camera blends to a fixed dialogue framing and the
    /// chat panel slides in from the right. Escape (or the Leave button) closes the dialogue at
    /// any time, even mid-reply. The 3D twin of the 2D ChatDemo's NPCInteractor2D.
    ///
    /// All conversation logic (model load, prefetch zone, TTS fan-out, conversation modes) lives
    /// in <see cref="NPCChatBase"/> — this class keeps only the 3D presentation: the souls camera
    /// blend to a dialogue point, the head-nod layered in LateUpdate, the player typing pose, and
    /// the 3D trigger colliders.
    /// </summary>
    public class NPCInteractor3D : NPCChatBase
    {
        [SerializeField] private Transform dialogueCameraPoint;    // fixed viewpoint framing the NPC

        [ViewOnly, SerializeField] private SoulsPlayerController player;
        private Animator npcAnimator;
        private Transform npcHead;
        private float nodWeight;

        private float lastTypeTime = -10f;
        private bool playerTypingPose;

        protected override KeyCode InteractKey => KeyCode.I;
        protected override bool PlayerReady => player != null && !player.IsBusy;
        protected override float DialogueOpenDelay => player.cam.TransitionDuration + 0.01f;

        protected override void Start()
        {
            // generic rigs (Rogue beggar, Wizard witch) have no humanoid avatar — GetBoneTransform
            // THROWS on a null avatar (it doesn't return null), which would abort base.Start()
            // and take the whole NPC (voice included) down with it
            npcAnimator = GetComponentInChildren<Animator>();
            if (npcAnimator != null && npcAnimator.avatar != null && npcAnimator.isHuman)
                npcHead = npcAnimator.GetBoneTransform(HumanBodyBones.Head);

            if (chatWindow != null && chatWindow.InputField != null)
                chatWindow.InputField.onValueChanged.AddListener(_ => lastTypeTime = Time.time);

            base.Start();   // prewarm, voice, prompt hide, window title, player zone transform
        }

        // Voice comes from the NPC: spatial audio (the voice component adds the AudioSource).
        protected override void ConfigureVoiceAudioSource(AudioSource src)
        {
            src.spatialBlend = 1f;           // 3D: voice is heard AT the NPC
            src.minDistance = 2f;
            src.maxDistance = 20f;
            src.rolloffMode = AudioRolloffMode.Linear;
        }

        // LLM+TTS mode: the talking gesture follows the AUDIO (ring actually audible), not the
        // LLM's token stream — he keeps talking after the window closes too. Text-only mode:
        // it follows the reply stream (state).
        protected override void OnTalkingChanged(bool talking)
            => PlayNPCAnimation(talking ? "Talking" : "Idle");

        protected override void Update()
        {
            base.Update();   // interact key, talk-anim watch, prefetch zone, Escape

            // mirror the NPC's talking gesture on the player while they are actually typing;
            // the pose lingers ~1.6 s after the last keystroke so pauses don't snap him out of it
            bool typing = state == NPCState.WaitingInInteraction
                       && chatWindow != null && chatWindow.InputField != null
                       && chatWindow.InputField.isFocused
                       && Time.time - lastTypeTime < 1.6f;
            if (typing != playerTypingPose && player != null)
            {
                playerTypingPose = typing;
                player.PlayDialoguePose(typing);
            }
        }

        // gentle head nod layered on top of the talking animation while the reply streams in
        // (LateUpdate runs after the Animator writes the pose, so the offset survives)
        private void LateUpdate()
        {
            if (npcHead == null) return;
            bool nodNow = speakReplies && kkVoice != null ? TalkAnimActive
                                                          : state == NPCState.TalkingInInteraction;
            nodWeight = Mathf.MoveTowards(nodWeight, nodNow ? 1f : 0f, Time.deltaTime * 3f);
            if (nodWeight > 0.001f)
                npcHead.localRotation *= Quaternion.Euler(Mathf.Sin(Time.time * 5.5f) * 8f * nodWeight, 0f, 0f);
        }

        protected override void OnInteractionStarted()
        {
            player.EnterInteractiveMode();

            // face each other
            player.FaceTowards(transform.position);
            Vector3 toPlayer = player.transform.position - transform.position;
            toPlayer.y = 0f;
            if (toPlayer.sqrMagnitude > 1e-4f)
                transform.rotation = Quaternion.LookRotation(toPlayer.normalized);

            // over-the-shoulder 3/4 framing computed from where the player actually stands;
            // the NPC sits slightly right of the camera axis so the chat panel (right side)
            // doesn't cover him
            Vector3 headPos = transform.position + Vector3.up * 1.45f;
            Vector3 side = Quaternion.Euler(0f, -38f, 0f) * toPlayer.normalized;
            Vector3 camPos = transform.position + side * 2.7f + Vector3.up * 1.6f;
            dialogueCameraPoint.position = camPos;
            // +yaw pushes the NPC left of frame center, clear of the right-docked chat panel
            dialogueCameraPoint.rotation = Quaternion.LookRotation((headPos - camPos).normalized) * Quaternion.Euler(0f, 17f, 0f);

            player.cam.MoveToInteraction(dialogueCameraPoint);
        }

        protected override void OnInteractionClosed(bool interrupted)
        {
            PlayNPCAnimation("Idle");   // gesture settles even if spatial speech finishes the sentence

            if (player != null)
            {
                player.cam.MoveToDefault();
                player.ExitInteractiveMode();
                if (interactPrompt != null) interactPrompt.Show();   // still in range
            }
        }

        private void PlayNPCAnimation(string stateName)
        {
            if (npcAnimator != null)
                npcAnimator.CrossFadeInFixedTime(stateName, 0.25f, 0);
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Player"))
            {
                player = other.GetComponent<SoulsPlayerController>();
                OnPlayerContact();   // prompt + contact loading (when no prefetch zone)
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.CompareTag("Player"))
            {
                player = null;
                OnPlayerLeft();
            }
        }
    }
}
