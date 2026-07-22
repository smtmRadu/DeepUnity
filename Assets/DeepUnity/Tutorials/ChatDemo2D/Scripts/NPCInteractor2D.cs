using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo2D
{
    /// <summary>
    /// Farm-village dialogue NPC backed by a local full-GPU LLM (Gemma3-270M or Qwen3.5-0.8B,
    /// selectable in the inspector together with the weight quantization) and Kokoro streaming
    /// TTS (82M non-AR, RTF ~0.3 — sentences are spoken WHILE the rest of the reply generates).
    /// Walk up to a villager and press E: the camera glides into a two-shot framing and the
    /// chat panel slides up from the bottom. Escape (or the Leave button) closes the dialogue at
    /// any time, even mid-reply. The 2D twin of ChatDemo3D's NPCInteractor3D — same state
    /// machine, same latent-loading prefetch zone, same per-token streaming TTS wiring (all of it
    /// inherited from <see cref="NPCChatBase"/>). Several NPCs can share one chat window: every
    /// listener fires on all interactors, but only the one actually in interaction reacts (AskNPC
    /// guards on its state; CloseInteraction no-ops when Idle).
    ///
    /// This class keeps only the 2D presentation: the camera two-shot glide, the
    /// CharacterAnimator2D talking bob, sprite facing, and the 2D trigger colliders. The TTS is
    /// Kokoro-only (the base ttsModel field is ignored). The prefetch zone needs no override:
    /// the base auto-detects the sprite and uses a planar circle.
    /// </summary>
    public class NPCInteractor2D : NPCChatBase
    {
        [SerializeField] private ChatWindow2D chatWindow;
        [SerializeField] private CharacterAnimator2D charAnim;  // idle bob — quickened while talking
        [Tooltip("Wired by the builder — enables the Give-items flow (hand the harvest over, get thanked, get paid).")]
        [SerializeField] private FarmingSystem farm;
        [Tooltip("Where the camera settles during dialogue, relative to the player/NPC midpoint. Negative Y drops the focus so the pair sits in the upper half of the screen, clear of the bottom chat panel.")]
        [SerializeField] private Vector2 dialogueFocusOffset = new Vector2(0f, -1.1f);

        [ViewOnly, SerializeField] private PlayerController2D player;

        protected override INPCChatWindow Window => chatWindow != null ? chatWindow : null;
        protected override KeyCode InteractKey => KeyCode.E;
        protected override bool PlayerReady => player != null && !player.IsBusy;
        protected override float DialogueOpenDelay => player.cam.TransitionDuration + 0.01f;

        // 2D demo is Kokoro-only — whatever the inherited ttsModel says, Kokoro speaks.
        protected override TtsModel EffectiveTtsModel => TtsModel.Kokoro;

        // ---------------------------------------------------------------- give-items flow
        private int pendingCoins;   // promised for the in-flight thank-you reply

        protected override void Update()
        {
            base.Update();
            // GIVE appears only while THIS NPC's dialogue is idle-waiting and the basket has
            // something in it (the Idle twin never writes, so the two NPCs can't fight over it)
            var give = chatWindow != null ? chatWindow.GiveButton : null;
            if (give != null && state != NPCState.Idle)
            {
                bool show = state == NPCState.WaitingInInteraction && farm != null && farm.HasAnyHarvest;
                if (give.gameObject.activeSelf != show) give.gameObject.SetActive(show);
            }
        }

        /// <summary>GIVE button handler (both interactors listen; the state guard picks the
        /// active one). Empties the basket, shows a flavor line, and feeds the NPC a HIDDEN
        /// prompt describing the gift — the model thanks the player in character and the coins
        /// land when the reply finishes.</summary>
        public void GiveItems()
        {
            if (state != NPCState.WaitingInInteraction || farm == null || !farm.HasAnyHarvest)
                return;
            string desc = farm.DescribeHarvest();
            int[] taken = farm.TakeAllHarvested();
            pendingCoins = farm.HarvestValue(taken);
            // visible flavor (NOT the prompt — that stays out of the dialog entirely)
            chatWindow.AddMessage("", $"<i><color=#C6C4BC>You hand over {desc}.</color></i>");
            AskNPCSilent($"[The player just handed you their fresh harvest: {desc}. Accept it, " +
                         $"thank them warmly in your own voice, and mention you are paying them " +
                         $"{pendingCoins} coins for it.]");
        }

        // the payment lands only after the thank-you finished streaming
        protected override void OnReplyFinished()
        {
            if (pendingCoins <= 0 || farm == null) return;
            farm.AddCoins(pendingCoins);
            chatWindow?.AddMessage("", $"<i><color=#E5C67B>+{pendingCoins} coins</color></i>");
            pendingCoins = 0;
        }

        // 2D audio: full volume regardless of camera distance — in a top-down farm the villager
        // you're talking to is effectively always "in front of" the player.
        protected override void ConfigureVoiceAudioSource(AudioSource src)
        {
            src.spatialBlend = 0f;
            src.playOnAwake = false;
        }

        // the villager bobs faster while "talking" — in LLM+TTS mode that follows the AUDIO
        // (ring actually audible, keeps bobbing after the text finishes streaming), in
        // text-only mode it follows the token stream
        protected override void OnTalkingChanged(bool talking)
        {
            if (charAnim != null) charAnim.talking = talking;
        }

        protected override void OnInteractionStarted()
        {
            player.EnterInteractiveMode();

            // face each other (tiny sprites only flip on X)
            player.FaceTowards(transform.position);
            charAnim?.Face(player.transform.position.x - transform.position.x);

            Vector3 focus = (player.transform.position + transform.position) * 0.5f
                          + (Vector3)dialogueFocusOffset;
            player.cam.EnterDialogue(focus);
        }

        protected override void OnInteractionClosed(bool interrupted)
        {
            // Escape mid-thank-you must not eat the payment — the vegetables are already gone
            if (pendingCoins > 0 && farm != null)
            {
                farm.AddCoins(pendingCoins);
                pendingCoins = 0;
            }
            if (chatWindow != null && chatWindow.GiveButton != null)
                chatWindow.GiveButton.gameObject.SetActive(false);

            if (player != null)
            {
                player.cam.ExitDialogue();
                player.ExitInteractiveMode();
                if (interactPrompt != null) interactPrompt.Show();   // still in range
            }
        }

        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.CompareTag("Player"))
            {
                player = other.GetComponent<PlayerController2D>();
                OnPlayerContact();   // prompt + contact loading (when no prefetch zone)
            }
        }

        private void OnTriggerExit2D(Collider2D other)
        {
            if (other.CompareTag("Player"))
            {
                player = null;
                OnPlayerLeft();
            }
        }
    }
}
