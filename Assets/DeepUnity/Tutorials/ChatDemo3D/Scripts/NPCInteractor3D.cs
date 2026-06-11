using System.Collections;
using System.Text;
using UnityEngine;

namespace DeepUnity.Tutorials.ChatDemo3D
{
    /// <summary>
    /// Souls-style dialogue NPC backed by Qwen3.5-0.8B (full local GPU inference).
    /// Walk into the trigger and press I: the camera blends to a fixed dialogue framing and the
    /// chat panel slides in from the right. Escape (or the Leave button) closes the dialogue at
    /// any time, even mid-reply. The 3D twin of the 2D ChatDemo's NPCInteractor.
    /// </summary>
    public class NPCInteractor3D : MonoBehaviour
    {
        public enum NPCState
        {
            Idle,
            PreparingForInteraction,
            WaitingInInteraction,
            TalkingInInteraction,
        }

        [SerializeField, ViewOnly] private NPCState state = NPCState.Idle;
        [SerializeField] private string npc_name = "Velmire, the Pale Herald";
        [TextArea(4, 12)]
        [SerializeField] private string system_prompt =
            "You are Velmire, the Pale Herald: a white-masked, soft-spoken emissary lingering by the gate of a ruined castle. " +
            "You greet travellers with honeyed courtesy that thinly veils mockery. You pity the player for wandering these dead " +
            "lands guideless and lordless, and you address them as 'lambkin' or 'poor wanderer'. You speak in flowery, " +
            "old-fashioned phrases, hint that you know more than you say, and never give a straight answer. " +
            "You know what waits beyond the wall of golden mist at the northern arch: the Sentinel of the Mist, a towering " +
            "hollow knight wielding a halberd, who has felled every challenger before. If asked about the mist, the arch or " +
            "the boss, you foreshadow it with morbid delight — urging the lambkin onward while clearly expecting them to die. " +
            "Stay in character at all times. Keep your replies to one to three short sentences.";

        [SerializeField] private SoulsChatWindow chatWindow;
        [SerializeField] private GameObject interactPrompt;        // screen-space "[I] Speak" box
        [SerializeField] private Transform dialogueCameraPoint;    // fixed viewpoint framing the NPC
        [SerializeField] private float temperature = 0.8f;
        [Tooltip("Keep the model resident on the GPU (~1.6 GB VRAM) after closing the chat — re-interactions skip the load entirely. Off = release on close and reload (~2-3 s) every interaction.")]
        [SerializeField] private bool keepModelLoaded = false;

        private Qwen3_5ForCausalLM llm;
        [ViewOnly, SerializeField] private SoulsPlayerController player;
        private Animator npcAnimator;
        private Coroutine dialogueCoroutine;
        private Transform npcHead;
        private float nodWeight;

        private float lastTypeTime = -10f;
        private bool playerTypingPose;

        private void Start()
        {
            npcAnimator = GetComponentInChildren<Animator>();
            if (npcAnimator != null)
                npcHead = npcAnimator.GetBoneTransform(HumanBodyBones.Head);

            if (chatWindow != null && chatWindow.InputField != null)
                chatWindow.InputField.onValueChanged.AddListener(_ => lastTypeTime = Time.time);

            // Scene-start prewarm: compiles the model's compute kernels (one per frame) and parses
            // the tokenizer in the background while the player walks around, so the dialogue
            // later opens without hitches.
            StartCoroutine(Qwen3_5ForCausalLM.Prewarm());

            if (interactPrompt != null) interactPrompt.SetActive(false);
            if (chatWindow != null) chatWindow.SetTitle(npc_name);
        }

        private void Update()
        {
            if (player != null && state == NPCState.Idle && !player.IsBusy && Input.GetKeyDown(KeyCode.I))
                StartInteraction();

            if (state != NPCState.Idle && Input.GetKeyDown(KeyCode.Escape))
                CloseInteraction();

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
            nodWeight = Mathf.MoveTowards(nodWeight, state == NPCState.TalkingInInteraction ? 1f : 0f, Time.deltaTime * 3f);
            if (nodWeight > 0.001f)
                npcHead.localRotation *= Quaternion.Euler(Mathf.Sin(Time.time * 5.5f) * 8f * nodWeight, 0f, 0f);
        }

        public void StartInteraction()
        {
            state = NPCState.PreparingForInteraction;
            if (interactPrompt != null) interactPrompt.SetActive(false);
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

            dialogueCoroutine = StartCoroutine(OpenDialogue());
        }

        private IEnumerator OpenDialogue()
        {
            player.cam.MoveToInteraction(dialogueCameraPoint);
            yield return new WaitForSeconds(player.cam.TransitionDuration + 0.01f);

            chatWindow.Open();
            chatWindow.SetInfoText("The pale figure regards you in silence...");

            if (llm == null)
                llm = new Qwen3_5ForCausalLM();   // cheap; weights stream to the GPU over the next frames

            // Waits for the weight stream, warms the kernels and caches the system prompt — all
            // budgeted per frame, so the game keeps rendering smoothly behind the dialogue.
            yield return llm.InitializeChat(system_prompt: system_prompt);

            chatWindow.SetInfoText("");
            state = NPCState.WaitingInInteraction;
            chatWindow.InputField.ActivateInputField();
            dialogueCoroutine = null;
        }

        /// <summary>Called by the Send button / submitting the input field.</summary>
        public void AskNPC()
        {
            if (chatWindow.InputField == null || string.IsNullOrWhiteSpace(chatWindow.InputField.text)
                || state != NPCState.WaitingInInteraction)
                return;

            string question = chatWindow.InputField.text;
            chatWindow.AddMessage("You", question);
            chatWindow.InputField.text = "";

            dialogueCoroutine = StartCoroutine(Talk(question));
        }

        private IEnumerator Talk(string question)
        {
            state = NPCState.TalkingInInteraction;
            chatWindow.SendButton.interactable = false;
            PlayNPCAnimation("Talking");

            bool isFirstToken = true;
            StringBuilder response = new StringBuilder();
            // base Chat defaults are neutral — pass Qwen's recommended top_k/presence preset explicitly
            yield return llm.Chat(question, max_new_tokens: 128, temperature: temperature, top_k: 20, top_p: 0.95f,
                presence_penalty: llm.Config.DefaultPresencePenalty,
                onTokenGenerated: (token) =>
                {
                    response.Append(token);
                    if (isFirstToken)
                    {
                        chatWindow.AddMessage(npc_name, response.ToString());
                        isFirstToken = false;
                    }
                    else
                    {
                        chatWindow.PopLastMessage();
                        chatWindow.AddMessage(npc_name, response.ToString());
                    }
                });

            PlayNPCAnimation("Idle");
            chatWindow.SendButton.interactable = true;
            chatWindow.InputField.ActivateInputField();
            state = NPCState.WaitingInInteraction;
            dialogueCoroutine = null;
        }

        /// <summary>Closes the dialogue from any state — Escape, the Leave button, or scripted.</summary>
        public void CloseInteraction()
        {
            // interrupting a half-written reply leaves the conversation cache mid-token,
            // so an interrupted model is always released, even with keepModelLoaded on
            bool interrupted = dialogueCoroutine != null;
            if (interrupted)
            {
                StopCoroutine(dialogueCoroutine);
                dialogueCoroutine = null;
            }

            state = NPCState.Idle;
            PlayNPCAnimation("Idle");

            if (!keepModelLoaded || interrupted)
            {
                llm?.Release();   // free the GPU buffers now (the finalizer can't call Unity APIs safely)
                llm = null;
                StartCoroutine(CollectGarbageIncremental());
            }

            chatWindow.Clear();
            chatWindow.Close();
            chatWindow.SendButton.interactable = true;

            if (player != null)
            {
                player.cam.MoveToDefault();
                player.ExitInteractiveMode();
                if (interactPrompt != null) interactPrompt.SetActive(true);   // still in range
            }
        }

        private void PlayNPCAnimation(string stateName)
        {
            if (npcAnimator != null)
                npcAnimator.CrossFadeInFixedTime(stateName, 0.25f, 0);
        }

        // Spreads the post-conversation cleanup over ~2 ms slices per frame instead of one
        // blocking GC.Collect (~400 ms). Incremental GC is enabled in Project Settings; if it
        // were disabled, CollectIncremental no-ops and the next natural collection handles it.
        private IEnumerator CollectGarbageIncremental()
        {
            while (UnityEngine.Scripting.GarbageCollector.CollectIncremental(2_000_000UL))
                yield return null;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Player"))
            {
                player = other.GetComponent<SoulsPlayerController>();
                if (interactPrompt != null && state == NPCState.Idle)
                    interactPrompt.SetActive(true);
            }
        }

        private void OnTriggerExit(Collider other)
        {
            if (other.CompareTag("Player"))
            {
                player = null;
                if (interactPrompt != null)
                    interactPrompt.SetActive(false);
            }
        }
    }
}
