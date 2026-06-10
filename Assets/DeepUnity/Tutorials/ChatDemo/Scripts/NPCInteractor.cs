using DeepUnity;
using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics.Eventing.Reader;
using System.Text;
using TMPro;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials.ChatDemo
{
    public class NPCInteractor : MonoBehaviour
    {
        public enum NPCState
        {
            Idle,
            PreparingForInteraction,
            WaitingInInteraction,
            TalkingInInteraction,
        }
        [SerializeField, ViewOnly] private NPCState state = NPCState.Idle;
        [SerializeField] private string npc_name = "Branik Hollowkeg";
        [SerializeField] private string system_prompt = " gruff but sharp-eyed owner of The Bent Tankard, a neutral-ground tavern where ale and rumors flow freely.\r\nDry-humored, observant, and slow to trust, he remembers every face and never shares information without a price.";
        CircleCollider2D trigger;
        TMP_Text pressIToInteractText;
        [SerializeField] ChatWindow chatWindow;
        [SerializeField] private float temperature = 0.8f;
        [Tooltip("Keep the model resident on the GPU (~1.6 GB VRAM) after closing the chat — re-interactions skip the load entirely. Off = release on close and reload (~2-3 s) every interaction.")]
        [SerializeField] private bool keepModelLoaded = false;
        private Qwen3_5ForCausalLM llm; // Qwen3.5-0.8B, full-GPU inference (streaming, hitch-free loader)


        [ViewOnly, SerializeField] KnightScript player;
        private void Start()
        {
            trigger = GetComponent<CircleCollider2D>();
            pressIToInteractText = GetComponent<TMP_Text>();

            // Scene-start prewarm: compiles the model's compute kernels (one per frame) and parses
            // the tokenizer in the background while the player walks around, so pressing "interact"
            // later loads without hitches. One call — everything else is internal to the model.
            StartCoroutine(Qwen3_5ForCausalLM.Prewarm());
        }

        private void Update()
        {
            if (player is not null && Input.GetKeyDown(KeyCode.I) && chatWindow.gameObject.activeSelf == false)
                StartInteraction();

        }
        public void StartInteraction()
        {
            Debug.Log("Interacting...");
            state = NPCState.PreparingForInteraction;
            pressIToInteractText.enabled = false;
            player.EnterInteractiveMode();

            StartCoroutine(LoadLLM());
            // pop up chatting window
        }
        IEnumerator LoadLLM()
        {
            yield return new WaitForSeconds(player.cam.TransitionDuration + 0.01f);

            chatWindow.SetInfoText("Loading LLM...");
            chatWindow.gameObject.SetActive(true);
            yield return null;

            if (llm == null)
                llm = new Qwen3_5ForCausalLM(); // cheap; weights stream to the GPU over the next frames

            chatWindow.SetInfoText($"Initializing {npc_name}...");
            // Waits for the weight stream, warms the kernels and caches the system prompt — all
            // budgeted per frame, so the game keeps rendering smoothly behind the loading text.
            yield return llm.InitializeChat(system_prompt: system_prompt);
            chatWindow.SetInfoText("");
            state = NPCState.WaitingInInteraction;
            yield return null;
        }


        public void AskNPC()
        {
            if (chatWindow.InputField == null || string.IsNullOrWhiteSpace(chatWindow.InputField.text) || state != NPCState.WaitingInInteraction)
            {
                Debug.Log("Couldn't ask");
                return;
            }


            string question = chatWindow.InputField.text;
            chatWindow.AddMessage("You", question);
            chatWindow.InputField.text = "";


            StartCoroutine(Talk(question));


        }
        IEnumerator Talk(string question)
        {
            state = NPCState.TalkingInInteraction;
            chatWindow.SendButton.GetComponent<Button>().interactable = false;

            bool is_first_token = true;
            StringBuilder response = new StringBuilder();
            yield return this.llm.Chat(question, max_new_tokens: 128, temperature:temperature, top_p:0.95f, onTokenGenerated: (x) =>
            {
                response.Append(x);

                if (is_first_token)
                {

                    chatWindow.AddMessage(npc_name, response.ToString());
                    is_first_token = false;
                }
                else
                {
                    chatWindow.PopLastMessage();
                    chatWindow.AddMessage(npc_name, response.ToString());

                }
            });

            // if (llm.conversation_cache_tokens != null)
            //     UnityEngine.Debug.Log(llm.conversation_cache_tokens.ToCommaSeparatedString());

            chatWindow.SendButton.GetComponent<Button>().interactable = true;
            chatWindow.InputField.ActivateInputField();
            state = NPCState.WaitingInInteraction;
        }


        public void CloseInteraction()
        {
            // Debug.Log("No longer interacting");
            state = NPCState.Idle;

            if (!keepModelLoaded)
            {
                llm?.Release(); // free the GPU buffers now (the finalizer can't call Unity APIs safely)
                llm = null;
                StartCoroutine(CollectGarbageIncremental());
            }
            chatWindow.Clear();
            chatWindow.gameObject.SetActive(false);
            pressIToInteractText.enabled = true;

            player.ExitInteractiveMode();
        }










        // Spreads the post-conversation cleanup over ~2 ms slices per frame instead of one
        // blocking GC.Collect (~400 ms). Incremental GC is enabled in Project Settings; if it
        // were disabled, CollectIncremental no-ops and the next natural collection handles it.
        IEnumerator CollectGarbageIncremental()
        {
            while (UnityEngine.Scripting.GarbageCollector.CollectIncremental(2_000_000UL))
                yield return null;
        }

        private void OnTriggerEnter2D(Collider2D other)
        {
            if (other.CompareTag("Player"))
            {
                pressIToInteractText.enabled = true;
                state = NPCState.WaitingInInteraction;
                this.player = other.gameObject.GetComponent<KnightScript>();
            }
        }

        private void OnTriggerExit2D(Collider2D other)
        {
            if (other.CompareTag("Player"))
            {
                pressIToInteractText.enabled = false;
                state = NPCState.WaitingInInteraction;
                this.player = null;
            }
        }
    }
}