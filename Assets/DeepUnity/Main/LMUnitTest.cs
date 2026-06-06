using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials
{
    // Bring-up harness for Qwen3.5-0.8B (text-only).
    public class LMUnitTest : MonoBehaviour
    {
        [SerializeField] private Text display;
        [SerializeField] private Text paramsDisplay;
        [Multiline]
        [SerializeField] private string system_prompt = "";
        [SerializeField] private string user_prompt = "Who are you?";
        [SerializeField] private int max_completion_tokens = 32;
        [SerializeField] private float temperature = 0f;
        [SerializeField] private bool enable_thinking = false;

        private Qwen3_5ForCausalLM model;

        private void Start()
        {
            Benckmark.Start();
            model = new Qwen3_5ForCausalLM();
            Benckmark.Stop("Qwen3.5 model init");

            StartCoroutine(Run());
        }

        private IEnumerator Run()
        {
            yield return model.InitializeChat(system_prompt);

            if (display != null)
                display.text = $"User:\n{user_prompt}\n\nAssistant:\n";

            yield return model.Chat(user_prompt, onTokenGenerated: x =>
            {
                if (display != null) display.text += x;
                if (paramsDisplay != null)
                    paramsDisplay.text = $"Inference: {model.TokensPerSecond:0.0} tok/s";
            },
            max_new_tokens: max_completion_tokens,
            temperature: temperature,
            enable_thinking: enable_thinking);
        }
    }
}
