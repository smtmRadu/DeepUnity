using System.Collections;
using System.Collections.Generic;
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

        private Qwen3_5ForCausalLM model;
        private Qwen3_5TokenizerFast tokenizer;

        private void Start()
        {
            Benckmark.Start();
            model = new Qwen3_5ForCausalLM();
            Benckmark.Stop("Qwen3.5 model init");

            tokenizer = model.tokenizer ?? new Qwen3_5TokenizerFast();
            StartCoroutine(WaitThenGenerate());
        }

        private IEnumerator WaitThenGenerate()
        {
            while (!model.IsReady) yield return new WaitForSeconds(0.01f);
            while (!tokenizer.IsReady) yield return new WaitForSeconds(0.01f);

            var msgs = new List<Dictionary<string, string>>();
            if (!string.IsNullOrEmpty(system_prompt))
                msgs.Add(new() { { "role", "system" }, { "content", system_prompt } });
            msgs.Add(new() { { "role", "user" }, { "content", user_prompt } });
            Tensor input_ids = tokenizer.ApplyChatTemplate(msgs, add_generation_prompt: true);

            Debug.Log("prompt token count: " + input_ids.Size(-1));
            Debug.Log("prompt decoded: " + tokenizer.Decode(input_ids)[0]);

            if (display != null)
                display.text = $"User:\n{user_prompt}\n\nAssistant:\n";

            yield return model.Generate(input_ids, onTokenGenerated: x =>
            {
                if (display != null) display.text += x;
                if (paramsDisplay != null)
                    paramsDisplay.text = $"Inference: {model.TokensPerSecond:0.0} tok/s";
            },
            max_new_tokens: max_completion_tokens,
            temperature: temperature);
        }
    }
}
