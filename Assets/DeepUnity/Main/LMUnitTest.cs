using System.Collections;
using UnityEngine;
using UnityEngine.UI;

namespace DeepUnity.Tutorials
{
    // Bring-up harness for the text-only LLMs — pick the model from the inspector dropdown.
    public class LMUnitTest : MonoBehaviour
    {
        public enum LM { Gemma3_270m, Qwen3_5 }

        [Tooltip("Which LLM to boot.")]
        [SerializeField] private LM lm = LM.Gemma3_270m;
        [Tooltip("Qwen only (Gemma = 270m). Sizes with exported params.")]
        [SerializeField] private Qwen3_5Size size = Qwen3_5Size.B0_8;
        [Tooltip("FP16 = reference; INT8 = per-row, ~lossless, half VRAM (recommended). INT4 = Qwen only (32-group, lossy + slower on small models).")]
        [SerializeField] private LLMQuant quantization = LLMQuant.FP16;
        [SerializeField] private Text display;
        [SerializeField] private Text paramsDisplay;
        [Multiline]
        [SerializeField] private string system_prompt = "";
        [SerializeField] private string user_prompt = "Who are you?";
        [SerializeField] private int max_completion_tokens = 32;
        [SerializeField] private bool enable_thinking = false;   // Qwen3.5 only; Gemma3 ignores it

        private LLM model;

        private void Start()
        {
            if (lm == LM.Gemma3_270m && quantization == LLMQuant.INT4)
                Debug.LogWarning("[LMUnitTest] Gemma3 has no INT4 runtime — running FP16.");

            Benckmark.Start();
            model = lm == LM.Gemma3_270m
                ? (LLM)new Gemma3ForCausalLM(quantization)
                : new Qwen3_5ForCausalLM(size, quantization);
            Benckmark.Stop($"{lm} {(lm == LM.Gemma3_270m ? $"{quantization} " : $"{size} {quantization} ")}model init");

            StartCoroutine(Run());
        }

        private IEnumerator Run()
        {
            yield return model.InitializeChat(system_prompt); // streams weights, warms kernels, caches the system prompt

            if (display != null)
                display.text = $"User:\n{user_prompt}\n\nAssistant:\n";

            yield return model.Chat(user_prompt, onTokenGenerated: x =>
            {
                if (display != null) display.text += x;
                if (paramsDisplay != null)
                    paramsDisplay.text = $"Inference: {model.TokensPerSecond:0.0} tok/s";
            },
            max_new_tokens: max_completion_tokens,
            temperature: model.Config.DefaultTemperature,           // each model's recommended preset
            top_k: model.Config.DefaultTopK,
            top_p: model.Config.DefaultTopP,                        // (Gemma 64/0.95, Qwen 20/1.0)
            presence_penalty: model.Config.DefaultPresencePenalty,
            enable_thinking: enable_thinking);
        }
    }
}
