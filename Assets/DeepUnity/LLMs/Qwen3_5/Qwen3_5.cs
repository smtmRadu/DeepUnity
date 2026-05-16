using System;
using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    // Qwen3.5-0.8B (text-only) FP16, full-GPU inference. Hybrid architecture:
    // 18 Gated DeltaNet layers + 6 full-attention layers (interval 4).
    public class Qwen3_5ForCausalLM
    {
        private string path;
        public Qwen3_5Modeling.Qwen3_5Model model;
        public Qwen3_5TokenizerFast tokenizer;
        private bool isFreshlyInitialized;

        public bool IsReady => model.IsReady && (tokenizer == null || tokenizer.IsReady);
        public float TokensPerSecond { get; private set; }

        public Qwen3_5ForCausalLM(
            string params_path = "Assets/DeepUnity/LLMs/Qwen3_5/params_it",
            string tokenizer_path = "Assets/DeepUnity/LLMs/Qwen3_5/Qwen3_5TokenizerFast.json",
            int cacheCapacity = 8192)
        {
            this.path = params_path;
            // Tokenizer is optional during early bring-up — when the JSON isn't present yet,
            // skip it so the model can still be exercised via Predict() with token-id Tensors.
            if (System.IO.File.Exists(tokenizer_path))
                this.tokenizer = new Qwen3_5TokenizerFast(tokenizer_path, load_async: true);

#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeChanged;
#endif
            Stopwatch sw = Stopwatch.StartNew();
            model = new Qwen3_5Modeling.Qwen3_5Model(params_path, cacheCapacity);
            ConsoleMessage.Info($"Qwen3.5 model created ({sw.Elapsed.TotalSeconds:0.00} s)");
        }

        ~Qwen3_5ForCausalLM()
        {
            model?.Dispose();
            ConsoleMessage.Info("Qwen3.5 released from GPU");
        }

#if UNITY_EDITOR
        private void OnPlayModeChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
            {
                model?.Dispose();
                ConsoleMessage.Info("Qwen3.5 released from GPU");
            }
        }
#endif

        public Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady) throw new Exception("Qwen3.5 is not ready. Check IsReady first.");
            int seqLen = input_ids.Size(-1);
            model.Forward(input_ids, useCache: false, lastPosOnly: false);
            return model.ReadLogits(seqLen);
        }

        public IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0f)
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            model.ResetCache();

            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p);
            if (tokenizer != null)
                onTokenGenerated?.Invoke(tokenizer.Decode(Tensor.Constant(tokenId))[0]);
            else
                onTokenGenerated?.Invoke(tokenId.ToString() + " ");
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p);
                if (tokenId == Qwen3_5Modeling.Qwen3_5Config.EOS_TOKEN_ID) break;

                if (tokenizer != null)
                    onTokenGenerated?.Invoke(tokenizer.Decode(Tensor.Constant(tokenId))[0]);
                else
                    onTokenGenerated?.Invoke(tokenId.ToString() + " ");
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        // Chat template / multi-turn API is wired against the Qwen tokenizer which is still
        // pending. Once Qwen3_5TokenizerFast lands the InitializeChat/Chat methods can be
        // ported from Gemma3.cs by swapping the role-marker tokens to Qwen's <|im_start|> /
        // <|im_end|> and the EOS to Qwen3_5Config.EOS_TOKEN_ID.
    }
}
