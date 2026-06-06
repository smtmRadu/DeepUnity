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

        public IEnumerator InitializeChat(string system_prompt = "")
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);
            Assert.AreNotEqual(system_prompt, null);

            model.ResetCache();

            if (string.IsNullOrEmpty(system_prompt))
            {
                isFreshlyInitialized = true;
                yield return true;
                yield break;
            }

            Stopwatch sw = Stopwatch.StartNew();

            var ids = new System.Collections.Generic.List<float>();
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID);
            AppendTextTokens("system\n", ids);
            (Tensor sysTok, _) = tokenizer.Encode(system_prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            for (int i = 0; i < sysTok.Size(-1); i++) ids.Add(sysTok[i]);
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID);
            AppendTextTokens("\n", ids);

            Tensor input_ids = Tensor.Constant(ids.ToArray());
            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            ConsoleMessage.Info($"Qwen3.5 system prompt computed ({sw.Elapsed.TotalSeconds:0.00} s).");

            isFreshlyInitialized = true;
            yield return true;
        }

        public IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = -1, float top_p = 1f, float min_p = 0f,
            bool enable_thinking = false)
        {
            if (!IsReady) throw new Exception("Call InitializeChat before Chat.");

            var ids = new System.Collections.Generic.List<float>();

            if (!isFreshlyInitialized)
            {
                // Close prior assistant turn that Chat() left open (we broke before forwarding <|im_end|>).
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID);
                AppendTextTokens("\n", ids);
            }
            isFreshlyInitialized = false;

            // Open user turn.
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID);
            AppendTextTokens("user\n", ids);
            (Tensor userTok, _) = tokenizer.Encode(prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            for (int i = 0; i < userTok.Size(-1); i++) ids.Add(userTok[i]);
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID);
            AppendTextTokens("\n", ids);

            // Open assistant turn with thinking prefix (mirrors ApplyChatTemplate).
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID);
            AppendTextTokens("assistant\n", ids);
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.THINK_OPEN_TOKEN_ID);
            if (enable_thinking)
            {
                AppendTextTokens("\n", ids);
            }
            else
            {
                AppendTextTokens("\n\n", ids);
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.THINK_CLOSE_TOKEN_ID);
                AppendTextTokens("\n\n", ids);
            }

            Tensor input_ids = Tensor.Constant(ids.ToArray());
            var e = model.ForwardYielding(input_ids, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p);
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p);
                if (tokenId == Qwen3_5Modeling.Qwen3_5Config.EOS_TOKEN_ID)
                {
                    ConsoleMessage.Info("Qwen3.5 ended the response.");
                    break;
                }

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        void AppendTextTokens(string text, System.Collections.Generic.List<float> dst)
        {
            (Tensor t, _) = tokenizer.Encode(text, add_special_tokens: false);
            for (int i = 0; i < t.Size(-1); i++) dst.Add(t[i]);
        }
    }
}
