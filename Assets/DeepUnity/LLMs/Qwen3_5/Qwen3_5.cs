using System;
using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    // Qwen3.5-0.8B (text-only) FP16, full-GPU inference. Hybrid architecture:
    // 18 Gated DeltaNet layers + 6 full-attention layers (interval 4).
    public class Qwen3_5ForCausalLM : LLM
    {
        private static readonly Qwen3_5ConfigDescriptor _config = new();
        private string path;
        public Qwen3_5Modeling.Qwen3_5Model model;
        public Qwen3_5TokenizerFast tokenizer;
        private bool isFreshlyInitialized;

        public override LLMConfig Config => _config;
        public override bool IsReady => model.IsReady && (tokenizer == null || tokenizer.IsReady);

        /// <summary>
        /// Qwen3.5-0.8B (text-only), full-GPU FP16 inference.
        ///
        /// Recommended sampling presets (set these on <see cref="Chat"/> / <see cref="Generate"/>):
        ///
        ///   Non-thinking, text tasks:
        ///     temperature=1.0, top_p=1.00, top_k=20, min_p=0.0, presence_penalty=2.0, repetition_penalty=1.0
        ///   Non-thinking, VL tasks:
        ///     temperature=0.7, top_p=0.80, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
        ///   Thinking, text tasks:
        ///     temperature=1.0, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=1.5, repetition_penalty=1.0
        ///   Thinking, VL or precise coding (e.g. WebDev) tasks:
        ///     temperature=0.6, top_p=0.95, top_k=20, min_p=0.0, presence_penalty=0.0, repetition_penalty=1.0
        ///
        /// Defaults below are the "non-thinking, text" preset. presence_penalty is subtractive
        /// (OpenAI/vLLM style); repetition_penalty is multiplicative (CTRL/HF style, 1.0 = off). Both
        /// run on the GPU over already-generated tokens. Set temperature=0 for greedy decoding.
        /// </summary>
        /// <param name="maxModelLength">
        /// Maximum sequence length (in tokens) the model supports in a single conversation. This sizes the
        /// KV cache, which is pre-allocated up front to this capacity. NOTE: the KV cache is currently a fixed
        /// pre-allocation; in the future we may make it dynamic (grow on demand, array-list style) so memory
        /// scales with the actual context length instead of always reserving the maximum.
        /// </param>
        public Qwen3_5ForCausalLM(
            string params_path = "Assets/DeepUnity/LLMs/Qwen3_5/params_it",
            string tokenizer_path = "Assets/DeepUnity/LLMs/Qwen3_5/Qwen3_5TokenizerFast.json",
            int maxModelLength = 8192)
        {
            this.path = params_path;
            WarnIfNotInResources("weights", params_path);
            WarnIfNotInResources("tokenizer", tokenizer_path);
            // Tokenizer is optional during early bring-up — when the JSON isn't present yet,
            // skip it so the model can still be exercised via Predict() with token-id Tensors.
            if (System.IO.File.Exists(tokenizer_path))
                this.tokenizer = new Qwen3_5TokenizerFast(tokenizer_path, load_async: true);

            model = new Qwen3_5Modeling.Qwen3_5Model(params_path, maxModelLength);
            // Feed the tokenizer's main-thread ctor cost to the weights object; the single consolidated
            // "model booted up" log is emitted from InitializeChat once everything is ready.
            model.weights.bootTokenizerMs = tokenizer?.ctorMs ?? 0;
        }

        /// <inheritdoc/>
        public override void Release()
        {
            model?.Dispose();
            ConsoleMessage.Info("Qwen3.5 released from GPU");
        }

        /// <summary>
        /// Compiles every compute kernel (one-time first-dispatch cost) behind the loading screen so the
        /// first real Chat/Generate reply is fast. Yields per layer; idempotent. Call once after creating
        /// the model, before InitializeChat. Waits internally for IsReady.
        /// </summary>
        public override IEnumerator Warmup() => model.Warmup();

        public override Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady) throw new Exception("Qwen3.5 is not ready. Check IsReady first.");
            int seqLen = input_ids.Size(-1);
            model.Forward(input_ids, useCache: false, lastPosOnly: false);
            return model.ReadLogits(seqLen);
        }

        public override IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 20, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 2f, float repetition_penalty = 1f)
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            model.ResetCache();

            var e = model.ForwardYielding(input_ids, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
            if (tokenizer != null)
                onTokenGenerated?.Invoke(tokenizer.Decode(Tensor.Constant(tokenId))[0]);
            else
                onTokenGenerated?.Invoke(tokenId.ToString() + " ");
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
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

        public override IEnumerator InitializeChat(string system_prompt = "")
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);
            Assert.AreNotEqual(system_prompt, null);

            model.ResetCache();

            if (string.IsNullOrEmpty(system_prompt))
            {
                LogBootSummary(0);
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
            var e = model.ForwardYielding(input_ids, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            LogBootSummary(sw.Elapsed.TotalMilliseconds);

            isFreshlyInitialized = true;
            yield return true;
        }

        public override IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 20, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 2f, float repetition_penalty = 1f, bool enable_thinking = false)
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
            var e = model.ForwardYielding(input_ids, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int tokenId = model.Sample(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                tokenId = model.Sample(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
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

        // Single consolidated boot log: total + per-step breakdown. The construction steps run in one
        // synchronous frame ("blocking"); the weight upload streams across frames afterward.
        void LogBootSummary(double systemPromptMs)
        {
            var w = model.weights;
            double blocking = w.bootTokenizerMs + w.bootKernelsMs + w.allocMs + w.bootCacheMs + w.bootRopeMs + w.bootScratchMs;
            double total = blocking + w.uploadMs + systemPromptMs;
            ConsoleMessage.Info(
                $"Qwen3.5 model booted up — {total:0} ms total\n" +
                $"   tokenizer ctor (main thread) : {w.bootTokenizerMs:0} ms\n" +
                $"   compute kernels lookup       : {w.bootKernelsMs:0} ms\n" +
                $"   weight buffers alloc         : {w.allocMs:0} ms\n" +
                $"   kv cache alloc               : {w.bootCacheMs:0} ms\n" +
                $"   rope kick (async)            : {w.bootRopeMs:0} ms\n" +
                $"   scratch buffers alloc        : {w.bootScratchMs:0} ms\n" +
                $"   = blocking (one frame)       : {blocking:0} ms\n" +
                $"   rope compute (async)         : {w.ropeAsyncMs:0} ms (overlaps upload)\n" +
                $"   weight upload (async, GPU)   : {w.uploadMs:0} ms over {w.uploadFrames} frames, worst frame {w.worstUploadMs:0.0} ms\n" +
                $"   kernel warmup (behind load)  : {w.warmupMs:0} ms (0 = warmup didn't run)\n" +
                $"   system prompt cache          : {systemPromptMs:0} ms");
        }

        void AppendTextTokens(string text, System.Collections.Generic.List<float> dst)
        {
            (Tensor t, _) = tokenizer.Encode(text, add_special_tokens: false);
            for (int i = 0; i < t.Size(-1); i++) dst.Add(t[i]);
        }
    }
}
