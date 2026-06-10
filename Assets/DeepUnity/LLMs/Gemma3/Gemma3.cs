using System;
using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    // Default Gemma3 implementation in DeepUnity. FP16 weights, full-GPU inference
    // (every kernel reads/writes ComputeBuffers — no per-step CPU<->GPU bounces).
    // System-prompt KV cache is persisted as packed FP16 .bin under Assets/Resources/Cache/.
    // Supersedes the earlier hybrid GPU/CPU `Gemma3OriginalForCausalLM`.
    public class Gemma3ForCausalLM : LLM
    {
        private static readonly Gemma3ConfigDescriptor _config = new();
        private string path;
        public Gemma3Modeling.Gemma3Model model;
        public Gemma3TokenizerFast tokenizer;
        private bool isFreshlyInitialized;

        public override LLMConfig Config => _config;
        public override bool IsReady => model.IsReady && (tokenizer == null || tokenizer.IsReady);

        /// <param name="maxModelLength">
        /// Maximum sequence length (in tokens) the model supports in a single conversation. This sizes the
        /// KV cache, which is pre-allocated up front to this capacity. NOTE: the KV cache is currently a fixed
        /// pre-allocation; in the future we may make it dynamic (grow on demand, array-list style) so memory
        /// scales with the actual context length instead of always reserving the maximum.
        /// </param>
        public Gemma3ForCausalLM(
            string params_path = "Assets/DeepUnity/LLMs/Gemma3/params_it",
            string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json",
            int maxModelLength = 2048)
        {
            this.path = params_path;
            WarnIfNotInResources("weights", params_path);
            WarnIfNotInResources("tokenizer", tokenizer_path);
            // Cached per path in the LLM base (see GetOrCreateTokenizer for why).
            this.tokenizer = GetOrCreateTokenizer(tokenizer_path, p => new Gemma3TokenizerFast(p, load_async: true));

            // Cheap: builds the weight-file manifest and kicks the background stream; the weights
            // upload to the GPU over subsequent frames under a per-frame budget.
            model = new Gemma3Modeling.Gemma3Model(params_path, maxModelLength);
        }

        /// <summary>
        /// One-call scene-start prewarm — run this as a coroutine while the player is doing
        /// something else and constructing/loading the model later becomes hitch-free: starts the
        /// background tokenizer parse (cached) and compiles every compute kernel, one per frame.
        /// Idempotent; needs no model instance and no weights.
        /// </summary>
        public static IEnumerator Prewarm(string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json")
        {
            GetOrCreateTokenizer(tokenizer_path, p => new Gemma3TokenizerFast(p, load_async: true));
            yield return Gemma3Modeling.Gemma3Model.PrewarmKernels();
        }

        /// <inheritdoc/>
        public override IEnumerator Warmup() => model.Warmup();

        /// <inheritdoc/>
        public override void Release()
        {
            model?.Dispose();
            OnReleased(); // unhook editor event + suppress the finalizer (it would double-release off-thread)
            ConsoleMessage.Info("Gemma3 released from GPU");
        }

        public int ParameterCount()
        {
            int p = Gemma3Modeling.Gemma3Config.VOCAB_SIZE * Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
            int H = Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
            int D = Gemma3Modeling.Gemma3Config.HEAD_DIM;
            float exp = Gemma3Modeling.Gemma3Config.ATTN_EXPANSION_FACTOR;
            int innerEmb = (int)(H * exp);
            int Hq = Gemma3Modeling.Gemma3Config.HEADS_Q;
            int Hkv = Gemma3Modeling.Gemma3Config.HEADS_KV;
            int I = Gemma3Modeling.Gemma3Config.MLP_INTERMEDIATE_SIZE;

            int perLayer = 0;
            perLayer += H * innerEmb;
            perLayer += H * innerEmb * Hkv / Hq;
            perLayer += H * innerEmb * Hkv / Hq;
            perLayer += innerEmb * H;
            perLayer += D + D;
            perLayer += H * I * 3;
            perLayer += H * 4;

            p += perLayer * Gemma3Modeling.Gemma3Config.NUM_LAYERS;
            p += H;
            return p;
        }

        public override Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady)
                throw new Exception("Gemma3 is not ready. Check IsReady first.");
            int seqLen = input_ids.Size(-1);
            model.Forward(input_ids, useCache: false, lastPosOnly: false);
            return model.ReadLogits(seqLen);
        }

        // presence_penalty / repetition_penalty are accepted for API parity with other LLMs but ignored —
        // Gemma3's sampler has no penalty support. top_k default mirrors Gemma's generation_config (64).
        public override IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 64, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f)
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            model.ResetCache();

            var e = ForwardPromptChunked(input_ids);
            while (e.MoveNext()) yield return e.Current;

            int[] sampled = new int[1];
            var s = model.SampleYielding(temperature, top_k, top_p, min_p, sampled);
            while (s.MoveNext()) yield return s.Current;
            int tokenId = sampled[0];
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                s = model.SampleYielding(temperature, top_k, top_p, min_p, sampled);
                while (s.MoveNext()) yield return s.Current;
                tokenId = sampled[0];
                if (tokenId == Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID) break;

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        // Forwards a prompt in small chunks — the KV cache carries context between them — so each
        // yielded frame's GPU work is one layer of CHUNK tokens instead of one layer of the whole
        // prompt (a long prompt forwarded whole tanks the framerate for its entire prefill).
        IEnumerator ForwardPromptChunked(Tensor input_ids)
        {
            const int CHUNK = 8;
            int total = input_ids.Size(-1);
            for (int start = 0; start < total; start += CHUNK)
            {
                int len = Math.Min(CHUNK, total - start);
                float[] part = new float[len];
                for (int i = 0; i < len; i++) part[i] = input_ids[start + i];
                var e = model.ForwardYielding(Tensor.Constant(part), useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;
            }
        }

        public override IEnumerator InitializeChat(string system_prompt = "")
        {
            // Warmup is part of initialization: kernel compiles + throwaway forwards happen here,
            // behind the caller's loading screen, never on the first reply. Idempotent.
            yield return Warmup();

            while (!IsReady) yield return new WaitForSeconds(0.01f);
            Assert.AreNotEqual(system_prompt, null);

            model.ResetCache();

            string cacheFolder = SystemPromptCacheFolder(system_prompt);
            bool loaded = false;
            yield return model.cache.TryLoadAsync(cacheFolder, r => loaded = r);
            if (loaded)
            {
                ConsoleMessage.Info($"Gemma3 system prompt cache hit ({model.cache.CachedTokenCount} tokens).");
                isFreshlyInitialized = true;
                yield return true;
                yield break;
            }

            Stopwatch sw = Stopwatch.StartNew();

            (Tensor, Tensor) tok = tokenizer.Encode(system_prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            Tensor prefix = Tensor.Constant(new float[]
            {
                Gemma3TokenizerFast.BOS_TOKEN_ID,
                Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID,
                2364f, 107f
            });
            Tensor input_ids = Tensor.Concat(-1, prefix, tok.Item1);

            var e = ForwardPromptChunked(input_ids);
            while (e.MoveNext()) yield return e.Current;

            ConsoleMessage.Info($"Gemma3 system prompt computed ({sw.Elapsed.TotalSeconds:0.00} s).");

            yield return model.cache.SaveAsync(cacheFolder);
            ConsoleMessage.Info($"Gemma3 system prompt cache saved ({model.cache.CachedTokenCount} tokens).");

            isFreshlyInitialized = true;
            yield return true;
        }

        static string SystemPromptCacheFolder(string system_prompt)
        {
            using (var sha = System.Security.Cryptography.SHA256.Create())
            {
                byte[] h = sha.ComputeHash(System.Text.Encoding.UTF8.GetBytes(system_prompt ?? string.Empty));
                var sb = new System.Text.StringBuilder(64);
                for (int i = 0; i < h.Length; i++) sb.Append(h[i].ToString("x2"));
                return System.IO.Path.Combine("Assets/Resources/Cache", sb.ToString());
            }
        }

        // presence_penalty / repetition_penalty / enable_thinking are accepted for API parity with other LLMs
        // but ignored — Gemma3-270m has no penalty or thinking support. top_k default mirrors Gemma's
        // generation_config (64).
        public override IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 64, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false)
        {
            if (!IsReady) throw new Exception("Call InitializeChat before Chat.");

            Tensor prefixT, postfixT;
            if (isFreshlyInitialized)
            {
                isFreshlyInitialized = false;
                prefixT = Tensor.Constant(new float[] { 108f });
                postfixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID, 107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID, 4368f, 107f
                });
            }
            else
            {
                prefixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID, 107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID, 2364f, 107f
                });
                postfixT = Tensor.Constant(new float[]
                {
                    Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID, 107f,
                    Gemma3TokenizerFast.START_OF_TURN_TOKEN_ID, 4368f, 107f
                });
            }

            (Tensor, Tensor) tok = tokenizer.Encode(prompt, add_special_tokens: false, truncation: true, max_length: 2048);
            Tensor input_ids = Tensor.Concat(-1, prefixT, tok.Item1, postfixT);

            var e = ForwardPromptChunked(input_ids);
            while (e.MoveNext()) yield return e.Current;

            int[] sampled = new int[1];
            var s = model.SampleYielding(temperature, top_k, top_p, min_p, sampled);
            while (s.MoveNext()) yield return s.Current;
            int tokenId = sampled[0];
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                s = model.SampleYielding(temperature, top_k, top_p, min_p, sampled);
                while (s.MoveNext()) yield return s.Current;
                tokenId = sampled[0];
                if (tokenId == Gemma3TokenizerFast.END_OF_TURN_TOKEN_ID)
                {
                    ConsoleMessage.Info("Gemma3 ended the response.");
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
    }
}
