using System;
using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    // MiniCPM5-1B (openbmb/MiniCPM5-1B), full-GPU inference on the shared Gemma3CS kernel set.
    //
    // CHECKPOINT VARIANTS — openbmb publishes two 1B checkpoints with the same architecture:
    //   - openbmb/MiniCPM5-1B      (default, RL-tuned; what we import and benchmark)
    //   - openbmb/MiniCPM5-1B-SFT  (supervised-finetune only, no RL stage; the RL pass mostly
    //     targets reasoning, so the SFT one can read better for plain chat/roleplay)
    // Both export with the same import_params.py command (just swap the hub id) and load with the
    // same params folder layout; pass `params_path` to point at an SFT export. We benchmark the
    // default one — this model is measured for engine metrics, not shipped as the default NPC
    // model (the roleplay finetunes target Qwen3.5).
    public class MiniCPM5ForCausalLM : LLM
    {
        private static readonly MiniCPM5ConfigDescriptor _config = new();
        private string path;
        public MiniCPM5Modeling.MiniCPM5Model model;
        public MiniCPM5TokenizerFast tokenizer;

        public override LLMConfig Config => _config;
        public override bool IsReady => model.IsReady && (tokenizer == null || tokenizer.IsReady);
        public override bool TokenizerReady => tokenizer == null || tokenizer.IsReady;

        /// <param name="quantization">
        /// Weight format: FP16 (weights_minicpm5_1B_fp16), weight-only INT8 (..._int8) or INT4
        /// (..._int4, GGUF Q4_0). Activations stay FP32. One quant mode per session — the keyword
        /// lives on the shared compute shader (shared with Gemma3!).
        /// </param>
        /// <param name="params_path">Optional override; null resolves from quantization. Point this
        /// at an SFT export (see class comment) to run openbmb/MiniCPM5-1B-SFT instead.</param>
        /// <param name="maxModelLength">
        /// Maximum sequence length (in tokens) for a conversation; sizes the pre-allocated KV cache
        /// and the RoPE tables. Defaults to 8192 as a sane VRAM budget — the model itself supports
        /// up to 131,072 (MAX_POSITION_EMBEDDINGS); raise this if you need the long context.
        /// </param>
        /// <param name="kv_quant">
        /// KV-cache precision (independent of the weight quant): FP16 (default), FP32 or INT8.
        /// </param>
        public MiniCPM5ForCausalLM(
            LLMQuant quantization = LLMQuant.FP16,
            string params_path = null,   // null resolves Resources-first (import_params.py convention)
            string tokenizer_path = "Assets/DeepUnity/InferenceEngine/LLM/MiniCPM5/MiniCPM5TokenizerFast.json",
            int maxModelLength = 8192,
            KVQuant kv_quant = KVQuant.FP16)
        {
            // Self-describing folder name weights_<model>_<size>_<quant> (matches import_params.py).
            string q = quantization == LLMQuant.INT8 ? "int8"
                     : quantization == LLMQuant.INT4 ? "int4" : "fp16";
            params_path ??= ResolveParamsDir("MiniCPM5", $"weights_minicpm5_1B_{q}");
            this.path = params_path;
            WarnIfNotInResources("weights", params_path);
            WarnIfNotInResources("tokenizer", tokenizer_path);
            // Cached per path in the LLM base (see GetOrCreateTokenizer for why).
            this.tokenizer = GetOrCreateTokenizer(tokenizer_path, p => new MiniCPM5TokenizerFast(p, load_async: true));

            model = new MiniCPM5Modeling.MiniCPM5Model(params_path, maxModelLength, quantization, kv_quant);
        }

        /// <summary>
        /// One-call scene-start prewarm — starts the background tokenizer parse (cached) and
        /// compiles every compute kernel, one per frame. Idempotent; needs no model instance.
        /// </summary>
        public static IEnumerator Prewarm(string tokenizer_path = "Assets/DeepUnity/InferenceEngine/LLM/MiniCPM5/MiniCPM5TokenizerFast.json")
        {
            GetOrCreateTokenizer(tokenizer_path, p => new MiniCPM5TokenizerFast(p, load_async: true));
            yield return MiniCPM5Modeling.MiniCPM5Model.PrewarmKernels();
            // Sweep the tokenizer-parse garbage NOW, spread over frames — otherwise the first big
            // load-time allocation triggers one blocking GC collection mid-walk-up.
            while (UnityEngine.Scripting.GarbageCollector.CollectIncremental(2_000_000UL))
                yield return null;
        }

        // Self-registering LLMRegistry catalog entry (auto-discovered by model pickers).
        [LLMEntry(20)]
        static LLMRegistry.Entry RegistryEntry() => new LLMRegistry.Entry
        {
            id = "MiniCPM5-1B",
            create = (q, kv) => new MiniCPM5ForCausalLM(quantization: q, kv_quant: kv),
            prewarm = () => Prewarm(),
        };

        /// <inheritdoc/>
        public override IEnumerator Warmup() => model.Warmup();

        /// <inheritdoc/>
        public override void Release()
        {
            model?.Dispose();
            OnReleased(); // unhook editor event + suppress the finalizer
            ConsoleMessage.Info("MiniCPM5 released from GPU");
        }

        public long ParameterCount()
        {
            long H = MiniCPM5Modeling.MiniCPM5Config.HIDDEN_SIZE;
            long D = MiniCPM5Modeling.MiniCPM5Config.HEAD_DIM;
            long Hq = MiniCPM5Modeling.MiniCPM5Config.HEADS_Q;
            long Hkv = MiniCPM5Modeling.MiniCPM5Config.HEADS_KV;
            long I = MiniCPM5Modeling.MiniCPM5Config.MLP_INTERMEDIATE_SIZE;
            long V = MiniCPM5Modeling.MiniCPM5Config.VOCAB_SIZE;

            long perLayer = 0;
            perLayer += H * (Hq * D);          // q_proj
            perLayer += H * (Hkv * D) * 2;     // k_proj + v_proj
            perLayer += (Hq * D) * H;          // o_proj
            perLayer += H * I * 3;             // gate/up/down
            perLayer += H * 2;                 // 2 norms

            long p = V * H * 2;                // embedding + UNTIED lm_head
            p += perLayer * MiniCPM5Modeling.MiniCPM5Config.NUM_LAYERS;
            p += H;                            // final norm
            return p;
        }

        public override Tensor Predict(Tensor input_ids, Tensor attn_mask = null)
        {
            if (!IsReady)
                throw new Exception("MiniCPM5 is not ready. Check IsReady first.");
            int seqLen = input_ids.Size(-1);
            model.Forward(input_ids, useCache: false, lastPosOnly: false);
            return model.ReadLogits(seqLen);
        }

        // presence_penalty / repetition_penalty are accepted for API parity but ignored (the shared
        // sampler kernel has no penalty support wired for this family yet). Recommended sampling
        // (generation_config): temperature 0.9, top_p 0.95.
        public override IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
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
                if (IsStopToken(tokenId)) break;

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        // generation_config eos_token_id = [1, 130073] — plain </s> or the ChatML <|im_end|>.
        static bool IsStopToken(int id) =>
            id == MiniCPM5Modeling.MiniCPM5Config.EOS_TOKEN_ID ||
            id == MiniCPM5Modeling.MiniCPM5Config.IM_END_TOKEN_ID;

        // Forwards a prompt in small chunks so each yielded frame's GPU work stays bounded
        // (same rationale as Gemma3ForCausalLM.ForwardPromptChunked).
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
            CurrentPhase = "boot (weights+warmup)";
            yield return Warmup();

            while (!IsReady) yield return new WaitForSeconds(0.01f);
            CurrentPhase = "idle";
            Assert.AreNotEqual(system_prompt, null);

            model.ResetCache();

            CurrentPhase = "prefill";
            // ChatML per chat_template.jinja: <s> then <|im_start|>system\n{prompt}<|im_end|>\n.
            // The special tokens are matched by the tokenizer's added-token regex, so encoding the
            // literal string yields the right ids.
            string header = $"<s><|im_start|>system\n{system_prompt}<|im_end|>\n";
            (Tensor ids, Tensor _) = tokenizer.Encode(header, add_special_tokens: false, truncation: true, max_length: 4096);

            var e = ForwardPromptChunked(ids);
            while (e.MoveNext()) yield return e.Current;

            CurrentPhase = "idle";
            ConsoleMessage.Info($"MiniCPM5-1B {model.Quant} ready — system prompt computed " +
                                $"({model.cache.CachedTokenCount} tokens)");

            yield return true;
        }

        // enable_thinking is accepted for API parity but ignored (we run the non-thinking preset).
        public override IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false)
        {
            if (!IsReady) throw new Exception("Call InitializeChat before Chat.");

            // ChatML turn: user message, then open the assistant turn for generation.
            string turn = $"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n";
            (Tensor ids, Tensor _) = tokenizer.Encode(turn, add_special_tokens: false, truncation: true, max_length: 4096);

            CurrentPhase = "decode";
            var e = ForwardPromptChunked(ids);
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
                if (IsStopToken(tokenId))
                {
                    ConsoleMessage.Info("MiniCPM5 ended the response.");
                    break;
                }

                tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
                onTokenGenerated?.Invoke(tokenStr);
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                yield return null;
            }

            // Close the assistant turn in the KV cache so the next Chat() turn is well-formed.
            var closeTurn = Tensor.Constant((float)MiniCPM5Modeling.MiniCPM5Config.IM_END_TOKEN_ID);
            e = model.ForwardYielding(closeTurn, useCache: true, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            TokensPerSecond = 0f;
            CurrentPhase = "idle";
            yield return true;
        }
    }
}
