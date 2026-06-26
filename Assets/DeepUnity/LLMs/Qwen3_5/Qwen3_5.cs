using System;
using System.Collections;
using System.Diagnostics;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    /// <summary>Qwen3.5 model size. Only sizes with exported params are listed.</summary>
    public enum Qwen3_5Size
    {
        /// <summary>Qwen3.5-0.8B (text-only).</summary>
        [Tooltip("Qwen3.5-0.8B (text-only) — the only size exported so far.")]
        B0_8,
    }

    // Qwen3.5-0.8B (text-only), full-GPU inference. Hybrid architecture:
    // 18 Gated DeltaNet layers + 6 full-attention layers (interval 4).
    // Weights run as packed FP16 or weight-only INT8 (see LLMQuant / import_params.py --quant int8).
    public class Qwen3_5ForCausalLM : LLM
    {
        private static readonly Qwen3_5ConfigDescriptor _config = new();
        private string path;
        private readonly Qwen3_5Size size;
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
        /// The <see cref="Chat"/>/<see cref="Generate"/> signature defaults are NEUTRAL (no truncation,
        /// no penalties — see <see cref="LLM"/>); pass one of the presets above explicitly. The
        /// "non-thinking, text" preset (the mode this demo runs) is also exposed via Config.Default*.
        /// presence_penalty is subtractive (OpenAI/vLLM style); repetition_penalty is multiplicative
        /// (CTRL/HF style, 1.0 = off). Both run on the GPU over already-generated tokens.
        /// Set temperature=0 for greedy decoding.
        /// </summary>
        /// <param name="size">Model size; resolves the default params folder (only 0.8B exported so far).</param>
        /// <param name="quantization">
        /// Weight format: FP16 (weights_qwen3.5_0.8B_fp16) or weight-only INT8 (..._int8, ~half the VRAM and
        /// disk; per-output-row scales, activations stay FP32). One quant mode per session — the
        /// keyword lives on the shared compute shader.
        /// </param>
        /// <param name="params_path">Optional override; null resolves from size + quantization.</param>
        /// <param name="maxModelLength">
        /// Maximum sequence length (in tokens) the model supports in a single conversation. This sizes the
        /// KV cache, which is pre-allocated up front to this capacity. NOTE: the KV cache is currently a fixed
        /// pre-allocation; in the future we may make it dynamic (grow on demand, array-list style) so memory
        /// scales with the actual context length instead of always reserving the maximum.
        /// </param>
        public Qwen3_5ForCausalLM(
            Qwen3_5Size size = Qwen3_5Size.B0_8,
            LLMQuant quantization = LLMQuant.FP16,
            string params_path = null,
            string tokenizer_path = "Assets/DeepUnity/LLMs/Qwen3_5/Qwen3_5TokenizerFast.json",
            int maxModelLength = 8192)
        {
            params_path ??= ResolveParamsPath(size, quantization);
            this.size = size;
            this.path = params_path;
            WarnIfNotInResources("weights", params_path);
            WarnIfNotInResources("tokenizer", tokenizer_path);
            // Tokenizer is optional during early bring-up — when the JSON isn't present yet,
            // skip it so the model can still be exercised via Predict() with token-id Tensors.
            // Cached per path in the LLM base (see GetOrCreateTokenizer for why).
            this.tokenizer = GetOrCreateTokenizer(tokenizer_path, p => new Qwen3_5TokenizerFast(p, load_async: true));

            model = new Qwen3_5Modeling.Qwen3_5Model(params_path, maxModelLength, quantization);
            // Feed the tokenizer's main-thread ctor cost to the weights object; the single consolidated
            // "model booted up" log is emitted from InitializeChat once everything is ready.
            model.weights.bootTokenizerMs = tokenizer?.ctorMs ?? 0;
        }

        // Human-readable model size for boot logs / UI (the enum's own name is terse).
        static string SizeLabel(Qwen3_5Size size) => size switch
        {
            Qwen3_5Size.B0_8 => "0.8B",
            _ => size.ToString(),
        };

        static string ResolveParamsPath(Qwen3_5Size size, LLMQuant quant)
        {
            // Self-describing folder name weights_<model>_<size>_<quant> (e.g.
            // weights_qwen3.5_0.8B_int8), matching import_params.py; resolved Resources-first
            // with a legacy fallback. The size grows as more exports land (see SizeLabel).
            string q = quant == LLMQuant.INT8 ? "int8" : quant == LLMQuant.INT4 ? "int4" : "fp16";
            return ResolveParamsDir("Qwen3_5", $"weights_qwen3.5_{SizeLabel(size)}_{q}");
        }

        /// <summary>
        /// One-call scene-start prewarm — run this as a coroutine while the player is doing
        /// something else (walking around, in a menu) and constructing/loading the model later
        /// becomes hitch-free. It (a) starts the background tokenizer parse and caches the result
        /// (the parse garbage otherwise triggers a ~300 ms GC collection mid-load), and (b) compiles
        /// every compute kernel, one per frame (the driver's one-time first-dispatch cost — up to
        /// ~800 ms for the biggest kernel — would otherwise land inside the loading window).
        /// Idempotent; needs no model instance and no weights.
        /// </summary>
        public static IEnumerator Prewarm(string tokenizer_path = "Assets/DeepUnity/LLMs/Qwen3_5/Qwen3_5TokenizerFast.json")
        {
            GetOrCreateTokenizer(tokenizer_path, p => new Qwen3_5TokenizerFast(p, load_async: true));
            yield return Qwen3_5Modeling.Qwen3_5Model.PrewarmKernels();
        }

        /// <inheritdoc/>
        public override void Release()
        {
            model?.Dispose();
            OnReleased(); // unhook editor event + suppress the finalizer (it would double-release off-thread)
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
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f)
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            model.ResetCache();

            var e = model.ForwardYielding(input_ids, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
            while (e.MoveNext()) yield return e.Current;

            int[] sampled = new int[1];
            var s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, sampled);
            while (s.MoveNext()) yield return s.Current;
            int tokenId = sampled[0];
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

                s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, sampled);
                while (s.MoveNext()) yield return s.Current;
                tokenId = sampled[0];
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
            // Warmup is part of initialization: kernel compiles + throwaway forwards happen here,
            // behind the caller's loading screen, never on the first reply. Idempotent.
            CurrentPhase = "boot (weights+warmup)";
            yield return Warmup();

            while (!IsReady) yield return new WaitForSeconds(0.01f);
            CurrentPhase = "idle";
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

            // Disk-cached system prompt: same prompt + same weights -> restore the KV/SSM state
            // (one frame-budgeted upload per layer) instead of recomputing the chunked prefill.
            bool restoredFromDisk = false;
            string cacheFile = null;
            if (SystemPromptDiskCache && Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE)
            {
                string dir = System.IO.Path.Combine(Application.persistentDataPath, "DeepUnity");
                System.IO.Directory.CreateDirectory(dir);
                cacheFile = System.IO.Path.Combine(dir, $"qwen35_prompt_{PromptCacheKey(ids)}.kv");
                if (System.IO.File.Exists(cacheFile))
                {
                    CurrentPhase = "kv-restore";
                    var load = model.cache.LoadYielding(cacheFile, ok => restoredFromDisk = ok);
                    while (load.MoveNext()) yield return load.Current;
                    if (!restoredFromDisk)
                    {
                        try { System.IO.File.Delete(cacheFile); } catch { }
                        model.ResetCache();   // a partial upload may have dirtied the state
                    }
                }
            }

            if (!restoredFromDisk)
            {
                CurrentPhase = "prefill";
                var e = ForwardPromptChunked(ids);
                while (e.MoveNext()) yield return e.Current;

                if (cacheFile != null)
                {
                    CurrentPhase = "kv-save";
                    var save = model.cache.SaveYielding(cacheFile);
                    while (save.MoveNext()) yield return save.Current;
                }
            }

            CurrentPhase = "idle";
            LogBootSummary(sw.Elapsed.TotalMilliseconds, restoredFromDisk);

            isFreshlyInitialized = true;
            yield return true;
        }

        public override IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false)
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

            CurrentPhase = "decode";
            var e = ForwardPromptChunked(ids);
            while (e.MoveNext()) yield return e.Current;

            int[] sampled = new int[1];
            var s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, sampled);
            while (s.MoveNext()) yield return s.Current;
            int tokenId = sampled[0];
            string tokenStr = tokenizer.Decode(Tensor.Constant(tokenId))[0];
            onTokenGenerated?.Invoke(tokenStr);
            yield return null;

            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                Stopwatch sw = Stopwatch.StartNew();
                Tensor nextInput = Tensor.Constant(tokenId);
                e = model.ForwardYielding(nextInput, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;

                s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, sampled);
                while (s.MoveNext()) yield return s.Current;
                tokenId = sampled[0];
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
            CurrentPhase = "idle";
            yield return true;
        }

        /// <summary>
        /// When on (default), InitializeChat persists the system prompt's KV/SSM state to
        /// persistentDataPath after the first prefill and restores it on later inits with the
        /// same prompt + weights — turning the prompt prefill into a fast, hitch-free disk load.
        /// </summary>
        public static bool SystemPromptDiskCache = true;

        // FNV-1a over the prompt token ids, weight path and cache capacity — any of these
        // changing must invalidate the cached state.
        string PromptCacheKey(System.Collections.Generic.List<float> ids)
        {
            ulong h = 14695981039346656037UL;
            void Mix(ulong v) { h ^= v; h *= 1099511628211UL; }
            foreach (var id in ids) Mix((ulong)(long)id);
            foreach (char c in path) Mix(c);
            Mix((ulong)model.cache.Capacity);
            return h.ToString("x16");
        }

        // Boot log: load time + system prompt (computed vs restored from disk, with token count).
        void LogBootSummary(double systemPromptMs, bool promptFromDisk = false)
        {
            var w = model.weights;
            double loadMs = w.bootTokenizerMs + w.bootKernelsMs + w.allocMs + w.bootCacheMs + w.bootRopeMs + w.bootScratchMs + w.uploadMs;
            int promptTokens = model.cache != null ? model.cache.CachedTokenCount : 0;
            string prompt = systemPromptMs <= 0
                ? "no system prompt"
                : (promptFromDisk
                    ? $"system prompt restored from disk ({promptTokens} tokens, {systemPromptMs:0} ms)"
                    : $"system prompt computed ({promptTokens} tokens, {systemPromptMs:0} ms)");
            ConsoleMessage.Info($"Qwen3.5-{SizeLabel(size)} {model.Quant} ready — load {loadMs:0} ms, {prompt}");

            // Detailed per-step breakdown, kept for debugging:
            // double blocking = w.bootTokenizerMs + w.bootKernelsMs + w.allocMs + w.bootCacheMs + w.bootRopeMs + w.bootScratchMs;
            // double total = blocking + w.uploadMs + systemPromptMs;
            // ConsoleMessage.Info(
            //     $"Qwen3.5 model booted up — {total:0} ms total\n" +
            //     $"   tokenizer ctor (main thread) : {w.bootTokenizerMs:0} ms\n" +
            //     $"   compute kernels lookup       : {w.bootKernelsMs:0} ms\n" +
            //     $"   weight manifest build        : {w.allocMs:0} ms (buffers created lazily during upload)\n" +
            //     $"   kv cache alloc               : {w.bootCacheMs:0} ms\n" +
            //     $"   rope kick (async)            : {w.bootRopeMs:0} ms\n" +
            //     $"   scratch buffers alloc        : {w.bootScratchMs:0} ms\n" +
            //     $"   = blocking (one frame)       : {blocking:0} ms\n" +
            //     $"   rope compute (async)         : {w.ropeAsyncMs:0} ms (overlaps upload)\n" +
            //     $"   weight stream (async)        : {w.uploadMs:0} ms over {w.uploadFrames} frames, worst slice {w.worstUploadMs:0.0} ms\n" +
            //     $"   kernel warmup (behind load)  : {w.warmupMs:0} ms (0 = warmup didn't run)\n" +
            //     $"   system prompt cache          : {systemPromptMs:0} ms" +
            //     (promptFromDisk ? " (restored from disk)" : ""));
        }

        // Forwards a prompt in small chunks — the KV cache / SSM states carry context between them —
        // so each yielded frame's GPU work is one layer of CHUNK tokens instead of one layer of the
        // whole prompt. A ~60-token prompt forwarded whole costs ~70 ms of GPU per layer-frame
        // (~14 fps for ~1.5 s during InitializeChat); chunked, frames stay within a 60 fps budget.
        IEnumerator ForwardPromptChunked(System.Collections.Generic.List<float> ids)
        {
            if (!Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE)
            {
                // Chunking needs the cache to carry state; without it, forward the whole prompt.
                var all = model.ForwardYielding(Tensor.Constant(ids.ToArray()), useCache: false, lastPosOnly: true);
                while (all.MoveNext()) yield return all.Current;
                yield break;
            }

            const int CHUNK = 8;
            for (int start = 0; start < ids.Count; start += CHUNK)
            {
                int len = Math.Min(CHUNK, ids.Count - start);
                float[] part = new float[len];
                for (int i = 0; i < len; i++) part[i] = ids[start + i];
                var e = model.ForwardYielding(Tensor.Constant(part), useCache: true, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;
            }
        }

        void AppendTextTokens(string text, System.Collections.Generic.List<float> dst)
        {
            (Tensor t, _) = tokenizer.Encode(text, add_special_tokens: false);
            for (int i = 0; i < t.Size(-1); i++) dst.Add(t[i]);
        }
    }
}
