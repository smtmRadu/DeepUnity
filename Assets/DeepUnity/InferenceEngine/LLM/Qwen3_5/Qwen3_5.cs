using System;
using System.Collections;
using System.Diagnostics;
using System.IO;
using UnityEngine;
using UnityEngine.Assertions;

namespace DeepUnity
{
    /// <summary>Qwen3.5 model size. Only sizes with exported params are listed.</summary>
    public enum Qwen3_5Size
    {
        /// <summary>Qwen3.5-0.8B (text-only).</summary>
        [Tooltip("Qwen3.5-0.8B (text-only).")]
        B0_8,
        /// <summary>Qwen3.5-2B (text-only). Same architecture as 0.8B; only hidden/intermediate dims grow.</summary>
        [Tooltip("Qwen3.5-2B (text-only) — ~4 GB fp16 / ~2.2 GB int8 / ~1.2 GB int4 on device.")]
        B2,
    }

    // Qwen3.5-0.8B (text-only), full-GPU inference. Hybrid architecture:
    // 18 Gated DeltaNet layers + 6 full-attention layers (interval 4).
    // Weights run as packed FP16 or weight-only INT8 (see LLMQuant / import_params.py --quant int8).
    public class Qwen3_5ForCausalLM : LLM
    {
        private static readonly Qwen3_5ConfigDescriptor _config = new();
        private string path;
        private readonly Qwen3_5Size size;
        private readonly int maxModelLength;
        public Qwen3_5Modeling.Qwen3_5Model model;
        public Qwen3_5TokenizerFast tokenizer;
        private bool isFreshlyInitialized;
        // Prompt from the last InitializeChat (or a successful conversation restore) — the
        // conversation-cache identity hash folds it in, so SaveConversationKV can be called
        // without repeating the prompt.
        private string lastSystemPrompt = "";

        public override LLMConfig Config => _config;
        public override bool IsReady => model != null && model.IsReady && (tokenizer == null || tokenizer.IsReady);
        public override bool TokenizerReady => tokenizer == null || tokenizer.IsReady;
        public override long TotalWeightBytes => model?.weights?.BytesTotal ?? 0;
        public override long UploadedWeightBytes => model?.weights?.BytesUploaded ?? 0;
        public override string WeightsLabel => ResidencyLog.Label(path);
        public override int CurrentContextTokens => model?.cache?.CachedTokenCount ?? 0;
        public override int MaxContextTokens => maxModelLength;

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
        /// <param name="size">Model size (0.8B or 2B); resolves the default params folder.</param>
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
        /// <param name="kv_quant">
        /// KV-cache precision (independent of the weight <paramref name="quantization"/>): FP16 (default,
        /// ~lossless, half the KV VRAM/bandwidth) or FP32 (reference). Only the 6 full-attention layers'
        /// K/V are affected; DeltaNet states stay FP32. INT8 KV is not wired up yet.
        /// </param>
        public Qwen3_5ForCausalLM(
            Qwen3_5Size size = Qwen3_5Size.B0_8,
            LLMQuant quantization = LLMQuant.FP16,
            string params_path = null,
            string tokenizer_path = "Assets/DeepUnity/InferenceEngine/LLM/Qwen3_5/Qwen3_5TokenizerFast.json",
            int maxModelLength = 8192,
            KVQuant kv_quant = KVQuant.FP16)
        {
            // Size preset first: weights/model/cache construction below all read the config statics.
            var swBoot = System.Diagnostics.Stopwatch.StartNew();
            BootTrace = ""; _lastMark = 0;
            Qwen3_5Modeling.Qwen3_5Config.ApplySize(size);
            params_path ??= ResolveParamsPath(size, quantization);
            this.size = size;
            this.maxModelLength = maxModelLength;
            this.path = params_path;
            WarnIfNotInResources("weights", params_path);
            WarnIfNotInResources("tokenizer", tokenizer_path);
            Mark(swBoot, "pre");
            // Tokenizer is optional during early bring-up — when the JSON isn't present yet,
            // skip it so the model can still be exercised via Predict() with token-id Tensors.
            // Cached per path in the LLM base (see GetOrCreateTokenizer for why).
            this.tokenizer = GetOrCreateTokenizer(tokenizer_path, p => new Qwen3_5TokenizerFast(p, load_async: true));
            Mark(swBoot, "tokenizer");

            model = new Qwen3_5Modeling.Qwen3_5Model(params_path, maxModelLength, quantization, kv_quant);
            Mark(swBoot, "model_total");
            // Feed the tokenizer's main-thread ctor cost to the weights object; the single consolidated
            // "model booted up" log is emitted from InitializeChat once everything is ready.
            model.weights.bootTokenizerMs = tokenizer?.ctorMs ?? 0;
        }

        /// <summary>Step-by-step wall-time trace of the LAST constructor run (freeze attribution;
        /// the Qwen3_5Model/Qwen3_5Weights ctors append their sub-steps too). Cheap to build,
        /// read by the zone-entry probe.</summary>
        public static string BootTrace = "";
        static double _lastMark;
        internal static void Mark(System.Diagnostics.Stopwatch sw, string name)
        {
            double t = sw.Elapsed.TotalMilliseconds;
            BootTrace += $"{name}:{t - _lastMark:0.0}ms ";
            _lastMark = t;
        }
        /// <summary>Sub-step append hook for the nested ctors (their own stopwatch deltas).</summary>
        internal static void Trace(string chunk) => BootTrace += chunk;

        // Human-readable model size for boot logs / UI (the enum's own name is terse).
        static string SizeLabel(Qwen3_5Size size) => size switch
        {
            Qwen3_5Size.B0_8 => "0.8B",
            Qwen3_5Size.B2   => "2B",
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

        // Self-registering LLMRegistry catalog entries — model pickers (NPC inspector etc.)
        // discover these by reflection; nothing else to extend when a new size lands.
        [LLMEntry(0)]
        static LLMRegistry.Entry RegistryEntry0_8B() => new LLMRegistry.Entry
        {
            id = "Qwen3.5-0.8B",
            create = (q, kv, maxLen) => new Qwen3_5ForCausalLM(Qwen3_5Size.B0_8, quantization: q, kv_quant: kv, maxModelLength: maxLen),
            prewarm = () => Prewarm(),
        };
        [LLMEntry(1)]
        static LLMRegistry.Entry RegistryEntry2B() => new LLMRegistry.Entry
        {
            id = "Qwen3.5-2B",
            create = (q, kv, maxLen) => new Qwen3_5ForCausalLM(Qwen3_5Size.B2, quantization: q, kv_quant: kv, maxModelLength: maxLen),
            prewarm = () => Prewarm(),
        };

        /// <summary>
        /// One-call scene-start prewarm — run this as a coroutine while the player is doing
        /// something else (walking around, in a menu) and constructing/loading the model later
        /// becomes hitch-free. It (a) starts the background tokenizer parse and caches the result
        /// (the parse garbage otherwise triggers a ~300 ms GC collection mid-load), and (b) compiles
        /// every compute kernel, one per frame (the driver's one-time first-dispatch cost — up to
        /// ~800 ms for the biggest kernel — would otherwise land inside the loading window).
        /// Idempotent; needs no model instance and no weights.
        /// </summary>
        public static IEnumerator Prewarm(string tokenizer_path = "Assets/DeepUnity/InferenceEngine/LLM/Qwen3_5/Qwen3_5TokenizerFast.json")
        {
            GetOrCreateTokenizer(tokenizer_path, p => new Qwen3_5TokenizerFast(p, load_async: true));
            yield return Qwen3_5Modeling.Qwen3_5Model.PrewarmKernels();
            // Sweep the tokenizer-parse garbage NOW, spread over frames — otherwise the first big
            // load-time allocation triggers one blocking ~230 ms collection mid-walk-up.
            while (UnityEngine.Scripting.GarbageCollector.CollectIncremental(2_000_000UL))
                yield return null;
        }

        /// <inheritdoc/>
        public override void Release()
        {
            model?.Dispose();   // the weights loader emits the standardized [GPU] released line
            model = null;
            OnReleased(); // unhook editor event + suppress the finalizer (it would double-release off-thread)
        }

        /// <inheritdoc/>
        public override IEnumerator ReleaseSlow(long bytesPerFrame = 64_000_000)
        {
            var m = model;
            model = null;      // the instance reads as released immediately; buffers trickle out
            OnReleased();
            if (m != null) yield return m.DisposeSlow(bytesPerFrame);
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
            tokenizer?.ResetStreamDecode();   // buffer split multibyte chars so they don't render as □ boxes
            if (tokenizer != null)
                onTokenGenerated?.Invoke(tokenizer.DecodeStreamStep(tokenId));
            else
                onTokenGenerated?.Invoke(tokenId.ToString() + " ");
            yield return null;

            int tokensPerFrame = System.Math.Max(1, InferencePerf.LlmDecodeTokensPerFrame);
            for (int t = 0; t < max_new_tokens - 1; t++)
            {
                // #29 reverse arbiter: audible SILENCE with synthesis pending outranks tok/s —
                // hold this token's burst so the voice refills faster. Hard-capped (InferencePerf.LlmHoldMaxFrames frames)
                // so decode always makes progress no matter what the voice reports.
                for (int hold = 0; FramePacing.TtsStarving && hold < InferencePerf.LlmHoldMaxFrames; hold++)
                { FramePacing.LlmDeferrals++; yield return null; }
                bool cede = FramePacing.TtsStarving;

                Stopwatch sw = Stopwatch.StartNew();
                var d = DecodeStep(Tensor.Constant(tokenId), sampled, temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
                while (d.MoveNext()) yield return d.Current;
                tokenId = sampled[0];
                if (tokenId == Qwen3_5Modeling.Qwen3_5Config.EOS_TOKEN_ID) break;

                if (tokenizer != null)
                    onTokenGenerated?.Invoke(tokenizer.DecodeStreamStep(tokenId));
                else
                    onTokenGenerated?.Invoke(tokenId.ToString() + " ");
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                // hand a frame back to rendering every LlmDecodeTokensPerFrame tokens (forced every
                // token while a voice is starving, so the TTS pump keeps its frames)
                if (cede || t % tokensPerFrame == tokensPerFrame - 1) yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        // One autoregressive decode step (single token). Two modes:
        //  - FAST (default): issue the forward + sampler SYNCHRONOUSLY and block once on the token
        //    readback. Now that a token is only ~a few ms of GPU work (the #31 coalesced kernels),
        //    this runs decode at COMPUTE speed. The old async-readback path spread every token over
        //    ~3-4 frames (ForwardYielding's frame + AsyncGPUReadback's ~2 + the loop's trailing
        //    yield), which capped play-mode decode at ~framerate/4 tok/s no matter how fast the GPU
        //    was — the "high FPS but slow text" symptom. The caller still yields once per token, so
        //    the app stays responsive (renders ~1 frame per token during a reply).
        //  - CEDING: while a speaking TTS voice is starving (3D concurrent talk), fall back to the
        //    async yielding path so the voice keeps its frames — audio continuity outranks tok/s
        //    there, and the #29 reverse arbiter already holds the burst anyway.
        IEnumerator DecodeStep(Tensor input, int[] result, float temperature, int top_k, float top_p,
                               float min_p, float presence_penalty, float repetition_penalty)
        {
            // #32 AutoTune: the async path also serves when the MEASURED per-token stall doesn't
            // fit this device's frame budget (InferencePerf decides once per session from the
            // first sync tokens' cost) — sync-at-any-price undid #20's smoothness on strong GPUs.
            bool cede = FramePacing.TtsStarving || !InferencePerf.UseSyncDecode;
            if (cede)
            {
                var e = model.ForwardYielding(input, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;
                var s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, result);
                while (s.MoveNext()) yield return s.Current;
            }
            else
            {
                var sw = Stopwatch.StartNew();
                model.Forward(input, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                result[0] = model.Sample(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
                InferencePerf.NoteSyncTokenMs((float)sw.Elapsed.TotalMilliseconds);   // probe feed (no-op once decided)
            }
        }

        protected override IEnumerator InitializeChatCore(string system_prompt = "")
        {
            // Warmup is part of initialization: kernel compiles + throwaway forwards happen here,
            // behind the caller's loading screen, never on the first reply. Idempotent.
            CurrentPhase = "boot (weights+warmup)";
            yield return Warmup();

            while (!IsReady) yield return new WaitForSeconds(0.01f);
            CurrentPhase = "idle";
            Assert.AreNotEqual(system_prompt, null);
            lastSystemPrompt = system_prompt;

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

            // Disk-cached system prompt: same prompt + same weights + same kv quant -> restore
            // the KV/SSM state (frame-budgeted uploads) instead of recomputing the chunked
            // prefill. v2 format persists FP32, FP16 AND INT8 KV, so this now triggers for every
            // NPC config (they run FP16/INT8 KV), in every ConversationMode.
            bool restoredFromDisk = false;
            string cacheFile = null;
            ulong promptHash = 0;
            if (SystemPromptDiskCache && DiskKVCache && Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE)
            {
                promptHash = PromptCacheKey(ids);
                cacheFile = System.IO.Path.Combine(CacheDir(), $"qwen35_prompt_{promptHash:x16}.kv");
                if (System.IO.File.Exists(cacheFile))
                {
                    CurrentPhase = "kv-restore";
                    var load = model.cache.LoadYielding(cacheFile, promptHash, ok => restoredFromDisk = ok);
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
                    var save = model.cache.SaveYielding(cacheFile, promptHash);
                    while (save.MoveNext()) yield return save.Current;
                }
            }

            CurrentPhase = "idle";
            LogBootSummary(sw.Elapsed.TotalMilliseconds, restoredFromDisk);

            isFreshlyInitialized = true;
            yield return true;
        }

        protected override IEnumerator ChatCore(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false)
        {
            if (!IsReady) throw new Exception("Call InitializeChat before Chat.");
            chatCancelRequested = false;

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

            // A cancel that landed during the PREFILL exits here: the chunked prefill always runs
            // to completion (breaking mid-chunk would tear the turn template off the KV), then the
            // first sample/emit is skipped — the turn ends EMPTY, and the next Chat closes it
            // exactly like any truncated turn.
            int[] sampled = new int[1];
            int tokenId = -1;
            int genTokens = 0;
            bool canceledInPrefill = chatCancelRequested;
            if (canceledInPrefill)
                ConsoleMessage.Info("Qwen3.5 reply canceled during prefill (KV consistent, empty turn closes as usual).");
            else
            {
                var s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, sampled);
                while (s.MoveNext()) yield return s.Current;
                tokenId = sampled[0];
                tokenizer.ResetStreamDecode();   // buffer split multibyte chars so they don't render as □ boxes
                onTokenGenerated?.Invoke(tokenizer.DecodeStreamStep(tokenId));
                genTokens = 1;
                yield return null;
            }

            var genSw = Stopwatch.StartNew();   // wall-clock over the decode loop, for the tok/s report
            int holdFrames = 0, cededToks = 0;  // diagnostic split for the tok/s report
            int tokensPerFrame = System.Math.Max(1, InferencePerf.LlmDecodeTokensPerFrame);
            for (int t = 0; !canceledInPrefill && t < max_new_tokens - 1; t++)
            {
                // #29 reverse arbiter: audible SILENCE with synthesis pending outranks tok/s —
                // hard-capped hold (see the Chat loop; liveness guaranteed).
                for (int hold = 0; FramePacing.TtsStarving && hold < InferencePerf.LlmHoldMaxFrames; hold++)
                { FramePacing.LlmDeferrals++; holdFrames++; yield return null; }
                bool cede = FramePacing.TtsStarving;
                if (cede) cededToks++;

                Stopwatch sw = Stopwatch.StartNew();
                var d = DecodeStep(Tensor.Constant(tokenId), sampled, temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
                while (d.MoveNext()) yield return d.Current;
                tokenId = sampled[0];
                if (tokenId == Qwen3_5Modeling.Qwen3_5Config.EOS_TOKEN_ID || chatCancelRequested)
                {
                    ConsoleMessage.Info(chatCancelRequested
                        ? "Qwen3.5 reply canceled at a token boundary (KV consistent, turn closes as usual)."
                        : "Qwen3.5 ended the response.");
                    break;
                }

                string tokenStr = tokenizer.DecodeStreamStep(tokenId);
                onTokenGenerated?.Invoke(tokenStr);
                genTokens++;
                TokensPerSecond = sw.ElapsedMilliseconds > 0 ? 1000f / sw.ElapsedMilliseconds : 0f;
                // hand a frame back to rendering every LlmDecodeTokensPerFrame tokens (forced every
                // token while a voice is starving, so the TTS pump keeps its frames)
                if (cede || t % tokensPerFrame == tokensPerFrame - 1) yield return null;
            }

            // in-game decode speed (the honest play-mode number — includes per-token frame yields,
            // unlike the tight-loop RTF probe). Lets us confirm the coalesced kernels + sync decode
            // are actually reaching the player.
            double genSec = genSw.Elapsed.TotalSeconds;
            if (genSec > 0 && genTokens > 0)
                ConsoleMessage.Info($"Qwen3.5 decode: {genTokens} tokens in {genSec:F2}s = {genTokens / genSec:F1} tok/s (in-game; " +
                                    $"held {holdFrames} frames for starving TTS, {cededToks} ceded tokens, " +
                                    $"sync={InferencePerf.UseSyncDecode}).");

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

        // Shared cache directory for every Qwen3.5 disk-cached KV state (system prompts AND
        // whole conversations) — persistentDataPath/DeepUnity.
        static string CacheDir()
        {
            string dir = System.IO.Path.Combine(Application.persistentDataPath, "DeepUnity");
            System.IO.Directory.CreateDirectory(dir);
            return dir;
        }

        // FNV-1a over the prompt token ids, weight path (= model size + weight quant), cache
        // capacity and KV quant — any of these changing must invalidate the cached state. The
        // same value is the file-name suffix AND the header contextHash the load validates.
        ulong PromptCacheKey(System.Collections.Generic.List<float> ids)
        {
            ulong h = 14695981039346656037UL;
            void Mix(ulong v) { h ^= v; h *= 1099511628211UL; }
            foreach (var id in ids) Mix((ulong)(long)id);
            foreach (char c in path) Mix(c);
            Mix((ulong)model.cache.Capacity);
            Mix((ulong)(int)model.KV);
            return h;
        }

        // ---------------------------------------------------------------- conversation KV persistence (WS-G)

        // Conversation identity: weight path (model size + weight quant), cache capacity, KV quant
        // and the system prompt STRING (the transcript itself is NOT hashed — the file name must
        // stay stable across turns so each save overwrites the previous snapshot; staleness vs the
        // caller's live transcript is arbitrated through acceptUserState instead).
        ulong ConversationContextHash(string systemPrompt)
        {
            ulong h = 14695981039346656037UL;
            void Mix(ulong v) { h ^= v; h *= 1099511628211UL; }
            foreach (char c in path) Mix(c);
            Mix((ulong)model.cache.Capacity);
            Mix((ulong)(int)model.KV);
            foreach (char c in systemPrompt ?? "") Mix(c);
            return h;
        }

        string ConversationCacheFile(string key, string systemPrompt)
            => System.IO.Path.Combine(CacheDir(),
                $"qwen35_conv_{SanitizeKey(key)}_{ConversationContextHash(systemPrompt):x16}.kv");

        static string SanitizeKey(string key)
        {
            if (string.IsNullOrEmpty(key)) return "npc";
            var sb = new System.Text.StringBuilder(key.Length);
            foreach (char c in key)
                sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');
            return sb.ToString();
        }

        // Extra-state blob riding in the conversation cache file (Qwen3_5Cache stores it opaquely):
        //   uint8  CONV_EXTRA_VERSION
        //   uint8  isFreshlyInitialized      (open assistant turn: Chat() must emit <|im_end|> first)
        //   int32  tokenSeen uint count (== vocab) + raw bytes (presence/repetition-penalty counts)
        //   int32  userState UTF-8 byte count + bytes (opaque caller state, e.g. the NPC transcript)
        const byte CONV_EXTRA_VERSION = 1;

        byte[] BuildConversationExtra(byte[] tokenSeenRaw, string userState)
        {
            using var ms = new MemoryStream();
            using var bw = new BinaryWriter(ms);
            bw.Write(CONV_EXTRA_VERSION);
            bw.Write(isFreshlyInitialized);
            bw.Write(tokenSeenRaw.Length / 4);
            bw.Write(tokenSeenRaw);
            byte[] us = string.IsNullOrEmpty(userState)
                ? Array.Empty<byte>() : System.Text.Encoding.UTF8.GetBytes(userState);
            bw.Write(us.Length);
            bw.Write(us);
            bw.Flush();
            return ms.ToArray();
        }

        bool TryParseConversationExtra(byte[] extra, out bool fresh, out byte[] tokenSeenRaw, out string userState)
        {
            fresh = false; tokenSeenRaw = null; userState = null;
            try
            {
                if (extra == null || extra.Length < 10) return false;
                using var br = new BinaryReader(new MemoryStream(extra));
                if (br.ReadByte() != CONV_EXTRA_VERSION) return false;
                fresh = br.ReadBoolean();
                int seenUints = br.ReadInt32();
                if (seenUints != model.VocabSize) return false;
                tokenSeenRaw = br.ReadBytes(seenUints * 4);
                if (tokenSeenRaw.Length != seenUints * 4) return false;
                int usLen = br.ReadInt32();
                if (usLen < 0 || usLen > (64 << 20)) return false;
                byte[] us = br.ReadBytes(usLen);
                if (us.Length != usLen) return false;
                userState = usLen == 0 ? "" : System.Text.Encoding.UTF8.GetString(us);
                return true;
            }
            catch { return false; }
        }

        /// <summary>
        /// Persists the ENTIRE current conversation state to disk under <paramref name="key"/>:
        /// the KV/SSM prefix (all cached tokens), the sampler's token-seen counts and the
        /// open-assistant-turn flag, plus <paramref name="userState"/> verbatim (the NPC
        /// transcript). Budgeted readbacks + worker-thread IO — runs behind gameplay. No-op when
        /// <see cref="LLM.DiskKVCache"/> is off, the model isn't ready or nothing is cached.
        /// </summary>
        public override void DeleteConversationKV(string key)
        {
            try
            {
                string dir = CacheDir();
                if (!System.IO.Directory.Exists(dir)) return;
                foreach (var f in System.IO.Directory.GetFiles(dir, $"qwen35_conv_{SanitizeKey(key)}_*.kv"))
                    System.IO.File.Delete(f);
            }
            catch (System.Exception e) { ConsoleMessage.Warning($"Qwen3.5 DeleteConversationKV: {e.Message}"); }
        }

        protected override IEnumerator SaveConversationKVCore(string key, string userState = null,
                                                              string system_prompt = null)
        {
            if (!DiskKVCache || !Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE) yield break;
            if (!IsReady || model.cache.CachedTokenCount <= 0) yield break;

            byte[] seen = null;
            var rb = model.ReadTokenSeenRaw(b => seen = b);
            while (rb.MoveNext()) yield return rb.Current;
            if (seen == null) yield break;   // readback error — skip this save, keep the old file

            string sysPrompt = system_prompt ?? lastSystemPrompt;
            byte[] extra = BuildConversationExtra(seen, userState);
            CurrentPhase = "kv-save";
            var save = model.cache.SaveYielding(ConversationCacheFile(key, sysPrompt),
                                                ConversationContextHash(sysPrompt), extra);
            while (save.MoveNext()) yield return save.Current;
            CurrentPhase = "idle";
        }

        /// <summary>
        /// Restores a conversation saved by <see cref="SaveConversationKV"/> with the same key /
        /// weights / quant / kv quant / system prompt. Waits for weights + warmup internally
        /// (same boot path as InitializeChat), validates the header hash and every payload size,
        /// and reports false on ANY mismatch — the caller falls back to re-prefilling. On success
        /// the KV/SSM prefix, token-seen counts and the open-turn flag are live again and the
        /// chat continues exactly where it left off (no InitializeChat needed).
        /// </summary>
        protected override IEnumerator TryRestoreConversationKVCore(string key, Action<bool> onResult,
            string system_prompt = null, Func<string, bool> acceptUserState = null)
        {
            if (!DiskKVCache || !Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE)
            {
                onResult?.Invoke(false);
                yield break;
            }
            string sysPrompt = system_prompt ?? lastSystemPrompt;
            string file = ConversationCacheFile(key, sysPrompt);
            if (!System.IO.File.Exists(file)) { onResult?.Invoke(false); yield break; }

            // weights + kernels must be live before any KV upload — same boot path as InitializeChat
            CurrentPhase = "boot (weights+warmup)";
            yield return Warmup();
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            model.ResetCache();   // clean slate (zeroes SSM states + token-seen)

            bool restored = false, extraFresh = false;
            byte[] extraSeen = null;
            string extraUserState = null;
            CurrentPhase = "kv-restore";
            var load = model.cache.LoadYielding(file, ConversationContextHash(sysPrompt),
                ok => restored = ok,
                extra =>
                {
                    // parse + caller veto BEFORE any GPU upload — a reject costs only the file read
                    if (!TryParseConversationExtra(extra, out extraFresh, out extraSeen, out extraUserState))
                        return false;
                    return acceptUserState == null || acceptUserState(extraUserState);
                });
            while (load.MoveNext()) yield return load.Current;

            if (!restored)
            {
                try { System.IO.File.Delete(file); } catch { }   // stale/corrupt/vetoed — next clean close rewrites it
                model.ResetCache();   // a partial upload may have dirtied the state
                CurrentPhase = "idle";
                onResult?.Invoke(false);
                yield break;
            }

            // sampler state rides along so a restored chat penalizes exactly like a true resume
            var up = model.UploadTokenSeenRaw(extraSeen);
            while (up.MoveNext()) yield return up.Current;

            isFreshlyInitialized = extraFresh;
            lastSystemPrompt = sysPrompt;
            CurrentPhase = "idle";
            ConsoleMessage.Info($"Qwen3.5 conversation restored from disk ({model.cache.CachedTokenCount} tokens)");
            onResult?.Invoke(true);
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
            // #32 adaptive prefill packing: InferencePerf grows/shrinks the per-frame slice pack
            // off measured prefill frame times (60 fps anchor, Smooth⇄Speed-biased) — fast GPUs
            // open dialogues in a fraction of the old fixed-pack time.
            int step = 0;
            for (int start = 0; start < ids.Count; start += CHUNK)
            {
                int len = Math.Min(CHUNK, ids.Count - start);
                float[] part = new float[len];
                for (int i = 0; i < len; i++) part[i] = ids[start + i];
                var e = model.ForwardYielding(Tensor.Constant(part), useCache: true, lastPosOnly: true);
                while (e.MoveNext())
                    if (++step % InferencePerf.EffectivePrefillPack() == 0)
                    {
                        float tYield = Time.realtimeSinceStartup;
                        yield return e.Current;
                        InferencePerf.NotePrefillFrameMs((Time.realtimeSinceStartup - tYield) * 1000f);
                    }
            }
        }

        void AppendTextTokens(string text, System.Collections.Generic.List<float> dst)
        {
            (Tensor t, _) = tokenizer.Encode(text, add_special_tokens: false);
            for (int i = 0; i < t.Size(-1); i++) dst.Add(t[i]);
        }
    }
}
