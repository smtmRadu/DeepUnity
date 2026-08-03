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

            int tokensPerFrame = BackendTradeoffTable.DecodeTokensPerFrame;
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
                // hand a frame back to rendering every tokensPerFrame tokens — the dial's decode row
                // (forced every token while a voice is starving, so the TTS pump keeps its frames)
                if (cede || t % tokensPerFrame == tokensPerFrame - 1) yield return null;
            }

            TokensPerSecond = 0f;
            yield return true;
        }

        // One autoregressive decode step (single token). Two modes, selected by
        // BackendTradeoffTable.UseSyncDecode — a const whose history of pointing BOTH ways lives on
        // that field and its class docs:
        //  - SLICED (shipped since 2026-08-02, the smoothness mandate): ForwardYielding issues the
        //    forward in InferencePerf.LlmDecodeSliceLayers-layer slices (its seqLen==1 path), the
        //    lm_head in its own frame, then SampleYielding waits on an AsyncGPUReadback for the
        //    token — no frame carries the token's whole ~30-55 ms GPU burst, the readback gates the
        //    next token, and tok/s falls to ~framerate/frames-per-token (~5-8 on the 1650). Priced
        //    in deliberately: speech at ~3 words/s is the pacing bottleneck, not text.
        //  - SYNC (2026-07-26 → 2026-08-02, kept compilable for A/B archaeology): issue everything
        //    in one burst and block once on the token readback — decode at COMPUTE speed (~12 tok/s
        //    on the 1650), the whole burst in one frame. Those are the 33 ms mean / 55 ms p95 GEN
        //    frames the path was retired for.
        // A starving voice forces the yielding path in EITHER mode — audio continuity outranked
        // tok/s even while tok/s was still a criterion (#29).
        IEnumerator DecodeStep(Tensor input, int[] result, float temperature, int top_k, float top_p,
                               float min_p, float presence_penalty, float repetition_penalty)
        {
            bool yielding = FramePacing.TtsStarving || !BackendTradeoffTable.UseSyncDecode;
            if (yielding)
            {
                var e = model.ForwardYielding(input, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                while (e.MoveNext()) yield return e.Current;
                var s = model.SampleYielding(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty, result);
                while (s.MoveNext()) yield return s.Current;
            }
            else
            {
                model.Forward(input, useCache: Qwen3_5Modeling.Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                result[0] = model.Sample(temperature, top_k, top_p, min_p, presence_penalty, repetition_penalty);
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

            // The system turn, per Qwen3_5ChatTemplate (= the vendored chat_template.jinja L46-64).
            // Every tag is emitted as its own token ID and only the text BETWEEN tags is encoded;
            // the fragments come from the template class so the ids and the template's text spelling
            // can never drift apart. The .Trim() is the template's `| trim` on message content
            // (L55) — without it, trailing whitespace left in an authored prompt (the inspector's
            // descriptionAndRules is a TextArea, so this happens) shifts every byte after it and
            // silently forks the KV-cache key away from what apply_chat_template hashes.
            var ids = new System.Collections.Generic.List<float>();
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID);
            AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.SystemRoleLine, ids);
            (Tensor sysTok, _) = tokenizer.Encode((system_prompt ?? "").Trim(), add_special_tokens: false, truncation: true, max_length: 2048);
            for (int i = 0; i < sysTok.Size(-1); i++) ids.Add(sysTok[i]);
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID);
            AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.TurnEndTail, ids);

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
                // named after the OWNER when there is one (one file per NPC, overwritten when its prompt
                // changes), else content-addressed as before. The hash still gates the load either way.
                cacheFile = System.IO.Path.Combine(CacheDir(), string.IsNullOrEmpty(CacheOwnerKey)
                    ? $"qwen35_prompt_{promptHash:x16}.kv"
                    : $"qwen35_prompt_{SanitizeKey(CacheOwnerKey)}.kv");
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
            => ChatTurn(prompt, false, onTokenGenerated, max_new_tokens, temperature, top_k, top_p, min_p,
                        presence_penalty, repetition_penalty, enable_thinking);

        protected override IEnumerator ChatToolResultCore(string toolResultJson, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false)
            => ChatTurn(toolResultJson, true, onTokenGenerated, max_new_tokens, temperature, top_k, top_p, min_p,
                        presence_penalty, repetition_penalty, enable_thinking);

        // Shared turn body for Chat / ChatToolResult — the ONLY difference is how the incoming
        // user turn is rendered: a tool result rides inside <tool_response> SPECIAL tokens
        // (<|im_start|>user\n<tool_response>\n{json}\n</tool_response><|im_end|>\n), exactly the
        // shape Qwen's chat template gives a role:"tool" message; a plain prompt renders as the
        // usual user turn.
        // Every literal in here comes from Qwen3_5ChatTemplate (the vendored chat_template.jinja):
        // the tags themselves are token IDS from Qwen3_5Config and the class supplies only the text
        // that sits between them, so there is nothing to hand-copy and nothing to drift. The
        // template splits the same bytes at other seams — it writes '<|im_start|>user' with no
        // newline (L133) and gets it from the head of '\n<tool_response>\n' (L135), where this emits
        // "user\n" and then the tag — so compare BYTES, not line-for-line shapes, before "fixing"
        // anything below.
        IEnumerator ChatTurn(string prompt, bool asToolResponse, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false)
        {
            if (!IsReady) throw new Exception("Call InitializeChat before Chat.");
            chatCancelRequested = false;

            var ids = new System.Collections.Generic.List<float>();

            if (!isFreshlyInitialized)
            {
                // Close prior assistant turn that Chat() left open (we broke before forwarding
                // <|im_end|>) — the template's L130 turn end, paid one turn late.
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID);
                AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.TurnEndTail, ids);
            }
            isFreshlyInitialized = false;

            // Open user turn (template L88; L133-141 for the tool-result flavour).
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID);
            AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.UserRoleLine, ids);
            if (asToolResponse)
            {
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.TOOL_RESPONSE_OPEN_TOKEN_ID);
                AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.ToolResponseOpenTail, ids);
            }
            // .Trim() = the template's `| trim` on user (L63) and tool (L82) content. A leading space
            // in the player's input field survives AskNPC's IsNullOrWhiteSpace check, and the tool
            // result arrives as serialized JSON whose framing whitespace is not ours to keep.
            (Tensor userTok, _) = tokenizer.Encode((prompt ?? "").Trim(), add_special_tokens: false, truncation: true, max_length: 2048);
            for (int i = 0; i < userTok.Size(-1); i++) ids.Add(userTok[i]);
            if (asToolResponse)
            {
                AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.ToolResponseCloseHead, ids);
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.TOOL_RESPONSE_CLOSE_TOKEN_ID);
            }
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_END_TOKEN_ID);
            AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.TurnEndTail, ids);

            // Open the assistant turn with its thinking prefix — the template's generation prompt
            // (L147-153). Thinking OFF still emits an EMPTY <think></think> block, which is the
            // template's default branch (it takes it whenever enable_thinking is undefined), so the
            // model always starts its answer past a closed </think> rather than never seeing one.
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID);
            AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.AssistantRoleLine, ids);
            ids.Add(Qwen3_5Modeling.Qwen3_5Config.THINK_OPEN_TOKEN_ID);
            if (enable_thinking)
            {
                AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.ThinkPrefillTail, ids);
            }
            else
            {
                AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.EmptyThinkMid, ids);
                ids.Add(Qwen3_5Modeling.Qwen3_5Config.THINK_CLOSE_TOKEN_ID);
                AppendTextTokens(Qwen3_5Modeling.Qwen3_5ChatTemplate.EmptyThinkTail, ids);
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
            int tokensPerFrame = BackendTradeoffTable.DecodeTokensPerFrame;
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
                // hand a frame back to rendering every tokensPerFrame tokens — the dial's decode row
                // (forced every token while a voice is starving, so the TTS pump keeps its frames)
                if (cede || t % tokensPerFrame == tokensPerFrame - 1) yield return null;
            }

            // in-game decode speed (the honest play-mode number — includes per-token frame yields,
            // unlike the tight-loop RTF probe). Lets us confirm the coalesced kernels + sync decode
            // are actually reaching the player.
            double genSec = genSw.Elapsed.TotalSeconds;
            if (genSec > 0 && genTokens > 0)
                ConsoleMessage.Info($"Qwen3.5 decode: {genTokens} tokens in {genSec:F2}s = {genTokens / genSec:F1} tok/s (in-game; " +
                                    $"held {holdFrames} frames for starving TTS, {cededToks} ceded tokens, " +
                                    $"sync={BackendTradeoffTable.UseSyncDecode}).");

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
            // .Trim() so the hash describes what is actually TOKENIZED, not what the caller handed in
            // (fix 2026-07-28). The turn encoders trim now, per the template's `| trim`; hashing the
            // raw string left this hash blind to that change, so a conversation saved yesterday from a
            // prompt with trailing whitespace passed validation today while its KV still encoded the
            // extra newline token and every position after it was shifted by one. Untrimmed
            // "…\n" + "\nYou are…" tokenizes [system][\n][\n][You]; the trimmed span gives
            // [system][\n\n][You]. Same hash, different KV — the worst kind of cache hit.
            foreach (char c in (systemPrompt ?? "").Trim()) Mix(c);
            return h;
        }

        // ONE file per conversation owner. The context hash used to be part of the name, which meant
        // editing an NPC's system prompt silently orphaned its saved conversation (and left the old file
        // behind forever) instead of replacing it. The hash still rides in the header, so a prompt change
        // fails validation, the stale file is deleted and the next clean close rewrites this same path.
        string ConversationCacheFile(string key, string systemPrompt)
            => System.IO.Path.Combine(CacheDir(), $"qwen35_conv_{SanitizeKey(key)}.kv");

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
                // BOTH shapes: the current one-file-per-owner name AND the legacy
                // qwen35_conv_<owner>_<contexthash>.kv written before the hash left the filename. The
                // pattern used to be the legacy one only, so after the rename this method deleted
                // nothing at all — a "forget this conversation" that quietly forgot nothing, and it
                // also left every pre-rename file orphaned on disk forever.
                string k = SanitizeKey(key);
                foreach (var f in System.IO.Directory.GetFiles(dir, $"qwen35_conv_{k}.kv"))
                    System.IO.File.Delete(f);
                foreach (var f in System.IO.Directory.GetFiles(dir, $"qwen35_conv_{k}_*.kv"))
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

            // Detailed per-step breakdown. UNCOMMENTED 2026-07-26: the summed `load` figure
            // above cannot tell you WHICH step owns a hitch, and the ~4 s of 20 fps at play
            // start was being attributed by guesswork. Costs one log line per boot.
            double blocking = w.bootTokenizerMs + w.bootKernelsMs + w.allocMs + w.bootCacheMs + w.bootRopeMs + w.bootScratchMs;
            double total = blocking + w.uploadMs + systemPromptMs;
            ConsoleMessage.Info(
                $"Qwen3.5 model booted up — {total:0} ms total\n" +
                $"   tokenizer ctor (main thread) : {w.bootTokenizerMs:0} ms\n" +
                $"   compute kernels lookup       : {w.bootKernelsMs:0} ms\n" +
                $"   weight manifest build        : {w.allocMs:0} ms (buffers created lazily during upload)\n" +
                $"   kv cache alloc               : {w.bootCacheMs:0} ms\n" +
                $"   rope kick (async)            : {w.bootRopeMs:0} ms\n" +
                $"   scratch buffers alloc        : {w.bootScratchMs:0} ms\n" +
                $"   = blocking (one frame)       : {blocking:0} ms\n" +
                $"   rope compute (async)         : {w.ropeAsyncMs:0} ms (overlaps upload)\n" +
                $"   weight stream (async)        : {w.uploadMs:0} ms over {w.uploadFrames} frames, worst slice {w.worstUploadMs:0.0} ms\n" +
                $"   kernel warmup (behind load)  : {w.warmupMs:0} ms (0 = warmup didn't run)\n" +
                $"   system prompt cache          : {systemPromptMs:0} ms" +
                (promptFromDisk ? " (restored from disk)" : ""));

            // The pacing the numbers above were produced under. This line replaces the old
            // AutoTune verdict (which could only be printed after the first reply had already been
            // measured): the dial is fixed, so a session can state its budgets at boot and any later
            // "why was this load/reply that speed?" is answered from the log instead of guessed.
            ConsoleMessage.Info(BackendTradeoffTable.Summary);
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
            // Prefill pacing = the dial, nothing measured (BackendTradeoffTable, 2026-07-26). `step`
            // counts ForwardYielding's yields, which are one per transformer layer plus one at the
            // end — so on a 24-layer model a whole CHUNK is 25 steps, and the top row's 25
            // steps/frame is exactly "one 8-token chunk per frame", the implementation limit.
            int step = 0;
            for (int start = 0; start < ids.Count; start += CHUNK)
            {
                int len = Math.Min(CHUNK, ids.Count - start);
                float[] part = new float[len];
                for (int i = 0; i < len; i++) part[i] = ids[start + i];
                // Only the LAST chunk's logits are ever read (ChatTurn samples the first reply token
                // right after this loop; InitializeChat reads none at all). Running the vocab GEMV on
                // every 8-token chunk streamed ~509 MB of fp16 lm_head per chunk — ~30% of prefill
                // GPU time — and threw all of it away. 2026-07-26.
                bool lastChunk = start + len >= ids.Count;
                var e = model.ForwardYielding(Tensor.Constant(part), useCache: true, lastPosOnly: true,
                                              computeLogits: lastChunk);
                while (e.MoveNext())
                    if (++step % BackendTradeoffTable.PrefillStepsPerFrame == 0)
                        yield return e.Current;
            }
        }

        void AppendTextTokens(string text, System.Collections.Generic.List<float> dst)
        {
            (Tensor t, _) = tokenizer.Encode(text, add_special_tokens: false);
            for (int i = 0; i < t.Size(-1); i++) dst.Add(t[i]);
        }
    }
}
