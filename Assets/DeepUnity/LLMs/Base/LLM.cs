using System;
using System.Collections;
using System.Collections.Generic;

namespace DeepUnity
{
    /// <summary>
    /// Abstract base for every DeepUnity full-GPU causal LLM (Gemma3, Qwen3.5, ...).
    ///
    /// It pins down the shared public surface so models drop in interchangeably and expose
    /// near-identical APIs: <see cref="IsReady"/>, <see cref="TokensPerSecond"/>,
    /// <see cref="Warmup"/>, <see cref="Predict"/>, <see cref="Generate"/>,
    /// <see cref="InitializeChat"/>, <see cref="Chat"/> and GPU release. Concrete models own
    /// their own weights / tokenizer / cache and the actual inference kernels.
    ///
    /// The editor play-mode hook and finalizer live here once and both route through
    /// <see cref="Release"/>, so a model is freed from the GPU on play-mode exit (editor) or on
    /// finalize (build). Sampling knobs that a given model doesn't support
    /// (penalties, thinking) are accepted on the shared signatures and simply ignored by that model.
    /// </summary>
    public abstract class LLM
    {
        protected LLM()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeChanged;
#endif
        }

        ~LLM()
        {
            Release();
        }

#if UNITY_EDITOR
        private void OnPlayModeChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode)
                Release();
        }
#endif

        /// <summary>Architecture + sampling-default descriptor for this model.</summary>
        public abstract LLMConfig Config { get; }

        /// <summary>True once the weights are uploaded and the tokenizer (if any) has finished loading.</summary>
        public abstract bool IsReady { get; }

        /// <summary>Rolling decode speed of the most recent generation step (0 while idle).</summary>
        public float TokensPerSecond { get; protected set; }

        // ---- prompt-cache hitch tuning (shared by every model's KV disk cache) ----------------
        // Saving/restoring the system-prompt KV state is spread across frames so it can run
        // behind gameplay. If it ever drops frames again, these are the knobs — every model's
        // cache (Qwen3_5Cache, Gemma3Cache, ...) reads them, so one nudge behaves the same
        // across models. All trade restore latency for smoothness; turn DOWN for smaller
        // per-frame cost:
        //   UploadFrameBudgetMs   max milliseconds of SetData copy work per frame while
        //                         restoring.
        //   UploadChunkFloats     floats per SetData call; smaller chunks let the budget cut
        //                         finer (64k floats = 256 KB ≈ 0.05-0.15 ms per call).
        //   SaveReadbacksInFlight max concurrent GPU readbacks while saving — also caps how
        //                         many readback→managed copies can land on a single frame
        //                         (results must be copied the frame they complete; deferring
        //                         isn't an option).
        // The weight STREAMING budget is separate: UPLOAD_BUDGET_BYTES in each model's
        // *Weights.cs — that one governs the boot-time worst slice, not these.
        // If the knobs ever stop being enough, the next lever is streaming the restore: parse
        // layer i on the worker WHILE layer i-1 uploads (reused scratch arrays) instead of
        // parsing the whole file up front — less transient managed memory, less GC.
        public static float UploadFrameBudgetMs = 0.5f;
        public static int UploadChunkFloats = 64 * 1024;
        public static int SaveReadbacksInFlight = 1;

        /// <summary>
        /// Coarse, frame-accurate tag of what the LLM machinery is doing right now ("idle" when
        /// nothing). Models write it around their long-running phases (kernel prewarm, weight
        /// stream + warmup, kv restore/save, prefill, decode) so a frame-spike probe can attribute
        /// slow frames to a phase instead of guessing.
        /// </summary>
        public static string CurrentPhase = "idle";

        /// <summary>
        /// Compiles every compute kernel (the one-time first-dispatch cost) behind a loading screen so the
        /// first real reply isn't a freeze. Default is a no-op; models with a warmup path override it.
        /// Idempotent. <see cref="InitializeChat"/> runs it automatically — call it yourself only when
        /// you want the warmup to happen at a different moment (e.g. earlier, behind your own UI).
        /// </summary>
        public virtual IEnumerator Warmup() { yield break; }

        // One parse per tokenizer file per session. Tokenizers are immutable vocab data, but parsing
        // one (10+ MB of JSON into ~250k dictionary entries) both costs ~1 s and feeds the GC enough
        // garbage to trigger a full stop-the-world collection — re-doing that on every model
        // construction caused visible frame freezes. Keyed by path; shared by all LLM subclasses.
        static readonly Dictionary<string, object> _tokenizerCache = new Dictionary<string, object>();

        /// <summary>
        /// Returns the cached tokenizer for <paramref name="path"/>, creating (and caching) it via
        /// <paramref name="create"/> on first use. Returns null when the file doesn't exist (models
        /// treat the tokenizer as optional during bring-up). Main-thread only.
        /// </summary>
        protected static T GetOrCreateTokenizer<T>(string path, Func<string, T> create) where T : class
        {
            if (string.IsNullOrEmpty(path) || !System.IO.File.Exists(path))
                return null;
            if (_tokenizerCache.TryGetValue(path, out object cached))
                return (T)cached;
            T tok = create(path);
            _tokenizerCache[path] = tok;
            return tok;
        }

        /// <summary>Single forward pass over <paramref name="input_ids"/>; returns the logits tensor.</summary>
        public abstract Tensor Predict(Tensor input_ids, Tensor attn_mask = null);

        // NOTE for new models: overrides must repeat these NEUTRAL defaults exactly. C# resolves
        // default arguments from the STATIC type at the call site — an override with different
        // defaults silently samples differently through an LLM-typed reference than through the
        // concrete type. Defaults mean "no effect" (temperature 1, no top-k/top-p/min-p
        // truncation, no penalties); a model's recommended preset lives in Config.Default* for
        // callers to pass explicitly.

        /// <summary>
        /// Free-form generation from raw token ids, streaming decoded text via <paramref name="onTokenGenerated"/>.
        /// Defaults are neutral (plain temperature-1 sampling) — pass the model's recommended preset from
        /// <see cref="Config"/>.Default* when you want it. <paramref name="presence_penalty"/> /
        /// <paramref name="repetition_penalty"/> are honored by models that support them and ignored by those that don't.
        /// </summary>
        public abstract IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f);

        /// <summary>
        /// Primes the chat with an (optionally cached) system prompt. Call once before <see cref="Chat"/>.
        /// </summary>
        public abstract IEnumerator InitializeChat(string system_prompt = "");

        /// <summary>
        /// One chat turn: encodes <paramref name="prompt"/> with the model's chat template and streams the reply.
        /// Defaults are neutral (plain temperature-1 sampling) — pass the model's recommended preset from
        /// <see cref="Config"/>.Default* when you want it. <paramref name="enable_thinking"/> and the penalty
        /// knobs are honored only by models that support them.
        /// </summary>
        public abstract IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 0, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false);

        /// <summary>Releases all GPU buffers held by the model. Safe to call more than once.</summary>
        public abstract void Release();

        /// <summary>
        /// Call at the end of a concrete <see cref="Release"/>: unhooks the editor play-mode event
        /// (a static event keeps the dead model alive — handlers accumulate per construction) and
        /// suppresses the finalizer (which would call Release again, off the main thread, where
        /// Unity GPU APIs are illegal).
        /// </summary>
        protected void OnReleased()
        {
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged -= OnPlayModeChanged;
#endif
            CurrentPhase = "idle";   // an interrupted coroutine must not leave a stale phase tag
            GC.SuppressFinalize(this);
        }

        private static readonly HashSet<string> _resourcesWarned = new HashSet<string>();

        /// <summary>
        /// Warns (once per unique path per session, from a model's constructor) when an asset path
        /// (<paramref name="what"/> = "weights" / "tokenizer") doesn't live under a <c>Resources</c> folder.
        /// Loose files under <c>Assets/…</c> are not packed into a player build and their path doesn't exist at
        /// runtime, so they'd be missing on build. By default they are kept out of Resources to save build size —
        /// move them there only when you actually want to ship the model inside the final build.
        /// </summary>
        protected static void WarnIfNotInResources(string what, string path)
        {
            if (string.IsNullOrEmpty(path))
                return;

            foreach (string seg in path.Replace('\\', '/').Split('/'))
                if (string.Equals(seg, "Resources", StringComparison.OrdinalIgnoreCase))
                    return; // already under a Resources folder — nothing to warn about

            if (!_resourcesWarned.Add($"{what}|{path}"))
                return; // warn only once per unique path per session

            ConsoleMessage.Warning($"On build, the {what} must be placed and referenced from resources folder (current: \"{path}\").");
        }
    }
}
