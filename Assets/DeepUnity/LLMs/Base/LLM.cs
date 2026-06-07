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

        /// <summary>
        /// Compiles every compute kernel (the one-time first-dispatch cost) behind a loading screen so the
        /// first real reply isn't a freeze. Default is a no-op; models with a warmup path override it.
        /// Idempotent — call once after construction, before <see cref="InitializeChat"/>.
        /// </summary>
        public virtual IEnumerator Warmup() { yield break; }

        /// <summary>Single forward pass over <paramref name="input_ids"/>; returns the logits tensor.</summary>
        public abstract Tensor Predict(Tensor input_ids, Tensor attn_mask = null);

        /// <summary>
        /// Free-form generation from raw token ids, streaming decoded text via <paramref name="onTokenGenerated"/>.
        /// <paramref name="presence_penalty"/> / <paramref name="repetition_penalty"/> are honored by models that
        /// support them and ignored by those that don't.
        /// </summary>
        public abstract IEnumerator Generate(Tensor input_ids, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 20, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f);

        /// <summary>
        /// Primes the chat with an (optionally cached) system prompt. Call once before <see cref="Chat"/>.
        /// </summary>
        public abstract IEnumerator InitializeChat(string system_prompt = "");

        /// <summary>
        /// One chat turn: encodes <paramref name="prompt"/> with the model's chat template and streams the reply.
        /// <paramref name="enable_thinking"/> and the penalty knobs are honored only by models that support them.
        /// </summary>
        public abstract IEnumerator Chat(string prompt, Action<string> onTokenGenerated,
            int max_new_tokens = 128, float temperature = 1f, int top_k = 20, float top_p = 1f, float min_p = 0f,
            float presence_penalty = 0f, float repetition_penalty = 1f, bool enable_thinking = false);

        /// <summary>Releases all GPU buffers held by the model. Safe to call more than once.</summary>
        public abstract void Release();

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
