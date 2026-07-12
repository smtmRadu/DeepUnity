using System;
using System.Collections;

namespace DeepUnity
{
    /// <summary>How a model leaves GPU residency when defetched (arg per call, or the
    /// init-time default policy for abandoned prefetches).</summary>
    public enum DefetchMode
    {
        /// <summary>Release every GPU buffer immediately (one-frame cost spike acceptable).</summary>
        Full,
        /// <summary>Release buffers progressively across frames under the same per-frame
        /// budget discipline as loading — the anti-frame-drop mirror of SlowPrefetch.</summary>
        Slow,
    }

    public enum ModelResidency { Unloaded, Prefetching, Ready, Defetching }

    // Root of the model hierarchy: ModelBase -> LLM / TTS / STT -> concrete models
    // (Qwen3_5, Gemma3, MiniCPM5 / Chatterbox, CosyVoice, Kokoro / QwenASR, Parakeet).
    //
    // Owns the GPU RESIDENCY contract every family implements — latent (along-the-frames)
    // loading and unloading that never drops the framerate:
    //
    //   PREFETCH  — weights stream to the GPU on a per-frame byte budget. The budget is a LIVE
    //               per-instance property sampled by the loader's pump every frame, so speed
    //               changes mid-flight are free:
    //                  voice.SlowPrefetch(30f);   // player spotted the NPC far away: trickle
    //                  voice.Boost();             // player is approaching: finish at full speed
    //                  voice.PausePrefetch();     // budget 0 — parked, resumable
    //   DEFETCH   — the mirror operation. Safe at ANY point, including mid-prefetch (the player
    //               walked away): in-flight IO is drained, queued uploads are discarded, and GPU
    //               buffers are released either instantly (DefetchMode.Full) or budgeted across
    //               frames (DefetchMode.Slow). After a defetch the model returns to Unloaded and
    //               can be prefetched again later.
    //   POLICY    — DefaultDefetchMode (init-time) governs Defetch() calls without an argument,
    //               so game code can standardize behavior per model instance.
    //
    // The demos don't use these yet — they are primitives for game-side residency logic
    // (proximity prefetch, LOD-style model streaming).
    //
    // CONCURRENCY & SAFETY CONTRACT (what implementations must guarantee):
    //   - All public API calls are MAIN-THREAD (standard Unity); none of them block: they only
    //     flip state/budgets — the actual work happens in coroutine pumps + background IO tasks.
    //     No call can deadlock another: Prefetch/SlowPrefetch/BoostFetch/PausePrefetch/Defetch
    //     may be called in any order, in any state, every frame if desired.
    //   - Budget changes (BoostFetch/PausePrefetch/SlowPrefetch) take effect next frame; budget
    //     0 is a HARD pause (zero bytes uploaded).
    //   - Defetch is safe mid-prefetch: in-flight disk reads are invalidated via a load-epoch,
    //     their results are discarded on arrival (returned to the byte pool), queued uploads are
    //     drained, and only then are GPU buffers released (instant or budgeted).
    //   - Prefetch requested WHILE a defetch is draining is remembered and starts automatically
    //     the moment the defetch completes (the "player came right back" case) — never lost.
    //   - Warmup()/inference must simply wait on IsReady; they never race the loader (buffers
    //     are published atomically per-tensor by the main-thread pump).
    public abstract class ModelBase : IDisposable
    {
        /// <summary>Weights + tokenizer streamed, model ready for inference.</summary>
        public abstract bool IsReady { get; }

        /// <summary>Where this model currently sits in the residency lifecycle.</summary>
        public abstract ModelResidency Residency { get; }

        /// <summary>Bytes of SetData work the loader may perform per frame. LIVE-adjustable at
        /// any point during a prefetch; sampled each frame by the upload pump. Defaults to the
        /// engine-wide full-speed budget (LLM.UploadBudgetBytes).</summary>
        public virtual long LoadBudgetBytesPerFrame { get; set; } = LLM.UploadBudgetBytes;

        /// <summary>Policy applied by <see cref="Defetch()"/> when no mode is passed — set it at
        /// initialization to standardize what happens to abandoned prefetches.</summary>
        public DefetchMode DefaultDefetchMode { get; set; } = DefetchMode.Full;

        /// <summary>Total exported weight bytes (0 until the manifest is parsed).</summary>
        public abstract long TotalWeightBytes { get; }
        /// <summary>Bytes currently resident on the GPU.</summary>
        public abstract long UploadedWeightBytes { get; }
        /// <summary>0..1 residency progress (1 once IsReady).</summary>
        public float LoadProgress => TotalWeightBytes <= 0 ? (IsReady ? 1f : 0f)
                                   : Math.Min(1f, (float)((double)UploadedWeightBytes / TotalWeightBytes));

        // ---- prefetch ------------------------------------------------------------------------

        /// <summary>Begin (or resume) streaming weights to the GPU at full speed.</summary>
        public void Prefetch() => StartPrefetch(LLM.UploadBudgetBytes);

        /// <summary>Identity used by the standardized [GPU] residency log lines — the loaders
        /// log with their weights-folder stem; models on non-manifest paths get the type name.</summary>
        public virtual string ResidencyLabel => GetType().Name;

        /// <summary>Latent prefetch: budget chosen so the REMAINING weights finish in roughly
        /// <paramref name="targetSeconds"/> (assuming ~60 fps), imperceptible per frame.
        /// Call <see cref="BoostFetch"/> anytime to finish fast.</summary>
        public void SlowPrefetch(float targetSeconds, float assumedFps = 60f)
        {
            long remaining = Math.Max(0, TotalWeightBytes - UploadedWeightBytes);
            long frames = (long)Math.Max(1f, targetSeconds * assumedFps);
            long budget = Math.Max(64 * 1024, remaining / frames);
            bool midLoad = Residency == ModelResidency.Prefetching;   // fresh loads log in BeginLoad
            StartPrefetch(budget);
            if (midLoad) ResidencyLog.Budget(ResidencyLabel, budget, remaining);
        }

        /// <summary>Finish an in-flight prefetch at full speed (e.g. the player got close).</summary>
        public void BoostFetch()
        {
            LoadBudgetBytesPerFrame = LLM.UploadBudgetBytes;
            if (Residency == ModelResidency.Prefetching)
                ResidencyLog.Budget(ResidencyLabel, LoadBudgetBytesPerFrame,
                                    Math.Max(0, TotalWeightBytes - UploadedWeightBytes));
        }

        /// <summary>Park an in-flight prefetch (hard pause — the pump idles at budget 0 without
        /// uploading a single byte). Resumable via BoostFetch/SlowPrefetch.</summary>
        public void PausePrefetch()
        {
            LoadBudgetBytesPerFrame = 0;
            if (Residency == ModelResidency.Prefetching)
                ResidencyLog.Budget(ResidencyLabel, 0,
                                    Math.Max(0, TotalWeightBytes - UploadedWeightBytes));
        }

        /// <summary>Family/loader hook: begin or resume the budgeted background load.
        /// Must be safe to call in any state (no-op when Ready or Defetching).</summary>
        protected abstract void StartPrefetch(long bytesPerFrame);

        // ---- defetch -------------------------------------------------------------------------

        /// <summary>Unload with the init-time default policy.</summary>
        public void Defetch() => Defetch(DefaultDefetchMode);

        /// <summary>Safe unload at ANY point — including mid-prefetch (cancels cleanly: drains
        /// in-flight IO, discards queued uploads, releases resident buffers per the mode).
        /// Returns the model to Unloaded; a later prefetch starts fresh.</summary>
        public abstract void Defetch(DefetchMode mode);

        // ---- shared lifecycle ----------------------------------------------------------------

        /// <summary>Waits until ready and pre-runs the kernels so the first real call is
        /// hitch-free. Idempotent.</summary>
        public abstract IEnumerator Warmup();

        /// <summary>Final teardown (play-mode exit): frees everything, instance not reusable.</summary>
        public abstract void Release();

        public void Dispose() => Release();
    }
}
