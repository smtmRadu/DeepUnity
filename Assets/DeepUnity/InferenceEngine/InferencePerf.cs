using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// THE central GPU-performance tuning board for the InferenceEngine. Every cross-engine
    /// scheduling knob lives here as a MUTABLE STATIC with its low-end/high-end direction
    /// documented — set them at boot (before models construct) to retarget a device class, or
    /// leave the defaults: they are the UNIVERSAL STANDARD (validated on an RTX 4060, safe
    /// everywhere) and the runtime self-tunes the rest per device — the underrun tuner escalates
    /// the voice's prebuffer/chunk size as needed and PERSISTS what it learns via PlayerPrefs
    /// (see PocketTTSVoice), and the tick-budget calibration adapts slice sizes every session.
    /// The intent: NOBODY hand-tweaks these per GPU; they hold on any hardware and only a
    /// deliberate design change should edit them.
    ///
    /// The mental model behind every knob: an NPC turn runs TWO GPU pipelines at once —
    /// the LLM decoding tokens (latency-critical: each token gates the next through a readback)
    /// and the TTS synthesizing audio (throughput-critical: the ring must stay ahead of
    /// playback). A strong GPU fits both in every frame; a weak one must ARBITRATE. "Low-end"
    /// below means GPUs where streaming TTS is near/under real-time (e.g. GTX 1650);
    /// "high-end" means TTS has multiples of real-time to spare (e.g. RTX 4060+).
    ///
    /// Related knobs that intentionally live elsewhere:
    ///   - LLM.UploadBudgetBytes           — weight-streaming bytes/frame (load-time, not talk-time).
    ///   - PocketTTS.GpuMacsPerTick        — SELF-CALIBRATING slice size of one heavy TTS tick
    ///                                       (PocketTTSVoice.CalibrateTickBudget drives it between
    ///                                       TtsTickCostMinMs/MaxMs below; don't set it manually).
    ///   - PocketTTSVoice inspector fields — per-NPC: prebufferSeconds / streamChunkFrames /
    ///                                       gpuBudgetMs (runtime-ESCALATED up to the caps below).
    /// </summary>
    public static class InferencePerf
    {
        // ================= #29 cross-engine arbiter (FramePacing) ==============================

        /// <summary>Ring seconds above which a speaking TTS cedes EVERY frame the LLM is
        /// decoding in (it has audio banked; the LLM needs the frame more). LOWER it on low-end
        /// GPUs to let TTS keep synthesizing longer; RAISE on high-end for max tok/s while
        /// speech runs — the audio side won't notice, it has headroom.</summary>
        public static float TtsCedeHeadroomSeconds = 1.0f;

        /// <summary>Ring seconds below which the TTS never cedes a frame (and, inversely, marks
        /// itself STARVING so the LLM holds its bursts — the reverse arbiter). Between this and
        /// TtsCedeHeadroomSeconds the TTS cedes alternate frames. RAISE on low-end GPUs to defend
        /// the audio earlier (costs tok/s); LOWER on high-end (the ring refills instantly anyway).</summary>
        public static float TtsRefillFloorSeconds = 0.5f;

        /// <summary>Hard cap (frames) on how long an LLM decode loop holds ONE token's burst for
        /// a starving TTS — the liveness guarantee (180 ≈ 3 s @60fps). LOWER it to favor reply
        /// latency over audio continuity; RAISE only if a low-end GPU still dribbles words with
        /// everything else maxed. Never remove: an unbounded hold serialized whole replies.</summary>
        public static int LlmHoldMaxFrames = 180;

        // ============ LLM play-mode pacing — the decode-SPEED ⇄ FRAMERATE tradeoff dial ============
        // The LLM decode is autoregressive: each token must finish (and read back) before the next
        // starts, so it can't be parallelized — the only lever in-game is HOW MANY frames you spend
        // per unit of LLM work. Spend fewer frames → faster text, lower FPS during the reply. Spend
        // more → smoother FPS, slower text. These two dials expose that tradeoff so it's tuned once
        // per GPU instead of hard-coded. Both are OVERRIDDEN to the smooth end automatically while a
        // TTS voice is starving (audio continuity wins there — see FramePacing / the #29 arbiter).

        /// <summary>Transformer layers issued per frame while PREFILLING the prompt (the one-time
        /// cost before the first token). Prefill is heavy, so it's spread over frames to avoid a
        /// single giant hitch. RAISE for faster prompt processing at the cost of chunkier frames
        /// as the reply spins up (fine on a strong GPU, or when a brief hitch is acceptable); LOWER
        /// to 1 for the smoothest frames on a weak GPU. ≤ 0 issues the whole prefill in ONE frame
        /// (fastest possible, biggest hitch). 24-layer model, so 24+ = single-frame prefill.</summary>
        public static int LlmPrefillLayersPerFrame = 1;

        /// <summary>Tokens decoded per frame during a reply. Each token is issued + sampled
        /// SYNCHRONOUSLY (only a few ms of GPU on the coalesced kernels), then the frame is handed
        /// back to rendering every this-many tokens. RAISE for faster text at a lower framerate
        /// while the NPC talks — the "spend frames on decode instead of idle FPS" trade, ideal on a
        /// weak GPU that's otherwise coasting mid-reply; LOWER to 1 for the smoothest framerate
        /// during generation. Higher values reveal text in bursts of N and hitch ~N×(token cost)
        /// per frame, so don't set it so high the app feels frozen. Forced to 1 while a TTS voice
        /// is starving.</summary>
        public static int LlmDecodeTokensPerFrame = 1;

        // ================= pocket-tts streaming voice ==========================================

        /// <summary>Ceiling for the ADAPTIVE prebuffer (PocketTTSVoice doubles its inspector
        /// prebufferSeconds on repeated ring underruns up to this). Effectively "longest silence
        /// we'd trade for unbroken phrases". RAISE on low-end GPUs for fewer, longer pauses;
        /// irrelevant on high-end (never escalates past the first rungs).</summary>
        public static float TtsPrebufferCapSeconds = 3f;

        /// <summary>Ceiling for the ADAPTIVE decode chunk size, in 80 ms latent frames (16 =
        /// 1.28 s per chunk; escalates from the inspector's streamChunkFrames once the prebuffer
        /// is maxed and underruns persist). Bigger chunks amortize the per-chunk windowed
        /// re-decode (the main streaming tax on weak GPUs) at the cost of coarser text-reveal
        /// cadence and later first audio. RAISE on very-low-end; 8 is plenty on high-end.</summary>
        public static int TtsMaxChunkFrames = 16;

        /// <summary>Heavy GPU ticks per frame while the voice is SILENT and refilling (prebuffer
        /// or underrun re-gate): nothing is audible, so frame smoothness buys nothing — push hard
        /// to end the gap sooner. RAISE on low-end (shorter gaps, choppier framerate during
        /// silence); 1-2 suffices on high-end. Audible playback always uses 1 (2 when the ring
        /// is low).</summary>
        public static int TtsSilentRefillHeavyTicks = 4;

        /// <summary>Multiplier on the voice's per-frame CPU issue budget (gpuBudgetMs) during
        /// silent refill. Same trade as TtsSilentRefillHeavyTicks, applied to the pump loop.</summary>
        public static float TtsSilentRefillBudgetScale = 2f;

        /// <summary>Milliseconds the pump spin-waits on an in-flight GPU readback while the ring
        /// is comfortable (shallow queues often complete mid-frame). RAISE to trade CPU for a bit
        /// of TTS throughput; LOWER for CPU thrift. When the ring is low the spin window is the
        /// whole frame budget regardless.</summary>
        public static double TtsGpuWaitSpinMs = 2.0;

        // ---- PocketTTS.GpuMacsPerTick self-calibration targets --------------------------------
        // One heavy TTS tick should cost about this much frame time over the scene's baseline.
        // The calibrator grows/shrinks GpuMacsPerTick to stay inside the band — on ANY GPU.

        /// <summary>Below this measured tick cost the slice GROWS (more TTS throughput per tick).
        /// RAISE both min/max on low-end GPUs to accept jankier talk-time frames for throughput;
        /// LOWER both for silkier frames on high-end.</summary>
        public static float TtsTickCostMinMs = 3f;

        /// <summary>Above this measured tick cost the slice SHRINKS (smoother frames) — except
        /// while the ring is low, where throughput wins and the slice never shrinks.</summary>
        public static float TtsTickCostMaxMs = 7f;

        /// <summary>Hard floor of the self-calibrated tick slice (MACs). Guards against external
        /// GPU load shrinking slices into dispatch-overhead territory. Rarely needs touching.</summary>
        public static long TtsMacsTickFloor = 200_000_000;

        /// <summary>Hard cap of the self-calibrated tick slice (MACs). RAISE on very strong GPUs
        /// if profiling shows the calibrator pinned here while frames stay cheap.</summary>
        public static long TtsMacsTickCap = 4_000_000_000;

        // ================= measured AutoTune (#32) =============================================
        // Sync-vs-async decode is a DEVICE property (token cost vs frame budget), not a universal
        // constant: always-sync (v0.14.7) made text 2× faster on a 4060 but put its ~14 ms token
        // stall into every reply frame — undoing #20's smoothness on exactly the GPUs that didn't
        // need the throughput. So the runtime MEASURES and decides per session: the first
        // AutoTuneMeasureTokens sync tokens of a session are the probe (timed inside
        // Qwen3_5.DecodeStep), then:
        //   tokenMs <= SyncStallBudgetMs            -> SYNC (the stall hides inside the frame
        //                                              budget) and pack floor(budget/tokenMs)
        //                                              tokens per frame;
        //   else if est. async tok/s >= MinUsable   -> ASYNC (#20 path: smooth frames, text at
        //                                              ~fps/AsyncFramesPerToken — reading speed);
        //   else                                    -> SYNC anyway (a weak GPU at low fps would
        //                                              dribble unusable text on the async path).
        // SESSION-LOCAL by design — re-measured at every boot, so nothing decided on one GPU (or
        // during one contended session) can ever degrade a run on another device.
        // SmoothVsSpeed is the ONE user-facing dial (see InferencePerfTuner for the scene slider).

        /// <summary>The Smooth ⇄ Speed preference. The AUTO-DETECTION always computes for a
        /// stable 60+ fps (AutoTargetFps — the anchor never moves); this value only biases the
        /// internal tradeoff around that result. 0.5 = pure auto (no bias). Offsets multiply the
        /// measured budgets (BiasMultiplier); the EXTREMES force the implementation limits:
        /// ≤0.02 = gentlest possible (async decode, 1 layer/frame prefill), ≥0.98 = fastest
        /// possible (sync decode, bulk prefill). Changing it mid-session should go through
        /// ResetAutoTune() (InferencePerfTuner does) so the next reply re-decides.</summary>
        public static float SmoothVsSpeed = 0.5f;

        /// <summary>The FIXED frame-rate anchor every auto computation targets: hold a stable
        /// 60 fps while the NPC replies, on any GPU. The slider does not move this — it biases
        /// around it.</summary>
        public const float AutoTargetFps = 60f;

        /// <summary>Bias the slider applies to the measured budgets: ×1 at center (pure auto),
        /// ×0.25 near Smooth (accept only a quarter of the 60 fps budget → lighter frames),
        /// ×4 near Speed (accept 4× the budget → faster text at lower fps, e.g. ~30 fps
        /// territory). Exact extremes bypass the math entirely (implementation limits).</summary>
        public static float BiasMultiplier => Mathf.Pow(4f, (Mathf.Clamp01(SmoothVsSpeed) - 0.5f) * 2f);

        /// <summary>Async-path floor: below this estimated tok/s a reply reads as a dribble, so a
        /// weak GPU stays sync even though its token stall breaks the frame budget.</summary>
        public static float MinUsableTokS = 12f;

        /// <summary>Empirical frames one token spans on the async path (ForwardYielding's frame +
        /// AsyncGPUReadback ~2 + the loop's trailing yield — v0.14.7's "framerate/4" symptom).</summary>
        public static float AsyncFramesPerToken = 3.5f;

        /// <summary>Sync probe tokens measured at each session start before the decision locks.</summary>
        public static int AutoTuneMeasureTokens = 6;

        static readonly System.Collections.Generic.List<float> tokenMsSamples =
            new System.Collections.Generic.List<float>();
        static float measureStartTime;
        static int measureStartFrame;
        static float minFrameDtMs;      // cheapest frame seen in the probe window ≈ scene baseline
        static bool decided;
        static bool syncDecode = true;

        /// <summary>True while decode should take the FAST synchronous path: the slider's hard
        /// extremes force the implementation limit (Speed end = always sync, Smooth end = always
        /// async); otherwise true during the probe window (the measurement needs sync tokens),
        /// then the measured policy. TTS starvation still overrides to async at the call site.</summary>
        public static bool UseSyncDecode => SmoothVsSpeed >= 0.98f ? true
                                          : SmoothVsSpeed <= 0.02f ? false
                                          : (!decided || syncDecode);

        /// <summary>Qwen3_5.DecodeStep reports each SYNC token's wall cost here; the
        /// AutoTuneMeasureTokens-th sample locks the session's decision (and the
        /// LlmDecodeTokensPerFrame packing) with one log line.</summary>
        public static void NoteSyncTokenMs(float ms)
        {
            if (decided) return;
            if (tokenMsSamples.Count == 0)
            {
                measureStartTime = Time.realtimeSinceStartup;
                measureStartFrame = Time.frameCount;
                minFrameDtMs = float.MaxValue;
            }
            minFrameDtMs = Mathf.Min(minFrameDtMs, Time.unscaledDeltaTime * 1000f);
            tokenMsSamples.Add(ms);
            if (tokenMsSamples.Count < AutoTuneMeasureTokens) return;

            tokenMsSamples.Sort();
            float tokenMs = tokenMsSamples[tokenMsSamples.Count / 2];   // median: JIT/contention robust
            float elapsed = Mathf.Max(1e-3f, Time.realtimeSinceStartup - measureStartTime);
            float fps = Mathf.Max(1f, (Time.frameCount - measureStartFrame) / elapsed);
            float asyncTokS = fps / Mathf.Max(1f, AsyncFramesPerToken);

            // The tradeoff always anchors on the FIXED 60 fps target: how much decode stall
            // fits in a 60 fps frame after the scene's own baseline cost? The slider only
            // multiplies that measured budget (×1 at center = pure auto).
            float targetFrameMs = 1000f / AutoTargetFps;
            float baselineMs = Mathf.Min(minFrameDtMs, targetFrameMs);   // vsync-quantized floors cap out
            float stallBudgetMs = Mathf.Max(0f, targetFrameMs - baselineMs) * BiasMultiplier;

            if (tokenMs <= stallBudgetMs)
            {
                syncDecode = true;   // full-speed text still holds the target frame rate
                LlmDecodeTokensPerFrame = Mathf.Max(1, (int)(stallBudgetMs / Mathf.Max(0.1f, tokenMs)));
            }
            else if (asyncTokS >= MinUsableTokS) { syncDecode = false; LlmDecodeTokensPerFrame = 1; }
            else { syncDecode = true; LlmDecodeTokensPerFrame = 1; }   // weak GPU: readable text wins
            decided = true;
            AutoTuneStatus = (syncDecode ? $"SYNC decode, {LlmDecodeTokensPerFrame} tok/frame"
                                         : $"ASYNC decode (~{asyncTokS:F0} tok/s est.)") +
                             $" — token {tokenMs:F1} ms, 60 fps anchor, bias ×{BiasMultiplier:F2}";
            Debug.Log($"[InferencePerf] AutoTune: token {tokenMs:F1} ms (median/{tokenMsSamples.Count}), baseline {baselineMs:F1} ms, {fps:F0} fps → " +
                      (syncDecode ? $"SYNC decode, {LlmDecodeTokensPerFrame} tok/frame"
                                  : $"ASYNC decode (~{asyncTokS:F0} tok/s est.)") +
                      $" | 60 fps anchor, stall budget {stallBudgetMs:F1} ms (Smooth⇄Speed {SmoothVsSpeed:F2}, bias ×{BiasMultiplier:F2}) | {SystemInfo.graphicsDeviceName}");
        }

        /// <summary>One-line description of the session's AutoTune decision (the tuner's
        /// inspector shows it live); "measuring…" until the probe tokens lock it.</summary>
        public static string AutoTuneStatus { get; private set; } = "measuring…";

        // ---- adaptive prefill packing (#32) ---------------------------------------------------
        // The dialogue-open latency IS the system-prompt prefill (plus the question prefill at
        // each reply) — pacing it with a fixed layers-per-frame constant wastes exactly the GPUs
        // that could swallow it whole. The pack self-tunes off MEASURED prefill frame times
        // against the same bias-scaled budget: fast GPUs converge to fat packs (dialogs open in a
        // fraction of the time), weak ones sink to the per-NPC slider floor.

        /// <summary>Layers-per-frame pack the chunked prefill runs at (adaptive). Converges
        /// within one system-prompt prefill and persists for the session.</summary>
        public static int PrefillPack = 6;

        /// <summary>ForwardPromptChunked reports each prefill frame's wall time here; the pack
        /// grows while whole frames sit comfortably under the 60 fps frame budget (× the slider
        /// bias) and shrinks fast when they overshoot it. Prefill is a loading moment, so it may
        /// FILL the frame (unlike decode, which must leave room for the scene).</summary>
        public static void NotePrefillFrameMs(float ms)
        {
            float targetFrameMs = 1000f / AutoTargetFps * BiasMultiplier;
            if (ms < targetFrameMs * 0.75f) PrefillPack = Mathf.Min(64, PrefillPack + 2);
            else if (ms > targetFrameMs * 1.1f) PrefillPack = Mathf.Max(1, PrefillPack - 4);
        }

        /// <summary>The layers-per-frame pack a prefill should actually run at: the slider's
        /// hard extremes force the implementation limits (Smooth end = 1 layer/frame, the
        /// gentlest possible; Speed end = 64 ≈ whole chunks in one frame), otherwise the
        /// measured adaptive pack. Models call this per yield.</summary>
        public static int EffectivePrefillPack()
        {
            if (SmoothVsSpeed <= 0.02f) return 1;
            if (SmoothVsSpeed >= 0.98f) return 64;
            return Mathf.Max(1, PrefillPack);
        }

        /// <summary>Drops the session decision — the next reply's first tokens re-probe. Called at
        /// boot and whenever ReplySpeedBias changes.</summary>
        public static void ResetAutoTune()
        {
            decided = false;
            syncDecode = true;
            tokenMsSamples.Clear();
            LlmDecodeTokensPerFrame = 1;
            PrefillPack = 6;
            AutoTuneStatus = "measuring…";
        }

        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetOnBoot() => ResetAutoTune();   // also covers editor domain-reload-off replays
    }
}
