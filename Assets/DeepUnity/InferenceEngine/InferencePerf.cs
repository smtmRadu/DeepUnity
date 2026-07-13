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
    }
}
