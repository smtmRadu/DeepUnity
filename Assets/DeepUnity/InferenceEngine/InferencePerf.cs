namespace DeepUnity
{
    /// <summary>
    /// THE cross-engine GPU-scheduling board for the InferenceEngine: what is left of the streaming
    /// voice's budgets, plus the LLM⇄TTS arbiter that decides which of the two pipelines a contended
    /// frame belongs to. Every knob lives here as a MUTABLE STATIC with its low-end/high-end
    /// direction documented — set them at boot (before models construct) to retarget a device class,
    /// or leave the defaults: they are the UNIVERSAL STANDARD (validated on an RTX 4060, safe
    /// everywhere). The intent: NOBODY hand-tweaks these per GPU; they hold on any hardware and only
    /// a deliberate design change should edit them.
    ///
    /// NOT on this board any more (2026-07-26): the LLM's OWN per-frame pacing — weight-fetch
    /// bytes, prefill steps, decode tokens. Those are five fixed rows in
    /// <see cref="BackendTradeoffTable"/>, one per Backend Tradeoff level, and the two self-tuning
    /// controllers that used to derive them here (plus the continuous SmoothVsSpeed slider that
    /// biased both) are deleted — BackendTradeoff.cs documents at length why measuring turned out to be
    /// the wrong instrument for that dial. The two axes only ever LOOKED like one knob: this board
    /// arbitrates between two pipelines that both want the same frame, and must therefore react
    /// every frame; the tradeoff table states how much work one pipeline may do in a frame it has
    /// already won, which is a choice about the machine, made once.
    ///
    /// ...and gone the same way on 2026-07-27: the TTS pacing that was self-tuning here. Heavy ticks
    /// per frame (speaking and refilling), prebuffer seconds, decode chunk frames, cede headroom and
    /// the MAC size of one heavy tick are now table rows too, and the three loops that used to learn
    /// them at runtime (a PlayerPrefs-persisted prebuffer/chunk escalation ladder, the tick-cost
    /// calibrator, a refill-rate EMA) are deleted with their knobs — TtsCedeHeadroomSeconds,
    /// TtsPrebufferCapSeconds, TtsMaxChunkFrames, TtsSilentRefillHeavyTicks, TtsTickCostMinMs/MaxMs
    /// and TtsMacsTickFloor/Cap. What still lives here is what a fixed row cannot state: RATES and
    /// SHAPES of the arbitration itself. Nothing here reads the table; the voice reads both.
    ///
    /// The mental model behind every knob: an NPC turn runs TWO GPU pipelines at once —
    /// the LLM decoding tokens (latency-critical: each token gates the next through a readback)
    /// and the TTS synthesizing audio (throughput-critical: the ring must stay ahead of
    /// playback). A strong GPU fits both in every frame; a weak one must ARBITRATE. "Low-end"
    /// below means GPUs where streaming TTS is near/under real-time (e.g. GTX 1650);
    /// "high-end" means TTS has multiples of real-time to spare (e.g. RTX 4060+).
    ///
    /// Related knobs that intentionally live elsewhere:
    ///   - BackendTradeoffTable            — ALL of the per-frame budgets that are a statement about
    ///                                       the MACHINE, as five fixed rows: fetch bytes, prefill
    ///                                       steps, decode tokens, and the TTS tick/prebuffer/chunk/
    ///                                       cede-headroom/tick-MAC columns.
    ///   - LLM.UploadBudgetBytes           — the weight-streaming bytes/frame in force RIGHT NOW
    ///                                       (load-time, not talk-time): the table's fetch row,
    ///                                       lowered while a walk-up prefetch is hiding under the
    ///                                       player's approach, restored from the table after.
    ///   - PocketTTS.GpuMacsPerTick        — slice size of one heavy TTS tick; reads the table's
    ///                                       row, fixed per tier (no longer self-calibrating).
    ///   - PocketTTSVoice inspector fields — per-NPC: gpuBudgetMs is authored; prebufferSeconds and
    ///                                       streamChunkFrames are PUSHED from the tier by
    ///                                       NPCChatBase and no longer escalate at runtime.
    /// </summary>
    public static class InferencePerf
    {
        // ================= #29 cross-engine arbiter (FramePacing) ==============================

        // The cede LEVEL (ring seconds above which a speaking voice cedes every LLM decode frame)
        // moved to BackendTradeoffTable.TtsCedeHeadroomSeconds on 2026-07-27 — it is a per-tier
        // number, 2.0 s on the low rows down to 1.0 s at Very Fast, and the table carries the
        // measurement note that produced it. The RATE bound below stayed: it is a property of the
        // arbiter's control loop, not of the GPU.

        /// <summary>Cede RATE bound: at most ONE frame in this many may be ceded to a decoding LLM,
        /// however much audio the ring has banked. The level knobs above only say WHETHER there is
        /// surplus; this says how fast the arbiter may spend it (see FramePacing — the ring is an
        /// integrator, so an unbounded rate always finds the same fixed point: an EMPTY ring).
        /// Synthesis at S× real-time holds the ring only while the pump keeps 1/S of the frames, so
        /// the sustainable cede share is 1 − 1/S — ≈29% at the 1.3-1.5× measured on the 1650. 3
        /// yields 1/3 above the headroom and ≈1/6 between floor and headroom. Note where that leaves
        /// the ring: BOTH bands still cede faster than synthesis earns, so the level does not park at
        /// the cede headroom (an earlier note here claimed it did) — it slides to just under the tier's
        /// REFILL FLOOR, the one band that never cedes. LOWER toward 1 (= unbounded, the pre-2026-07-26
        /// behaviour) for max tok/s while speech runs; RAISE to defend audio harder on a GPU whose
        /// synthesis is barely real-time.
        /// Keep it ODD: the pump's middle band cedes on odd frames only, so an odd stride alternates
        /// the allowed frames' parity and that band cedes half as often as the full-cede band — the
        /// gradient that makes the headroom an equilibrium. An even stride locks the parity for the
        /// whole decode phase, so the middle band would cede on every allowed frame or on none,
        /// decided at random per reply.</summary>
        public static int TtsCedeFrameStride = 3;

        /// <summary>Seconds to bank before RESUMING playback after a genuine mid-reply starve — one
        /// where no clause was in flight, i.e. synthesis actually lost the race to playback. A clause
        /// prefill does not come here at all (PocketTTSVoice leaves the stream open and zero-fills
        /// through its bounded dead window); that separation is the 2026-07-27 fix.
        /// <para>DELIBERATELY NOT derived from the tier's refill floor, which is what it used to be
        /// (regateSeconds = min(prebuffer, floor × 2)). That coupling was a trap: the floor wants to
        /// RISE to cover a clause dead window, the re-gate wants to STAY SHORT because re-gating long
        /// is what turned a 0.08-0.40 s dry ring into a 1.20-2.00 s freeze (2026-07-26/27). Two
        /// different questions — "when do I stop being polite" versus "how much do I bank before
        /// speaking again" — so two numbers.</para>
        /// <para>Capped by the voice's prebuffer, since banking more than a fresh reply would is
        /// indefensible. RAISE only if a genuine starve produces word…pause…word dribble; every other
        /// symptom belongs to BackendTradeoffTable.</para>
        /// <para>0.25, down from 0.5 (2026-07-28), because the `pause after drain` line let the two
        /// terms be compared directly and the re-gate was still the bigger one: 0.56 s of real dry ring
        /// carried 1.20 s of re-gated silence, and 0.32 s carried 0.40 s. Four dry bursts inside 0.56 s
        /// is ~0.14 s each — nothing a listener would call a pause — so waiting 0.5 s to resume was
        /// paying roughly 3× the outage to hide it. 0.25 s is still ~1.8× a burst, which is the point:
        /// enough to resume on a phrase rather than a syllable, not enough to be the outage itself.</para></summary>
        public static float TtsRegateSeconds = 0.25f;

        /// <summary>Ring seconds below which audible silence is IMMINENT while the player is
        /// listening — the emergency band. Distinct from the tier's refill floor the same way the
        /// re-gate above is distinct from it: the floor defends a reserve ("stop being polite"),
        /// this responds to the reserve being nearly GONE. Two effects share the threshold:
        /// <para>(1) hurry-flush — the streaming synth suspends its chunk cadence and decodes every
        /// <c>PocketTTS.StreamHurryMinFrames</c> latents it has (see <c>PocketTTS.StreamHurry</c>).
        /// Without it a chunk-16 flush delivers 1.28 s of audio as ONE lump, so with the ring at
        /// zero the 0.25 s re-gate could not resume until the whole lump landed — measured
        /// 1.20-2.80 s of "re-gated" silence per reply (2026-07-30 log) against dry spells half
        /// that size. The re-gate number above is only as fine as the lumps that feed it.</para>
        /// <para>(2) the LLM waits (<c>FramePacing.NoteTtsStarving</c>) — mid-reply low ring
        /// deliberately does NOT hold the LLM (on weak GPUs low is the normal state; see the
        /// reverse arbiter in <c>PocketTTSVoice.PumpPipeline</c>), but below this band a hole is no
        /// longer a risk, it is a certainty at playback speed, and preventing it outranks a few
        /// tokens of decode.</para>
        /// <para>Keep WELL below every tier's <c>ttsRefillFloorSeconds</c>: floor-hover is the
        /// steady state on Smooth and must not trip the emergency path — a flush costs ~16 heavy
        /// ticks whatever its size, so constant small flushes would burn the chunk amortization
        /// exactly when throughput is thinnest. RAISE if dropouts still hit zero before help
        /// arrives; LOWER if text generation visibly stalls at every clause trough.</para></summary>
        public static float TtsPanicFloorSeconds = 0.25f;

        /// <summary>Hard cap (frames) on how long an LLM decode loop holds ONE token's burst for
        /// a starving TTS — the liveness guarantee (180 ≈ 3 s @60fps). LOWER it to favor reply
        /// latency over audio continuity; RAISE only if a low-end GPU still dribbles words with
        /// everything else maxed. Never remove: an unbounded hold serialized whole replies.</summary>
        public static int LlmHoldMaxFrames = 180;

        // The LLM's side of that tradeoff (prefill steps and decode tokens per frame) moved to
        // BackendTradeoffTable on 2026-07-26 — see the class docs above. What stays here is only the
        // arbitration: the #29 knobs decide WHOSE frame it is, the table decides how much the LLM
        // packs into the frames it gets, and TTS starvation still overrides the table's decode
        // packing to one-token-per-frame at the call site (audio continuity wins there).

        // ================= pocket-tts streaming voice ==========================================

        // Prebuffer seconds, decode chunk frames and the silent-refill tick count are tier columns
        // in BackendTradeoffTable as of 2026-07-27 (with the two escalation CEILINGS that only the
        // deleted ladder needed — TtsPrebufferCapSeconds / TtsMaxChunkFrames — gone entirely).

        /// <summary>Multiplier on the voice's per-frame CPU issue budget (gpuBudgetMs) during
        /// silent refill. The tick-count side of that same trade is the tier's silent-refill column
        /// in BackendTradeoffTable; this scales the pump loop's issue window to match.</summary>
        public static float TtsSilentRefillBudgetScale = 2f;

        /// <summary>Milliseconds the pump spin-waits on an in-flight GPU readback while the ring
        /// is comfortable (shallow queues often complete mid-frame). RAISE to trade CPU for a bit
        /// of TTS throughput; LOWER for CPU thrift. When the ring is low the spin window is the
        /// whole frame budget regardless.</summary>
        public static double TtsGpuWaitSpinMs = 2.0;

        // The slice size of one heavy TTS tick is BackendTradeoffTable.TtsMacsPerTick, read by
        // PocketTTS.GpuMacsPerTick. Deleted with the calibrator that used to walk it (2026-07-27):
        // TtsTickCostMinMs/MaxMs (the 3-7 ms band it aimed at) and TtsMacsTickFloor/Cap (200M-4G,
        // the range it walked). The tier states the slice instead — see BackendTradeoff.cs for why
        // measuring it turned into a feedback path straight to starvation.
    }
}
