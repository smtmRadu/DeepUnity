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

        /// <summary>Transformer layers of ONE decode token issued per frame — the slice width of
        /// sliced decode (2026-08-02, the smoothness mandate: no single frame may carry a whole
        /// token's GPU burst; latency is explicitly no longer a criterion). Read by every family's
        /// <c>ForwardYielding</c> on its seqLen==1 path (Qwen3.5; Gemma3/MiniCPM5 ported to the same
        /// constant); prefill is untouched — its unit stays one layer, counted by the tradeoff
        /// table's prefill column.
        /// <para>Why this lives HERE and not as a BackendTradeoffTable column, when it plainly prices
        /// LLM work per frame: the table's rows answer "how capable is this machine", and this number
        /// answers nothing of the sort. It is the mandate's CEILING on what one frame may be asked to
        /// carry — identical on every machine, like the divisor shapes above it. A fast GPU pays
        /// nothing for slicing (its slices are sub-millisecond and tok/s ≈ fps / frames-per-token, so
        /// it scales with the framerate it already has); a slow GPU is the machine the ceiling exists
        /// FOR.</para>
        /// <para>The arithmetic behind 6, on the reference 1650 at Smooth: a Qwen3.5-0.8B token reads
        /// ~0.55 GB of INT8 layer weights plus ~0.5 GB of fp16 lm_head (~30-55 ms of GPU issued as
        /// one burst — the 33/55 ms GEN rows sliced decode retires). 24 layers ÷ 6 = 4 slices of
        /// ~6-9 ms each, the lm_head alone in a fifth frame (~11-15 ms — one GEMV, indivisible
        /// without a kernel change, and therefore the true frame-cost floor of this scheme). With the
        /// async token readback that is ~6-7 frames per token, i.e. ~5-8 tok/s at talk-time
        /// framerates — down from the sync path's ~12, accepted because speech at ~3 words/s is the
        /// real pacing bottleneck and ~5 tok/s ≈ 3.7 words/s still clears it.</para>
        /// <para>RAISE toward the layer count for fewer, fatter decode frames (numLayers = one-burst
        /// issue with only the head split off); LOWER toward 1 for the pre-#20 per-layer spread,
        /// whose failure mode is on record (task #20: ~30 frames per token, and in uncapped scenes
        /// the CPU out-issues the GPU until Present() stalls). The async readback bounds the backlog
        /// to ONE token in flight, which is why the #20 backup does not return at sane widths.</para></summary>
        public static int LlmDecodeSliceLayers = 6;

        // The LLM's side of that tradeoff (prefill steps and decode tokens per frame) moved to
        // BackendTradeoffTable on 2026-07-26 — see the class docs above. What stays here is only the
        // arbitration: the #29 knobs decide WHOSE frame it is, the table decides how much the LLM
        // packs into the frames it gets, and TTS starvation still overrides the table's decode
        // packing to one-token-per-frame at the call site (audio continuity wins there).
        // LlmDecodeSliceLayers (2026-08-02) is not that tradeoff coming back: the table states what a
        // machine may SPEND per frame, the slice width states what any frame may be ASKED TO CARRY —
        // a shape of the mandate, not a budget of the machine.

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

        /// <summary>Divisor on the tier's speaking tick count while the voice CRUISES — the ring
        /// holds more than the tier's cede headroom, so every dead window the pump's boosts exist
        /// for is already covered by banked audio. Full-rate synthesis there buys nothing but
        /// finishing the reply's audio sooner, and what it measurably cost (GTX 1650, frame probe
        /// 2026-08-02) was 4 ticks ≈ 16-24 ms of GPU dropped into every frame with the ring at
        /// 3-7 s and the LLM idle — the reported 60-70→25-35 fps dips while an NPC speaks. The
        /// #29 arbiter only ever ceded frames to the LLM; cruise is the RENDERER's seat at that
        /// table. This is a shape of the arbitration (how much gentler cruising is than speaking),
        /// so it lives here; the magnitudes it divides stay tier columns. RAISE for smoother
        /// frames and slower ring growth; 1 disables cruising entirely.</summary>
        public static int TtsCruiseTickDivisor = 2;

        /// <summary>Ring seconds ABOVE the tier's cede headroom needed to ENTER cruise; leaving
        /// happens at the headroom itself. The band exists because the ring is an integrator: a
        /// single-threshold test parks at its own boundary and flips tick counts frame by frame —
        /// the exact trap documented on <c>ttsCedeHeadroomSeconds</c>. Half a second ≈ seconds of
        /// steady state per crossing at cruise-rate drain.</summary>
        public static float TtsCruiseEnterExtraSeconds = 0.5f;

        /// <summary>Divisor on the pump's BOOST tick count (the pushHard band: low ring while
        /// audible, uncovered clause prefill, silent refill) on frames the LLM ALSO issued GPU
        /// work. Unsplit, the two engines stack: the tier's boost ticks (≈24-36 ms of GPU at
        /// Smooth's 6) land in the same frame as an LLM token burst (≈15-25 ms) — the 97-162 ms
        /// GEN+SPK frames in every talk-perf report, ~10-15 fps for the seconds a reply is both
        /// generated and spoken. Splitting is nearly free where it looks most dangerous: a clause
        /// prefill is FRAME-bound (~24 FrameBreak ticks), so at half the ticks on half-length
        /// frames its WALL time barely moves, and the panic floor below still holds the LLM
        /// outright when a hole is imminent (that negotiation — TTS sprints on held frames,
        /// LLM resumes above the floor — is the intended steady state on weak GPUs). A shape of
        /// the arbitration, applied to tier magnitudes: general across GPUs by construction.
        /// RAISE for smoother collision frames and slower refills under decode; 1 disables.</summary>
        public static int TtsSharedFrameTickDivisor = 2;

        /// <summary>What one retained-prompt TEXT ROW (AppendRowsKVYielding, the #32 clause
        /// start) is BILLED at when the MAC dial sizes its per-tick batches, whenever its real
        /// MAC count is smaller. A text row is ~76 MMAC but ~40 tiny GEMV dispatches, and on a
        /// latency-bound GPU those dispatches are the cost: measured 2-4 ms of GPU per row on
        /// the GTX 1650 against the ~1 ms its MACs suggest. Billed at face value, Smooth's
        /// 900 MMAC tick bought 11 rows and the pump's 4-tick clause-start allowance stacked 44
        /// rows ≈ 90-170 ms of GPU into ONE frame — the `ar_frame @prefill ring 0.00` family in
        /// every worst-20 (96-169 ms), the last conversation-time spike class standing after the
        /// 2026-08-02 hunts. 400 M ≈ the dispatch-latency-equivalent of a row on the reference
        /// box: Smooth then packs 2 rows/tick (~4-8 ms GPU), scaling up the tiers exactly like
        /// the block prefill's LinearRows batches always did. A shape of the COST MODEL, not a
        /// tier magnitude — the tier dial stays <c>ttsMacsPerHeavyTick</c>. LOWER only if clause
        /// dead windows measurably starve the ring on a strong GPU; RAISE if clause starts still
        /// spike on a weaker-than-1650 card.</summary>
        public static long TtsTextRowDispatchEquivMacs = 400_000_000;

        /// <summary>#36: elements (floats) of a freshly created ComputeBuffer that one frame may
        /// FIRST-TOUCH (zero-fill) during the walk-up preallocation pass. Creating a buffer only
        /// reserves it — the driver's physical commit, and on a full card the WDDM residency
        /// migration, wait for the first dispatch that writes it. Untouched, that first write was
        /// the first real prefill/flush (the once-per-session 160+290 ms walk-up frames); touched
        /// whole-buffer, the commit just moved into the touch (measured 262+283 ms on the ~24 MB
        /// mimi SEANet scratches — migration cost tracks BYTES, not dispatches). 1M elements =
        /// 4 MB/frame ≈ a few ms of commit each, ~30 extra walk-up frames total on the Smooth
        /// window sizes. RAISE on cards with VRAM headroom (fewer prealloc frames); LOWER if the
        /// prealloc frames themselves ever show up in a spike report.</summary>
        public static int TtsFirstTouchElemsPerFrame = 1_000_000;

        // The slice size of one heavy TTS tick is BackendTradeoffTable.TtsMacsPerTick, read by
        // PocketTTS.GpuMacsPerTick. Deleted with the calibrator that used to walk it (2026-07-27):
        // TtsTickCostMinMs/MaxMs (the 3-7 ms band it aimed at) and TtsMacsTickFloor/Cap (200M-4G,
        // the range it walked). The tier states the slice instead — see BackendTradeoff.cs for why
        // measuring it turned into a feedback path straight to starvation.
    }
}
