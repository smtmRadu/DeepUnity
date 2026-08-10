using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// The five discrete settings of the Backend Tradeoff dial: how much of a frame the inference
    /// backends may spend.
    /// <para>The numbers each one implies live in <see cref="BackendTradeoffTable"/> — one row per level,
    /// and that table is the ONLY place any of them appears.</para>
    /// </summary>
    /// <para><b>Called the "LLM Tradeoff" dial until 2026-07-27.</b> It never was LLM-only. The fetch row is written
    /// into <c>LLM.UploadBudgetBytes</c>, which is the ONE weight-streaming budget every backend uploads
    /// against — Parakeet, QwenASR, Kokoro, CosyVoice3, Chatterbox and PocketTTS included — and as of the
    /// same date the table also owns PocketTTS's own per-frame pacing (the tts rows below). "Backend" =
    /// every inference backend the engine runs, which is what this dial has always priced.</para>
    /// <para><b>What each level is calibrated FOR</b> (author's intent, 2026-07-26 — the anchor any
    /// future retuning should preserve). The rows are ours to move; what must not drift is which class
    /// of machine each one targets:</para>
    /// <list type="bullet">
    ///   <item><b>Very Smooth</b> — the floor. Old 2 GB-VRAM cards. Assume everything is scarce and
    ///   never spend a frame you do not have to <i>on the LLM</i>; the audio side is the exception, and
    ///   the reason is spelled out in the table's counter-intuition note.</item>
    ///   <item><b>Smooth</b> — GTX 1650 class, 4 GB laptop GPU. The reference machine this table was
    ///   tuned on, and the setting expected to feel good there.</item>
    ///   <item><b>Balanced</b> — ~6 GB class, RTX 4050 / 5050.</item>
    ///   <item><b>Fast</b> — RTX 4060 / 4070.</item>
    ///   <item><b>Very Fast</b> — RTX 3080 / 4080 and up.</item>
    /// </list>
    /// <para><b>Read those as bandwidth tiers, not compute tiers</b> (author's note, 2026-07-26). Not one
    /// of the four LLM budgets in this table is bound by shader throughput: <i>fetch</i> is host→VRAM
    /// transfer, <i>prefill steps</i> is CPU dispatch plus frame present, and <i>decode</i> is dominated
    /// by reading the weights once per token. The engine's own measurements say the same thing — INT8
    /// weights buy ~1.44× on decode (half the bytes to read) but only ~1.007× on prefill, where the
    /// per-chunk cost is already amortised over 8 tokens; and decode at this model size is documented as
    /// dispatch-bound rather than math-bound. So when a card does not match its tier, suspect memory
    /// bandwidth, PCIe link width, or a CPU that cannot issue fast enough — long before you suspect it is
    /// short on FLOPS.</para>
    public enum BackendTradeoffLevel
    {
        /// <summary>Hardware that has nothing to spare (2 GB VRAM, old cards). The model loads and thinks
        /// slowly and the game never stutters for it — while the VOICE gets the most frames of any level,
        /// because on a card this slow that is the only thing that keeps audio unbroken.</summary>
        VerySmooth = 0,
        /// <summary>The GTX 1650 / 4 GB laptop tier — the reference machine for this table.</summary>
        Smooth = 1,
        /// <summary>Code default. A healthy mid-range GPU: a visible cost while the model works,
        /// none while it doesn't.</summary>
        Balanced = 2,
        Fast = 3,
        /// <summary>Model speed first. Expect a visible fps drop while loading and while replying, and
        /// the CHEAPEST voice pacing of any level — a card this fast synthesizes multiples of real-time
        /// on one tick per frame. Meant for a GPU with headroom to spare — see the fetch note in the
        /// table.</summary>
        VeryFast = 4,
    }

    /// <summary>
    /// EVERY per-frame budget the inference backends run on, as five fixed rows.
    ///
    /// <para><b>Why a table instead of a slider.</b> This replaced a continuous 0..1 dial
    /// (<c>SmoothVsSpeed</c>) that fed a <c>BiasMultiplier = 4^((s-0.5)*2)</c> into two self-tuning
    /// controllers, which then discovered the concrete values by measuring frame times. That design
    /// had three problems, all of them real and all of them observed:</para>
    /// <list type="number">
    ///   <item>Only the multiplier and two hard extremes were actually wired to the slider. The fetch
    ///   budget and both layer/token splits were independent hard-coded constants, so most of the dial's
    ///   apparent range did nothing.</item>
    ///   <item>The two controllers interpreted the same multiplier differently — prefill scaled the whole
    ///   frame budget, decode scaled only the headroom left after the scene's own cost — so one slider
    ///   position meant two different things depending on which phase you were in.</item>
    ///   <item>A self-tuning controller hides its own bugs. The prefill tuner compared a whole
    ///   vsync-padded frame against a budget representing the LLM's share of one, so on any scene below
    ///   ~80 fps it could only ever shrink; it sat pinned at its floor and prefill crawled at ~10-20
    ///   tok/s against a ~414 tok/s kernel ceiling. Nothing in the logs said so, because from the
    ///   controller's point of view it was working perfectly (found and fixed 2026-07-26).</item>
    /// </list>
    /// <para>Fixed rows cannot do any of that. What you set is what runs, it is one lookup to see why a
    /// frame cost what it cost, and calibrating a machine means editing five numbers in one place.
    /// The price is that nothing adapts to the GPU on its own: a new machine needs the dial moved by
    /// hand. That is the intended trade (user 2026-07-26). The TTS rows joined on 2026-07-27, replacing
    /// three more adaptive loops in PocketTTSVoice — a persisted prebuffer/chunk escalation ladder, a
    /// tick-cost calibrator and a refill-rate EMA — for exactly the same reasons, plus one specific to
    /// audio: every rung those loops climbed cost the player an audible gap to learn something the dial
    /// already knew.</para>
    ///
    /// <para><b>The rows.</b></para>
    /// <code>
    ///                                     VerySmooth   Smooth   Balanced     Fast   VeryFast
    ///   fetch MB/frame                             4        8         16       24         32
    ///   slow prefetch (÷8)                       0.5      1.0        2.0      3.0        4.0
    ///   prefill steps/frame                        1        3          6       12         25
    ///   decode tokens/frame                        1        1          1        2          3
    ///   tts ticks/frame (speaking)                 4        4          3        2          1
    ///   tts ticks/frame (silent refill)            6        6          4        3          2
    ///   tts prebuffer seconds                    1.5      1.0       0.75      0.5        0.5
    ///   tts streamChunkFrames                     16       16          8        8          8
    ///   tts refill floor seconds                1.25      1.0       0.75      0.6        0.5
    ///   tts cede headroom seconds                2.0      2.0        2.0      1.5        1.0
    ///   tts MACs per heavy tick                 900M     900M       1.5G     2.5G         4G
    /// </code>
    ///
    /// <para><b>fetch</b> — bytes of model weights uploaded to the GPU per frame, for whichever backend
    /// is streaming (it is ONE global budget, <c>LLM.UploadBudgetBytes</c>, that every LLM/ASR/TTS loader
    /// samples). Never a whole-model load in one frame: weights always stream, this is only the ceiling
    /// one frame may carry. Measured note carried over from that field: 24 MB/frame produced a ~10.7 ms
    /// worst slice and a visible fps dip, 8 MB keeps slices roughly 3× smaller. So
    /// <see cref="BackendTradeoffLevel.Fast"/> and above are past a known-bad point by choice, not by
    /// accident.</para>
    ///
    /// <para><b>slow prefetch</b> — the same budget while the player is merely walking INTO a prefetch
    /// zone and nothing is waiting on the model. Always fetch ÷ <see cref="SlowPrefetchDivisor"/>, so it
    /// tracks the dial instead of being re-derived at the call site (it used to be an inline <c>/8</c>
    /// inside NPCChatBase). Opening the dialogue boosts back to the full row value.</para>
    ///
    /// <para><b>prefill steps/frame</b> — how many of <c>ForwardYielding</c>'s yields the caller packs
    /// into one Unity frame while processing the prompt. The unit matters: that coroutine yields once per
    /// transformer layer plus once at the end, so a 24-layer model is <b>25 steps per 8-token chunk</b>,
    /// and throughput is <c>tok/s ≈ 8 × fps × steps / 25</c>. At 38 fps that is ~12 tok/s at 1 step and
    /// ~165 tok/s at 25 (measured; frames land near 44 ms there). 25 = one whole chunk per frame, the
    /// implementation limit — there is no point going higher. Prefill is a loading moment with nothing
    /// on screen to disturb, which is why the top row spends frames freely.</para>
    ///
    /// <para><b>decode tokens/frame</b> — tokens generated before the frame is handed back to rendering.
    /// Decode is autoregressive, so a token's RESULT cannot be split across frames the way prefill can;
    /// through the sync-decode era (2026-07-26 → 2026-08-02) that made this the only decode lever, each
    /// token costing its full stall (~36 ms on a GTX 1650) in one frame, so 2 meant a ~72 ms frame —
    /// deliberately reserved for the top rows, with text appearing in visible bursts of N. Under sliced
    /// decode (2026-08-02 — see the sync-vs-sliced paragraph at the end) a token's ISSUE spans several
    /// frames by construction, so rows above 1 no longer pack whole tokens into one frame; the column
    /// survives as the decode loops' trailing-yield cadence and as the record of what the top rows
    /// meant.</para>
    ///
    /// <para><b>THE TTS ROWS RUN THE OTHER WAY ROUND, AND THAT IS THE POINT</b> (2026-07-27). Read the
    /// four LLM rows left→right and the budgets grow: a weak machine spends less, a strong one more. Read
    /// the tts tick rows and they SHRINK. That is not a transcription slip. Framerate is negotiable and
    /// audio continuity is not — a dropped frame is a blink, a dry ring is the NPC stopping mid-word — so
    /// a weak machine has to spend MORE of every frame defending the audio, not fewer. Which is precisely
    /// why the dial is no longer called "smooth ⇄ fast" (renamed 2026-07-27): it does not state a
    /// preference, it states <i>how capable is this machine</i>, and each backend derives from that
    /// whatever its own answer happens to be. Same reasoning behind the cede headroom falling as the tier
    /// rises: a fast card refills any reserve it spends, a slow one does not.</para>
    ///
    /// <para><b>tts ticks/frame</b> — GPU-heavy PocketTTS pipeline ticks (a prefill chunk, a Mimi-decode
    /// slice) the voice's pump may issue in one frame. THIS is the number that decides whether the audio
    /// survives. Measured on the reference 1650 at 19-38 fps:</para>
    /// <list type="bullet">
    ///   <item>At <b>1 tick/frame during audible playback, synthesis runs 0.35-0.8× real-time</b> — below
    ///   playback — so the ring drains monotonically and NO prebuffer, however large, can save it: a
    ///   deficit integrated over a long enough reply always reaches zero. At 4 ticks it is 1.3-2.5×, so
    ///   the ring GROWS while speech plays and can absorb the fixed ~470 ms per-clause "dead window"
    ///   (each clause restarts a ~150-row prompt prefill that yields ~24 frame-breaks and produces zero
    ///   samples).</item>
    ///   <item>So the tick count is the value that actually fixes underruns — and a <b>small prebuffer
    ///   becomes possible as a consequence</b>, which is why the prebuffer row is short. Observed on one
    ///   reply: prebuffer 1.0 s gave TTFA 3472 ms where 3.0 s gave 6203 ms. The deleted ladder had it
    ///   backwards, buying ~3.2 s of latency on every reply to fight a deficit no buffer can fight.</item>
    ///   <item>The <b>silent refill</b> column is always the higher of the two: nothing is audible during
    ///   a prebuffer, a mid-reply re-gate or a clause's prompt prefill, so frame smoothness buys
    ///   literally nothing there — push hard and end the silence sooner.</item>
    /// </list>
    ///
    /// <para><b>tts prebuffer seconds</b> — audio banked before playback starts, i.e. time-to-first-audio
    /// traded against underrun safety. Short by design now (see above): the ticks defend the ring, this
    /// only covers the very first clause's dead window.</para>
    ///
    /// <para><b>tts streamChunkFrames</b> — 80 ms latent frames accumulated per streaming Mimi decode
    /// (8 = 0.64 s of audio per decode). Bigger chunks amortise the fixed per-chunk/per-clause cost over
    /// more audio, at the price of a coarser text-reveal cadence and slightly later first audio. The
    /// empirical evidence that it helps is that the deleted escalation ladder, tuning itself
    /// independently, climbed to 12 and then to 16 on this very machine — so the low rows simply START
    /// where it ended up, without paying an audible gap per rung climbed.</para>
    ///
    /// <para><b>tts refill floor seconds</b> — ring seconds below which the voice stops ceding
    /// altogether. This, NOT the cede headroom, is where the ring actually parks through a decode
    /// phase: both ceding bands spend faster than synthesis earns (see FramePacing's fixed-point
    /// argument), so the level falls until it reaches the one band that never cedes — just under this
    /// floor. Which makes it the number that has to clear a clause's dead window, and for a long time
    /// it did not: a global 0.5 s against a measured 404-448 ms `synth→first-audio` (2026-07-28). Read
    /// the row's value as "the reserve the voice defends", and the headroom below as "when it starts
    /// spending".</para>
    ///
    /// <para><b>tts cede headroom seconds</b> — ring seconds above which a speaking voice cedes EVERY
    /// frame the LLM is decoding in (it has audio banked; the LLM needs the frame more). With the cede
    /// RATE bounded by <c>InferencePerf.TtsCedeFrameStride</c> this bounds how fast the reserve is
    /// spent; it is not where the ring settles. 2.0 on the low rows, not 1.0
    /// (2026-07-26): at 1.0 the arbiter called a 1 s ring "banked" that a single clause transition can
    /// empty, and the pump's cede check returns BEFORE the per-clause prefill boost is computed, so a
    /// ceding voice made no prefill progress at all. It falls to 1.0 at the top row because a card
    /// synthesizing at several times real-time refills whatever it spends.</para>
    ///
    /// <para><b>tts MACs per heavy tick</b> — slice size of one heavy tick, read straight off this table
    /// by <c>PocketTTS.GpuMacsPerTick</c>. Fixed per tier since 2026-07-27; before that a calibrator
    /// walked it between 200M and 4G chasing a 3-7 ms measured tick cost, starting from a 900M default
    /// (which is why 900M is still what the two low rows say). Deleted for the usual reason plus a
    /// specific one: slices are derived FROM this value, so a smaller slice means MORE frame-bound ticks
    /// per clause prefill — the calibrator's shrink branch fed straight into the starvation it existed to
    /// prevent.</para>
    ///
    /// <para><b>Not in the table: sync vs sliced decode.</b> The old code chose between reading a token
    /// back inside the same frame (sync — the frame stalls for the token, tok/s ≈ fps) and picking it up
    /// a few frames later via AsyncGPUReadback (async — clean frames, but ~3.5 frames per token, i.e.
    /// tok/s ≈ fps/3.5). From 2026-07-26 it was always sync, and the whole decode auto-tuner went with
    /// it. The reason was that async is the one knob whose right answer depends on the hardware rather
    /// than on taste: the old code carried a <c>MinUsableTokS = 12</c> floor precisely because on a weak
    /// GPU async produces a dribble (38 fps ÷ 3.5 ≈ 11 tok/s — below its own floor), and a fixed table
    /// cannot measure its way to that decision.</para>
    /// <para><b>Reversed 2026-08-02</b> — not because that reasoning was wrong, but because its premise
    /// was withdrawn. The author's smoothness mandate ("nu mai conteaza delayul, doar sa fie smooth —
    /// niciun frame peste 50 ms, ideal nici peste 33") strikes tok/s from the criteria list entirely,
    /// and with it the whole hardware-dependence argument: a knob whose right answer no longer depends
    /// on anything is exactly what a const can state. What sync decode could never fix is that the
    /// token's ENTIRE burst — ~30-55 ms of GPU on the reference 1650, because decode reads every weight
    /// once per token — lands in ONE frame, measured as the 33 ms mean / 55 ms p95 / 57-80 ms max GEN
    /// rows of every talk-perf report; no readback strategy softens a lump already in the queue. So
    /// decode is now ISSUED in slices (<c>InferencePerf.LlmDecodeSliceLayers</c> layers per frame, the
    /// lm_head alone in its own frame — it is ~half the per-token bytes) and the token is read back
    /// asynchronously. The dribble the 2026-07-26 decision guarded against is the price, paid knowingly:
    /// speech at ~3 words/s is the real pacing bottleneck, and the ~5-8 tok/s this lands at still clears
    /// it. The slice width is deliberately NOT a tier column — a ceiling on what one frame may be asked
    /// to carry is the mandate itself, identical on every machine, not a machine preference; the
    /// constant's own docs carry the arithmetic.</para>
    /// </summary>
    public static class BackendTradeoffTable
    {
        /// <summary>One row of the dial. Everything is per-frame; nothing here is a multiplier.</summary>
        public readonly struct Row
        {
            /// <summary>Display name for the inspector, next to the dial.</summary>
            public readonly string label;
            /// <summary><b>Model-weight bytes uploaded to the GPU per frame.</b> A CEILING on one
            /// frame's transfer, never a whole-model load: weights always stream across many frames.
            /// <para>Becomes <c>LLM.UploadBudgetBytes</c>, which every model's <c>*Weights</c> upload
            /// pump re-reads each frame — so it applies to the LLM, both STT models and all four TTS
            /// voices at once. Walk-up prefetch uses <see cref="SlowFetchBytesPerFrame"/> = this ÷
            /// <see cref="BackendTradeoffTable.SlowPrefetchDivisor"/>. Cite the constant, never the
            /// number: this comment said ÷ 8 for three days after the divisor became 16.</para>
            /// <para>RAISE → the model is ready sooner, each loading frame carries a fatter slice.
            /// LOWER → clean frames while loading, longer wait before the NPC can talk.</para>
            /// <para><b>Too high:</b> a visible fps dip while models stream in — measured, 24 MB/frame
            /// gave a ~10.7 ms worst slice, 8 MB keeps it roughly 3× smaller. That is why the top two
            /// tiers are deliberately past a known-bad point. <b>Too low:</b> the player reaches the NPC
            /// before it is loaded and waits, staring at a disabled Send button.</para></summary>
            public readonly int fetchBytesPerFrame;
            /// <summary><b>Prompt-prefill yields packed into one Unity frame.</b> NOT layers — the unit
            /// is <c>ForwardYielding</c>'s yields, which is once per transformer layer plus one at the
            /// end, so a 24-layer model is <b>25 steps per 8-token chunk</b> and 25 means "a whole chunk
            /// in one frame", the implementation ceiling.
            /// <para>Read by <c>Qwen3_5.ForwardPromptChunked</c>, <c>Gemma3</c> and <c>MiniCPM5</c>
            /// (identical loops). Throughput is <c>tok/s ≈ 8 × fps × steps / 25</c>.</para>
            /// <para>RAISE → the prompt is processed sooner, frames during the load get chunkier.
            /// LOWER → clean frames, longer wait before the first token.</para>
            /// <para><b>Too low:</b> a long pause after you press Send, before anything appears, and on
            /// first contact a multi-second wait while the system prompt is processed (at 1 step it is
            /// ~12 tok/s — an 800-token prompt takes over a minute). <b>Too high:</b> a visible freeze
            /// while the reply spins up.</para></summary>
            public readonly int prefillStepsPerFrame;

            /// <summary><b>Whole tokens generated before the frame is handed back to rendering.</b>
            /// Decode is autoregressive, so unlike prefill a single token's RESULT cannot be split
            /// across frames; through the sync-decode era this was the only decode lever, its unit
            /// whole tokens.
            /// <para>Read by <c>Qwen3_5.Generate</c>/<c>ChatCore</c>'s decode loop. Since 2026-08-02
            /// (sliced decode — the class docs' last paragraph) each token's ISSUE already spans
            /// several frames, so this only sets the loops' trailing-yield cadence; the sync-era
            /// meanings below are kept for the day the const flips back.</para>
            /// <para>RAISE → text arrives faster, each frame swallows N token stalls (~36 ms each on a
            /// GTX 1650, so 2 ≈ a 72 ms frame). LOWER → smoothest frames while the NPC talks.</para>
            /// <para><b>Too high:</b> the game feels frozen while a reply streams, AND the text appears
            /// in visible bursts of N words rather than flowing — this one is a look change, not just a
            /// performance one. <b>Too low:</b> nothing breaks; the reply is simply slower.</para></summary>
            public readonly int decodeTokensPerFrame;

            /// <summary><b>Heavy TTS pipeline ticks per frame while speech is AUDIBLE.</b> A "heavy tick"
            /// is one GPU-slice yield from the synthesizer (a prefill chunk or a Mimi-decode slice).
            /// <b>FALLS as the tier rises</b> — see the counter-intuition paragraph in the class docs.
            /// <para>Read by <c>PocketTTSVoice.PumpPipeline</c>'s <c>maxHeavyTicks</c>.</para>
            /// <para>THIS IS THE VALUE THAT DECIDES WHETHER AUDIO BREAKS. Synthesis throughput is
            /// frame-rate bound, not GPU bound: measured at 1 tick/frame it runs 0.35-0.8× real-time
            /// (the ring drains no matter how large the prebuffer), at 4 ticks 1.3-2.5× (the ring grows).
            /// </para>
            /// <para><b>Too low:</b> speech stops partway through a sentence and resumes — the symptom
            /// that cost a full day to trace. Look for <c>ring starved mid-reply</c> in the console and a
            /// non-zero <c>in-reply silence</c> on the per-reply line. <b>Too high:</b> dropped frames
            /// while the NPC talks (accepted deliberately at the low tiers: audio continuity is
            /// non-negotiable, framerate is).</para></summary>
            public readonly int ttsSpeakingTicksPerFrame;

            /// <summary><b>Heavy TTS ticks per frame while nothing is audible</b> — filling the initial
            /// prebuffer, recovering from a mid-reply re-gate, or running a clause's prompt prefill.
            /// Always ≥ the speaking value: with nothing to disturb, frame smoothness buys nothing.
            /// <para>Read by the same <c>maxHeavyTicks</c>; also multiplied by
            /// <c>InferencePerf.TtsSilentRefillBudgetScale</c> for the per-frame CPU budget.</para>
            /// <para><b>Too low:</b> a long silence before the NPC's first word (watch <c>buffer-gate</c>
            /// in the TTFA line) and a long hole after any starve. <b>Too high:</b> a frame-rate dip
            /// during a gap nobody was listening to — the cheapest place in the engine to spend
            /// frames.</para></summary>
            public readonly int ttsSilentTicksPerFrame;

            /// <summary><b>Seconds of audio banked before playback is allowed to start.</b> Pushed onto
            /// <c>PocketTTSVoice.prebufferSeconds</c> by <c>NPCChatBase.EnsureVoice</c>.
            /// <para>A prebuffer <i>delays</i> the problem, it cannot fix it: if synthesis is slower
            /// than playback the ring drains through any starting level, which is exactly why the old
            /// self-escalating ladder kept climbing to 3.0 s and never stopped starving. Keep this
            /// SMALL and let the tick count do the work.</para>
            /// <para>RAISE → more runway before the first dropout, later first word. LOWER → the NPC
            /// answers sooner.</para>
            /// <para><b>Too high:</b> dead air before every reply — measured 3472 ms TTFA at 1.0 s
            /// versus 6203 ms at 3.0 s, i.e. this alone was 2.7 s of the wait. <b>Too low:</b> playback
            /// starts on almost nothing and the first clause boundary drains it.</para></summary>
            public readonly float ttsPrebufferSeconds;

            /// <summary><b>80 ms latent frames per streaming Mimi decode</b> — the audio-generation
            /// cadence, NOT Unity frames. 8 = 0.64 s of audio per decode, 16 = 1.28 s. Pushed onto
            /// <c>PocketTTSVoice.streamChunkFrames</c> by <c>NPCChatBase.EnsureVoice</c>.
            /// <para>Why 12→16 on Smooth (2026-07-28): a Mimi flush costs <b>16 heavy ticks whatever the
            /// chunk size</b> — with a 56-latent window every <c>SlicesFor</c> resolves to 1, so the count
            /// is just the unconditional yields (1 quant/upsample + 4×2 transformer + 1 conv0 + 2×3
            /// SEANet). Fewer, fatter flushes therefore cut ticks per second of audio from 16.7 to 12.5,
            /// and that — not raw MAC count — is the operative saving, because the pump rations TICKS.</para>
            /// <para>Do NOT reason from the naive window ratio (48/8, 52/12, 56/16). That prices the whole
            /// <c>newFrames + MIMI_DECODE_CTX</c> window and an earlier version of this comment claimed
            /// "~19% more speech per unit of work" from it. Wrong: the #30 tail-restricted decode already
            /// skips ~60-75% of the window's dispatch, and summing the real per-slice MACs gives
            /// 5.45 / 4.76 / 4.42 GMAC per second of audio for chunk 8 / 12 / 16 — the 12→16 saving is
            /// <b>~7%</b>. Same direction, much smaller than advertised.</para>
            /// <para><b>Too low:</b> more fixed cost per second of speech, so the ring falls behind.
            /// <b>Too high:</b> a coarser text-reveal cadence, AND it used to mean a later first word
            /// and chunk-rounded starve recovery: the ring only ever jumped by whole chunks, so any
            /// prebuffer in (0.17 s, chunk×0.08 s] opened at the chunk boundary (measured
            /// <c>buffer-gate</c> rose from ~850 ms at chunk 12 to 1207-1404 ms at 16), and a 0.25 s
            /// re-gate after a dry ring still waited on a 1.28 s lump (measured 1.20-2.80 s of
            /// re-gated silence per reply, 2026-07-30).</para>
            /// <para>The panic hurry-flush (2026-07-30, <c>InferencePerf.TtsPanicFloorSeconds</c> /
            /// <c>PocketTTS.StreamHurry</c>) removed that rounding: while playback is gated or the
            /// ring is in the panic band, delivery granularity is <c>StreamHurryMinFrames</c> (4),
            /// not this column. What this column still sets: the amortization (see above), the
            /// sawtooth amplitude the ring swings through ABOVE the panic band — i.e. how often the
            /// panic band is even entered — and the clause-boundary supply gap the refill floor must
            /// cover, so it and <c>ttsRefillFloorSeconds</c> still move together.</para></summary>
            public readonly int ttsStreamChunkFrames;

            /// <summary><b>Ring seconds below which the voice stops being polite</b> — it pumps EVERY
            /// frame, at the silent tick count, and the LLM waits. Read as <c>lowRing</c> in
            /// <c>PocketTTSVoice.Update</c> (the cede gate, the tick cap, and the re-gate threshold).
            /// <para>This number has one job and it is arithmetic: it must exceed the longest stretch
            /// during which synthesis produces NO samples, because playback drains at 1× through it.
            /// That stretch is a clause's prompt prefill, and it is measured on every reply — the
            /// <c>synth→first-audio</c> field of the TTFA line. On the reference GTX 1650 that reads
            /// 404-448 ms, so the old global 0.5 s left ~50 ms of margin and the ring died six times in
            /// one session. Budget roughly 2× the observed dead window.</para>
            /// <para>Keep it BELOW <c>ttsCedeHeadroomSeconds</c>: floor = "help me now", headroom =
            /// "I'm comfortable, take the frame". Inverting them makes the arbiter incoherent.</para>
            /// <para>RAISE → the voice defends a bigger reserve and steals more frames from decode.
            /// LOWER → faster text, and the next clause boundary is a coin flip.</para>
            /// <para><b>Too low:</b> `ring starved mid-reply` warnings, audio stopping mid-clause.
            /// <b>Too high:</b> the voice almost never cedes, so text generation crawls while the NPC
            /// speaks.</para></summary>
            public readonly float ttsRefillFloorSeconds;

            /// <summary><b>Ring seconds above which a speaking voice yields its frame to the decoding
            /// LLM.</b> Read by the <c>FramePacing.LlmBusy</c> cede gate in <c>PocketTTSVoice.Update</c>.
            /// <para>Beware the trap this knob was fixed for: the ring is an integrator, so a pure LEVEL
            /// test has its fixed point at the level itself — the arbiter spends the buffer down to
            /// exactly this number and parks there with no reserve. That is survivable only because
            /// <c>InferencePerf.TtsCedeFrameStride</c> now bounds the cede RATE as well.</para>
            /// <para>RAISE → the voice keeps more reserve, the LLM decodes a little slower while the NPC
            /// speaks. LOWER → faster text, less audio safety margin.</para>
            /// <para><b>Too low:</b> the reply enters its post-decode phase with no runway and the first
            /// clause transition empties the ring. <b>Too high:</b> visibly slower text generation
            /// during speech.</para></summary>
            public readonly float ttsCedeHeadroomSeconds;

            /// <summary><b>MAC budget of one heavy TTS tick</b> — how much GPU work a single slice
            /// carries. Read via <c>PocketTTS.GpuMacsPerTick</c> by <c>PocketTTSFlowLM.LinearRows</c> and
            /// <c>PocketTTSMimi</c>, which derive their slice COUNT from it.
            /// <para>Note the second-order effect that made this dangerous to auto-tune: slices are
            /// derived from this number, so halving it roughly DOUBLES the frame-bound tick count of the
            /// next prefill and the next Mimi flush. The old calibrator shrank it by 10% per frame it
            /// mispriced, feeding straight back into starvation.</para>
            /// <para>RAISE → fewer, fatter slices: less per-tick overhead, chunkier frames.
            /// LOWER → many thin slices, smoother frames, more total pacing overhead.</para>
            /// <para><b>Too high:</b> one tick overruns the frame budget and the pump is cut short.
            /// <b>Too low:</b> the tick count explodes and synthesis falls behind playback even at a
            /// generous ticks-per-frame.</para></summary>
            public readonly long ttsMacsPerTick;

            public Row(string label, int fetchMB, int prefillSteps, int decodeTokens,
                       int ttsSpeakTicks, int ttsSilentTicks, float ttsPrebufferSec,
                       int ttsChunkFrames, float ttsRefillFloorSec, float ttsCedeHeadroomSec,
                       long ttsMacs)
            {
                this.label = label;
                fetchBytesPerFrame = fetchMB * 1024 * 1024;
                prefillStepsPerFrame = prefillSteps;
                decodeTokensPerFrame = decodeTokens;
                ttsSpeakingTicksPerFrame = ttsSpeakTicks;
                ttsSilentTicksPerFrame = ttsSilentTicks;
                ttsPrebufferSeconds = ttsPrebufferSec;
                ttsStreamChunkFrames = ttsChunkFrames;
                ttsRefillFloorSeconds = ttsRefillFloorSec;
                ttsCedeHeadroomSeconds = ttsCedeHeadroomSec;
                ttsMacsPerTick = ttsMacs;
            }
        }

        /// <summary>How much slower a walk-up prefetch is than the level's full budget. The player is
        /// still approaching, nothing is waiting on the model, so the stream hides under the walk
        /// instead of competing with the frame that renders it.
        /// <para>8 → 16 (2026-08-03, the last smoothness item): at ÷8 the walk-up itself ran a
        /// measured 110 → 59 fps for the whole prefetch window — the upload slice plus its driver
        /// residency work costs ~8 ms on the reference card, which is the FRAME budget, not a
        /// hide-under-the-walk budget. ÷16 halves the slice; the prefetch stretches to ~2× wall
        /// time, and whatever is still streaming when the dialogue opens is finished by
        /// BoostPrefetch, which has owned exactly that job since 2026-07-30.</para></summary>
        public const int SlowPrefetchDivisor = 16;

        /// <summary>False since 2026-08-02: decode issue is sliced across frames and the token read
        /// back asynchronously — see the class docs' sync-vs-sliced paragraph for the full history
        /// (always-sync was itself the 2026-07-26 verdict, retired when the smoothness mandate struck
        /// tok/s from the criteria). Still a const and still not a dial, for the same reason in
        /// mirror image: the right answer now depends on nothing. Flipping it back restores the
        /// one-burst + blocking-readback decode wholesale — A/B archaeology only.</summary>
        public const bool UseSyncDecode = false;

        // Indexed by (int)BackendTradeoffLevel. THE ONLY PLACE THESE NUMBERS EXIST. If a value shows up
        // anywhere else in the engine, that is the bug this table was written to prevent.
        // ttsFloor must stay BELOW ttsCede on every row (floor = "help me now", cede = "take the
        // frame"), and ABOVE that tier's worst `synth→first-audio` — the clause dead window playback
        // has to coast through. 2026-07-28: the floor was a global 0.5 s while the reference machine
        // measured 404-448 ms, i.e. 50 ms of margin; six starves in one session.
        // 2026-08-02, the smoothness mandate (author: "nu vreau frame drops mari... nici macar
        // 50ms" — no single frame may carry a burst; latency is explicitly NOT a criterion any
        // more): the silent-refill columns used to run ABOVE the speaking ones on the reasoning
        // that "frame smoothness buys nothing while nothing is audible". That reasoning was
        // wrong on its face — the player is watching the text stream during exactly those
        // windows — and the 6-tick sprints were the 24-36 ms half of every 100 ms collision
        // frame. Silent now equals speaking on every row (the sprint is gone, not rebalanced),
        // and the prebuffer is deeper to buy the same starve-safety with TIME instead of with
        // per-frame GPU: the dead windows a sprint used to outrun are now simply covered by a
        // bank that playback cannot outrun. TTFA pays for it; that is the accepted trade.
        //                          label       fetchMB  prefill  decode  ttsSpk  ttsSil  ttsPre  ttsChunk  ttsFloor  ttsCede  ttsMACs
        static readonly Row[] Rows =
        {
            new Row("Very Smooth",         4,       1,      1,      4,      4,   2.0f,      16,    1.25f,    2.0f,   900_000_000),
            new Row("Smooth",              8,       3,      1,      4,      4,   1.5f,      16,    1.0f,     2.0f,   900_000_000),
            new Row("Balanced",           16,       6,      1,      3,      3,   1.0f,       8,    0.75f,    2.0f, 1_500_000_000),
            new Row("Fast",               24,      12,      2,      2,      2,   0.75f,      8,    0.6f,     1.5f, 2_500_000_000),
            new Row("Very Fast",          32,      25,      3,      1,      1,   0.5f,       8,    0.5f,     1.0f, 4_000_000_000),
        };

        /// <summary>The level in force. Set from the NPC inspector's Backend Tradeoff dial; engine-wide,
        /// because the budgets it controls are engine-wide (one GPU, one frame).</summary>
        public static BackendTradeoffLevel Level = BackendTradeoffLevel.Balanced;

        public static int LevelCount => Rows.Length;

        /// <summary>The active row. Clamped rather than throwing: a serialized enum from an older scene
        /// must not be able to break a boot.</summary>
        public static Row Current => Rows[Mathf.Clamp((int)Level, 0, Rows.Length - 1)];

        public static Row At(BackendTradeoffLevel level) => Rows[Mathf.Clamp((int)level, 0, Rows.Length - 1)];

        /// <summary>Inspector name of a level ("Balanced"), without exposing the row struct.</summary>
        public static string LabelOf(BackendTradeoffLevel level) => At(level).label;

        public static string Label => Current.label;

        // ---- the derived budgets: read these, never the table ----------------------------------

        /// <summary>Full-speed weight-upload ceiling for this level, bytes per frame.</summary>
        public static int FetchBytesPerFrame => Current.fetchBytesPerFrame;

        /// <summary>Walk-up prefetch ceiling, bytes per frame. Floored so a very small fetch budget
        /// cannot round down to a stream that never finishes.</summary>
        public static int SlowFetchBytesPerFrame =>
            Mathf.Max(64 * 1024, Current.fetchBytesPerFrame / SlowPrefetchDivisor);

        /// <summary>Prefill coroutine yields packed into one frame.</summary>
        public static int PrefillStepsPerFrame => Mathf.Max(1, Current.prefillStepsPerFrame);

        /// <summary>Whole tokens decoded before the frame is handed back.</summary>
        public static int DecodeTokensPerFrame => Mathf.Max(1, Current.decodeTokensPerFrame);

        /// <summary>Heavy TTS ticks one frame may issue while speech is audible.</summary>
        public static int TtsSpeakingTicksPerFrame => Mathf.Max(1, Current.ttsSpeakingTicksPerFrame);

        /// <summary>Heavy TTS ticks one frame may issue while nothing is audible (refill / clause
        /// prefill).</summary>
        public static int TtsSilentTicksPerFrame => Mathf.Max(1, Current.ttsSilentTicksPerFrame);

        /// <summary>Seconds banked before playback starts. Pushed onto the voice component by
        /// NPCChatBase, like every other per-NPC voice setting.</summary>
        public static float TtsPrebufferSeconds => Current.ttsPrebufferSeconds;

        /// <summary>Latent frames per streaming decode chunk. Pushed onto the voice component too.</summary>
        public static int TtsStreamChunkFrames => Mathf.Max(1, Current.ttsStreamChunkFrames);

        /// <summary>Ring seconds below which the voice pumps every frame at the silent tick count and
        /// the LLM waits. Must exceed the tier's worst clause dead window — see the field docs.</summary>
        public static float TtsRefillFloorSeconds => Current.ttsRefillFloorSeconds;

        /// <summary>Ring seconds above which a speaking voice cedes every LLM decode frame.</summary>
        public static float TtsCedeHeadroomSeconds => Current.ttsCedeHeadroomSeconds;

        /// <summary>MAC budget of one heavy TTS tick — what <c>PocketTTS.GpuMacsPerTick</c> returns.</summary>
        public static long TtsMacsPerTick => Current.ttsMacsPerTick;

        /// <summary>One line for the boot log, so a session's pacing is never a mystery after the fact.
        /// The old auto-tuner printed a measured verdict here; a fixed table can state it up front.</summary>
        public static string Summary =>
            $"Backend Tradeoff: {Label} — fetch {FetchBytesPerFrame / 1e6:0.0} MB/frame " +
            $"(walk-up {SlowFetchBytesPerFrame / 1e6:0.0}), prefill {PrefillStepsPerFrame} steps/frame, " +
            $"decode {DecodeTokensPerFrame} tok/frame, " +
            (UseSyncDecode ? "sync decode; " : $"sliced decode ({InferencePerf.LlmDecodeSliceLayers} layers/frame); ") +
            $"tts {TtsSpeakingTicksPerFrame}/" +
            $"{TtsSilentTicksPerFrame} ticks/frame (speaking/silent), floor {TtsRefillFloorSeconds:0.##}s, " +
            $"prebuffer {TtsPrebufferSeconds:0.##}s, " +
            $"chunk {TtsStreamChunkFrames}f, cede above {TtsCedeHeadroomSeconds:0.#}s";

        // ================== SYMPTOM → COLUMN, the table read backwards =========================
        // Every field above documents what happens when IT is wrong. This is the same knowledge
        // indexed the other way, because in practice you start from something you saw or heard in the
        // game and need to know which column to reach for. Each row names the console line that
        // confirms it, so a guess is never necessary.
        //
        //  WHAT YOU NOTICE                                  COLUMN               CONFIRM IT WITH
        //  ---------------------------------------------------------------------------------------
        //  speech stops mid-sentence, then resumes          ttsRefillFloorSec    "ring starved mid-reply"
        //    (words are never lost — playback re-gates)       (raise)            names floor vs dead window
        //    FIRST: is the floor above the clause dead                            side by side — if the
        //    window the warning prints? that ordering is                          floor is the smaller
        //    the whole diagnosis. ticks only after.                               number, that is the bug
        //  long dead air before the NPC's first word         ttsPrebufferSeconds "TTFA … buffer-gate NNNN ms"
        //                                                      (lower)
        //  gap at every sentence boundary, rhythmically      ttsStreamChunkFrames "TTFA … synth→first-audio"
        //    (the per-clause dead window, not a dropout)       (raise)
        //  frames drop only while the NPC is speaking        ttsSpeakingTicks     none — expected at low
        //                                                      (lower)            tiers, audio wins
        //  frames drop only while models load                fetchBytesPerFrame  "[GPU] … MB/frame"
        //                                                      (lower)
        //  player arrives before the NPC can answer          fetchBytesPerFrame  "[GPU] … SLOW prefetch"
        //                                                      (raise)
        //  long wait after Send, before any text             prefillStepsPerFrame "system prompt computed
        //                                                      (raise)            (N tokens, NNNN ms)"
        //  text appears in bursts of N words                 decodeTokensPerFrame none — that IS N
        //    (sync-decode era only; sliced decode              (lower)
        //    streams token by token by construction)
        //  whole game hitches while a reply streams          InferencePerf.       "Qwen3.5 decode: N tok
        //    (2026-08-02: the burst is sliced now, so a        LlmDecodeSliceLayers  in N.Ns (held N frames)"
        //    GEN-frame hitch means the slice is too fat —      (lower)             + talk-perf GEN rows
        //    the decode column no longer packs whole tokens)
        //  text generation crawls while the NPC talks        ttsCedeHeadroom      "TtsDeferrals"
        //                                                      (lower)
        //  everything is slightly behind on a new machine    the whole dial        "Backend Tradeoff: …"
        //                                                      (move one tier)    at boot
        //
        // If a symptom is NOT in this list, it is probably not a pacing problem — look for a bug before
        // reaching for the dial. Two of today's three "pacing" symptoms turned out to be logic errors
        // (a reveal comparing against the wrong playback position, and a re-gate that re-armed on the
        // full prebuffer mid-reply), and no dial value would have fixed either.
    }
}
