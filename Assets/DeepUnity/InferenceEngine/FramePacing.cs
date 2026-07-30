using UnityEngine;

namespace DeepUnity
{
    // #29 cross-engine GPU frame arbiter.  When an NPC both GENERATES (LLM decode bursts) and
    // SPEAKS (streaming TTS ticks), the two engines used to issue their GPU bursts into the same
    // frame's queue — the 22-27 ms GEN+SPK+AUD band in the talk-perf report. The LLM has no
    // slack (each token's burst gates the next through a readback), while streaming TTS has
    // plenty (pocket-tts only needs ~12.5 latents/s and the ring buffers seconds ahead). So the
    // LLM marks every frame it issues GPU work, and TTS voices cede those frames whenever the
    // ring can afford it (see PocketTTSVoice.PumpPipeline) — at a bounded RATE, since "can afford
    // it" is a level test and a level test alone drains the ring to empty (see LlmBusy below).
    public static class FramePacing
    {
        static int llmIssueFrame = int.MinValue;

        /// <summary>LLM dispatch sites (prefill + decode) call this on every frame they issue GPU work.</summary>
        public static void NoteLlmIssue() => llmIssueFrame = Time.frameCount;

        /// <summary>True if the LLM issued GPU work this frame or the previous one AND the cede-rate
        /// bound below still allows this frame to be ceded. The 1-frame
        /// grace covers Update-vs-coroutine ordering: a voice's Update may run before the LLM
        /// coroutine resumes within the same frame, and while decoding the LLM issues (nearly)
        /// every frame — last frame's mark predicts this frame's burst.
        /// The sentinel guard matters: frameCount − int.MinValue OVERFLOWS negative, which read
        /// as "busy/starving since forever" at every session start — the un-guarded TtsStarving
        /// version held the LLM for the WHOLE first reply of each session (the "first message
        /// takes 5 s to speak" bug, 3166 held frames measured).</summary>
        public static bool LlmBusy => llmIssueFrame != int.MinValue && Time.frameCount - llmIssueFrame <= 1
                                      && CedeAllowedThisFrame;

        // ---- cede RATE bound (2026-07-26) -----------------------------------------------------
        // The pump's test is a LEVEL test ("ring ≥ the tier's cede headroom → cede", a
        // BackendTradeoffTable column since 2026-07-27), and a level test
        // is only half an arbitration rule: it says WHETHER there is surplus, never how fast the
        // arbiter may spend it. Unbounded, the answer was "all of it" — while decoding, the LLM
        // issues in (nearly) every frame, so a voice above the headroom produced NOTHING for the
        // whole decode phase while playback kept draining at 1×.
        // The ring is an INTEGRATOR, so that rule has a fixed point, and it is the BOTTOM of the
        // ring rather than the top: with synthesis at S× real-time the ring only holds when the pump
        // keeps 1/S of the frames, i.e. the sustainable cede share is 1 − 1/S (≈29% at the 1.3-1.5×
        // measured here). The pump's bands ask for 100% (above the headroom) and 50% (alternate
        // frames above the refill floor); BOTH overshoot that, so the ring falls until it reaches the
        // one band that does not cede at all — just under the refill floor. Measured 2026-07-26 on
        // the 1650 chat demo: a 3.0 s prebuffer accumulated by the underrun tuner, ring living at
        // ~0.5 s by mid-reply, two `ring starved mid-reply` events, 0.8-6.0 s of in-reply silence.
        // The pump's #33 clause-prefill boost is what turned the ~0.45 s per-clause dead window from
        // a hole into a survivable dip, and the cede check `return`s BEFORE it is even computed — so
        // above the old 1.0 s headroom a clause prefill made ZERO progress until the ring had drained
        // down to the headroom. That is the arbiter spending the prebuffer on nothing.
        // Fix: bound the RATE here — at most one ceded frame in InferencePerf.TtsCedeFrameStride.
        // This costs the LLM almost nothing, because the ring's fixed point ALREADY forced the
        // achieved cede share to ≈29%; the bound just makes the arbiter stop at that share while the
        // ring is still full instead of discovering it at empty. It is also the missing half of the
        // liveness argument: the reverse direction has been bounded since day one (LlmHoldMaxFrames —
        // "an unbounded hold serialized whole replies"), this one never was.
        static int lastCedeFrame = int.MinValue;
        static long ttsDeferrals;

        // Sentinel-guarded like the marks above (frameCount − int.MinValue overflows negative, which
        // would read as "ceded since forever" and pin the pump to full effort for a whole session).
        static bool CedeAllowedThisFrame =>
            lastCedeFrame == int.MinValue ||
            Time.frameCount - lastCedeFrame >= Mathf.Max(1, InferencePerf.TtsCedeFrameStride);

        /// <summary>Diagnostic: frames a TTS pump ceded to the LLM (#29 probe logs the per-turn
        /// delta). It is ALSO the rate bound's only observation point — the pump bumps this at each
        /// of its cede sites and calls nothing else into the arbiter, so the setter records WHICH
        /// frame was ceded. Several voices ceding in one frame still count as one ceded frame: the
        /// bound rations FRAMES, not voices.</summary>
        public static long TtsDeferrals
        {
            get => ttsDeferrals;
            set
            {
                if (value > ttsDeferrals) lastCedeFrame = Time.frameCount;
                ttsDeferrals = value;
            }
        }

        // ---- reverse direction (weak GPUs, e.g. GTX 1650): a SPEAKING voice whose ring has run
        // low while more synthesis is pending outranks tok/s — audible word-by-word dribble is
        // worse than a slower reply (the text reveal is audio-synced anyway). The voice marks the
        // starvation every frame it persists; LLM decode loops hold their next token burst while
        // the mark is fresh. Self-clearing: with the LLM idle the ring refills past the floor (or
        // the voice finishes), the mark goes stale, decode resumes. No deadlock possible.
        static int ttsStarveFrame = int.MinValue;

        /// <summary>Speaking TTS voices call this every frame their ring is under the refill floor
        /// with more synthesis pending.</summary>
        public static void NoteTtsStarving() => ttsStarveFrame = Time.frameCount;

        /// <summary>True if a TTS voice reported starvation this frame or the previous one
        /// (sentinel-guarded — see LlmBusy for the overflow bug this prevents).</summary>
        public static bool TtsStarving => ttsStarveFrame != int.MinValue && Time.frameCount - ttsStarveFrame <= 1;

        /// <summary>Diagnostic: frames an LLM decode ceded to a starving TTS.</summary>
        public static long LlmDeferrals;

        // Statics survive domain-reload-off replays; Time.frameCount resets to 0 each session,
        // so LAST session's frame marks would read as "this frame" (or overflow) — clear them.
        [RuntimeInitializeOnLoadMethod(RuntimeInitializeLoadType.SubsystemRegistration)]
        static void ResetOnBoot()
        {
            llmIssueFrame = int.MinValue;
            ttsStarveFrame = int.MinValue;
            TtsDeferrals = 0;
            LlmDeferrals = 0;
            lastCedeFrame = int.MinValue;   // after TtsDeferrals: its setter is what records this
        }
    }
}
