using UnityEngine;

namespace DeepUnity
{
    // #29 cross-engine GPU frame arbiter.  When an NPC both GENERATES (LLM decode bursts) and
    // SPEAKS (streaming TTS ticks), the two engines used to issue their GPU bursts into the same
    // frame's queue — the 22-27 ms GEN+SPK+AUD band in the talk-perf report. The LLM has no
    // slack (each token's burst gates the next through a readback), while streaming TTS has
    // plenty (pocket-tts only needs ~12.5 latents/s and the ring buffers seconds ahead). So the
    // LLM marks every frame it issues GPU work, and TTS voices cede those frames whenever the
    // ring can afford it (see PocketTTSVoice.PumpPipeline).
    public static class FramePacing
    {
        static int llmIssueFrame = int.MinValue;

        /// <summary>LLM dispatch sites (prefill + decode) call this on every frame they issue GPU work.</summary>
        public static void NoteLlmIssue() => llmIssueFrame = Time.frameCount;

        /// <summary>True if the LLM issued GPU work this frame or the previous one. The 1-frame
        /// grace covers Update-vs-coroutine ordering: a voice's Update may run before the LLM
        /// coroutine resumes within the same frame, and while decoding the LLM issues (nearly)
        /// every frame — last frame's mark predicts this frame's burst.
        /// The sentinel guard matters: frameCount − int.MinValue OVERFLOWS negative, which read
        /// as "busy/starving since forever" at every session start — the un-guarded TtsStarving
        /// version held the LLM for the WHOLE first reply of each session (the "first message
        /// takes 5 s to speak" bug, 3166 held frames measured).</summary>
        public static bool LlmBusy => llmIssueFrame != int.MinValue && Time.frameCount - llmIssueFrame <= 1;

        /// <summary>Diagnostic: frames a TTS pump ceded to the LLM (#29 probe logs the per-turn delta).</summary>
        public static long TtsDeferrals;

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
        }
    }
}
