using UnityEngine;

namespace DeepUnity
{
    // #29 cross-engine GPU frame arbiter. When an NPC both GENERATES (LLM decode bursts) and
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
        /// every frame — last frame's mark predicts this frame's burst.</summary>
        public static bool LlmBusy => Time.frameCount - llmIssueFrame <= 1;

        /// <summary>Diagnostic: frames a TTS pump ceded to the LLM (#29 probe logs the per-turn delta).</summary>
        public static long TtsDeferrals;
    }
}
