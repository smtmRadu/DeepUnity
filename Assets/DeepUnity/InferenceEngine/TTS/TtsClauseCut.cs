namespace DeepUnity
{
    /// <summary>
    /// Shared clause-cut scanner for the streamed-TTS voices (PocketTTS / Kokoro / CosyVoice —
    /// Chatterbox keeps its own soft-cut variant). Fed LLM text accumulates in each voice's
    /// pending buffer; this finds where to cut it into the next synthesized utterance.
    ///
    /// clausesPerChunk is the reply-pacing quality knob: the cut lands after the Nth sentence
    /// ender, so N sentences reach the TTS as ONE utterance and the prosody flows naturally
    /// across their boundaries instead of resetting per sentence. 1 (default) = today's
    /// lowest-latency behavior: every finished sentence speaks immediately.
    ///
    /// Details:
    ///  - an ender RUN ("...", "?!") counts as ONE ender, cut at its end — otherwise an
    ///    ellipsis alone would satisfy clausesPerChunk=3;
    ///  - a run touching the buffer end doesn't count yet (the next LLM delta may extend it,
    ///    e.g. "." arriving as the first dot of "..."); the voice's FlushText covers reply end;
    ///  - a comma past emergencyChunkChars cuts immediately regardless of the count — it is
    ///    the run-on-sentence escape hatch, not the normal path.
    /// </summary>
    internal static class TtsClauseCut
    {
        internal static bool IsEnder(char c) => c == '.' || c == '!' || c == '?' || c == ';' || c == '\n';

        /// <summary>Index to cut at (inclusive), or -1 when the buffer holds no complete chunk yet.</summary>
        internal static int FindCut(string s, int clausesPerChunk, int emergencyChunkChars)
        {
            int need = clausesPerChunk < 1 ? 1 : clausesPerChunk;
            int enders = 0;
            for (int i = 0; i < s.Length; i++)
            {
                char c = s[i];
                if (c == ',' && i >= emergencyChunkChars) return i;   // run-on escape hatch: cut NOW
                if (!IsEnder(c)) continue;
                if (i + 1 >= s.Length || IsEnder(s[i + 1])) continue;  // mid-run, or run at buffer end
                if (++enders >= need) return i;
            }
            return -1;
        }
    }
}
