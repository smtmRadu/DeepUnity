namespace DeepUnity
{
    /// <summary>
    /// Root of the tokenizer hierarchy (TokenizerBase -> BPETokenizer for text LLMs, plus the
    /// TTS/STT tokenizers). The one contract every tokenizer shares is readiness: text BPE
    /// tokenizers parse a large vocab asynchronously and flip <see cref="IsReady"/> true when the
    /// parse finishes, while the speech (TTS) and STT tokenizers load synchronously in their
    /// constructor and are ready immediately (the default).
    ///
    /// Encode/Decode signatures deliberately stay on the concrete classes — they diverge by
    /// modality (LLM text&lt;-&gt;ids, TTS text-&gt;speech ids, STT ids-&gt;text), so hoisting a
    /// shared method surface here would be a leaky abstraction. This base exists so game/loader
    /// code can hold and gate any tokenizer uniformly on <see cref="IsReady"/>.
    /// </summary>
    // Family root for every tokenizer (F2: TokenizerBase -> BPETokenizer / TTS / STT).
    public abstract class TokenizerBase
    {
        /// <summary>True once the tokenizer is usable. Synchronous (speech/STT) tokenizers are
        /// ready on construction (the default true); async vocab parsers override and flip this
        /// false→true via the protected setter when their background parse finishes.</summary>
        public virtual bool IsReady { get; protected set; } = true;
    }
}
