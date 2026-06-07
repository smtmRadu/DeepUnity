namespace DeepUnity
{
    /// <summary>
    /// Model-agnostic descriptor of an LLM's architecture and recommended sampling defaults.
    ///
    /// Each model keeps its own detailed static <c>*Config</c> class (the single source of truth used
    /// by the modeling/weights/cache code); the per-model descriptor below simply <b>forwards</b> to it,
    /// so this base never duplicates a value — it just exposes a uniform view (e.g. <c>llm.Config.VocabSize</c>)
    /// that future, model-agnostic code can read without knowing which model it's talking to.
    ///
    /// The sampling defaults are <c>virtual</c> with the "non-thinking text" preset baked in; a model whose
    /// recommended preset differs overrides only the properties it needs.
    /// </summary>
    public abstract class LLMConfig
    {
        // ---- Architecture ----
        public abstract int HiddenSize { get; }
        public abstract int VocabSize { get; }
        public abstract int NumLayers { get; }
        public abstract int MaxPositionEmbeddings { get; }
        public abstract int HeadDim { get; }
        public abstract float RmsEps { get; }
        public abstract bool TieEmbedding { get; }

        // ---- Special token ids ----
        public abstract int EosTokenId { get; }
        public abstract int PadTokenId { get; }
        public abstract int BosTokenId { get; }

        // ---- Sampling defaults (non-thinking text preset; override per model as needed) ----
        public virtual float DefaultTemperature       => 1f;
        public virtual int   DefaultTopK              => 20;
        public virtual float DefaultTopP              => 1f;
        public virtual float DefaultMinP              => 0f;
        public virtual float DefaultPresencePenalty   => 0f;
        public virtual float DefaultRepetitionPenalty => 1f;
    }
}
