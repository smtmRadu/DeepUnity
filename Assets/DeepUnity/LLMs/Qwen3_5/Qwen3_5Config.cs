namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        public enum Qwen3_5LayerType
        {
            FullAttention,
            LinearAttention, // Gated DeltaNet
        }

        public static class Qwen3_5Config
        {
            // Vocab / tokens
            public static int
                VOCAB_SIZE = 248320,
                ENDOFTEXT_TOKEN_ID = 248044,    // <|endoftext|>  (pad)
                IM_START_TOKEN_ID = 248045,     // <|im_start|>
                IM_END_TOKEN_ID = 248046,       // <|im_end|>     (chat EOS)
                EOS_TOKEN_ID = 248046,          // alias for IM_END
                THINK_OPEN_TOKEN_ID = 248068,   // <think>
                THINK_CLOSE_TOKEN_ID = 248069,  // </think>
                IMAGE_TOKEN_ID = 248056,
                VIDEO_TOKEN_ID = 248057,
                VISION_START_TOKEN_ID = 248053,
                VISION_END_TOKEN_ID = 248054;

            // Model dims
            public static int
                HIDDEN_SIZE = 1024,
                MLP_INTERMEDIATE_SIZE = 3584,
                NUM_LAYERS = 24,
                MAX_POSITION_EMBEDDINGS = 262_144;

            // Full-attention layer
            public static int
                HEADS_Q = 8,
                HEADS_KV = 2,
                HEAD_DIM = 256;
            public static float
                ROPE_THETA = 10_000_000f,
                PARTIAL_ROTARY_FACTOR = 0.25f; // first 25% of head_dim is rotated
            public static int ROTATED_DIMS => (int)(HEAD_DIM * PARTIAL_ROTARY_FACTOR); // 64
            public static bool ATTENTION_BIAS = false;
            public static bool ATTN_OUTPUT_GATE = true;

            // Linear (Gated DeltaNet) layer
            public static int
                LINEAR_NUM_KEY_HEADS   = 16,
                LINEAR_NUM_VALUE_HEADS = 16,
                LINEAR_KEY_HEAD_DIM    = 128,
                LINEAR_VALUE_HEAD_DIM  = 128,
                LINEAR_CONV_KERNEL_DIM = 4;
            public static int LINEAR_KEY_DIM   => LINEAR_NUM_KEY_HEADS   * LINEAR_KEY_HEAD_DIM;   // 2048
            public static int LINEAR_VALUE_DIM => LINEAR_NUM_VALUE_HEADS * LINEAR_VALUE_HEAD_DIM; // 2048
            public static int LINEAR_CONV_DIM  => LINEAR_KEY_DIM * 2 + LINEAR_VALUE_DIM;          // 6144

            // Norm
            public static float RMS_EPS = 1e-6f;

            // Misc
            public static bool TIE_EMBEDDING = true;

            // DIAGNOSTIC: set false to bypass the KV cache (skips ResetCache zero-fill + the full-attention
            // cache write/read; forward runs with useCache:false). Use only to isolate boot/inference stalls
            // — multi-token generation output is NOT correct while this is off.
            public static bool USE_KV_CACHE = true;

            // 24 layers, full_attention every 4th. Pattern: L L L F  L L L F  ...
            public static Qwen3_5LayerType[] layer_types = new Qwen3_5LayerType[]
            {
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.FullAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.FullAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.FullAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.FullAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.FullAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.LinearAttention,
                Qwen3_5LayerType.FullAttention,
            };
        }
    }

    // Model-agnostic descriptor for Qwen3.5 — forwards to the static Qwen3_5Config above.
    // Sampling defaults are the "non-thinking text" preset (only presence_penalty differs from the base).
    public sealed class Qwen3_5ConfigDescriptor : LLMConfig
    {
        public override int HiddenSize            => Qwen3_5Modeling.Qwen3_5Config.HIDDEN_SIZE;
        public override int VocabSize             => Qwen3_5Modeling.Qwen3_5Config.VOCAB_SIZE;
        public override int NumLayers             => Qwen3_5Modeling.Qwen3_5Config.NUM_LAYERS;
        public override int MaxPositionEmbeddings => Qwen3_5Modeling.Qwen3_5Config.MAX_POSITION_EMBEDDINGS;
        public override int HeadDim               => Qwen3_5Modeling.Qwen3_5Config.HEAD_DIM;
        public override float RmsEps              => Qwen3_5Modeling.Qwen3_5Config.RMS_EPS;
        public override bool TieEmbedding         => Qwen3_5Modeling.Qwen3_5Config.TIE_EMBEDDING;

        public override int EosTokenId            => Qwen3_5Modeling.Qwen3_5Config.EOS_TOKEN_ID;
        public override int PadTokenId            => Qwen3_5Modeling.Qwen3_5Config.ENDOFTEXT_TOKEN_ID;
        public override int BosTokenId            => Qwen3_5Modeling.Qwen3_5Config.IM_START_TOKEN_ID; // Qwen has no classic BOS; <|im_start|> opens a sequence.

        public override float DefaultPresencePenalty => 2f;
    }
}

// Reference (config.json, text_config):
// {
//   "attn_output_gate": true,
//   "head_dim": 256,
//   "hidden_act": "silu",
//   "hidden_size": 1024,
//   "intermediate_size": 3584,
//   "num_attention_heads": 8,
//   "num_key_value_heads": 2,
//   "num_hidden_layers": 24,
//   "vocab_size": 248320,
//   "tie_word_embeddings": true,
//   "rms_norm_eps": 1e-6,
//   "linear_conv_kernel_dim": 4,
//   "linear_key_head_dim": 128,
//   "linear_num_key_heads": 16,
//   "linear_num_value_heads": 16,
//   "linear_value_head_dim": 128,
//   "rope_parameters": { "rope_theta": 10000000, "partial_rotary_factor": 0.25 }
// }
