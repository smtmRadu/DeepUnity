namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        
        public static class EmbeddingGemmaConfig
        {
            public static int? ATTN_LOGIT_SOFTCAPPING = null;
            public static float? ROPE_SCALING = null;
            public static int
                PAD_IDX = 0,
                EOS_IDX = 1,
                BOS_IDX = 2,
                VOCAB_SIZE = 262144,
                HIDDEN_SIZE = 768,
                MLP_INTERMEDIATE_SIZE = 1152,
                HEAD_FFN_INTERMEDIATE_SIZE = 3072,
                NUM_LAYERS = 24,
                MAX_POSITION_EMBEDDINGS = 2_048,
                ROPE_LOCAL_BASE_FREQUENCY = 10_000,
                ROPE_THETA = 1_000_000,
                HEAD_DIM = 256,
                HEADS_Q = 3,
                HEADS_KV = 1,
                SLIDING_WINDOW = 512;

            public static float
                RMS_EPS = 1e-6f,
                QUERY_PRE_ATTENTION_SCALAR = 256,
                ATTN_EXPANSION_FACTOR = 1f;

            public static bool
                TIE_EMBEDDING = true,
                USE_BIDIRECTIONAL_ATTENTION = true;
            // 5:1 ratio
            public static GemmaLayerType[] layer_types = new GemmaLayerType[]
            {
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.FullAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.FullAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.FullAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.FullAttention,
            };

        }
    }
}

// Total params = 268 098 176
/*Gemma3TextConfig {
  "_sliding_window_pattern": 6,
  "architectures": [
    "Gemma3TextModel"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": null,
  "bos_token_id": 2,
  "dtype": "float32",
  "eos_token_id": 1,
  "final_logit_softcapping": null,
  "head_dim": 256,
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 1152,
  "layer_types": [
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "sliding_attention",
    "full_attention"
  ],
  "max_position_embeddings": 2048,
  "model_type": "gemma3_text",
  "num_attention_heads": 3,
  "num_hidden_layers": 24,
  "num_key_value_heads": 1,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_local_base_freq": 10000.0,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": 512,
  "transformers_version": "4.56.0",
  "use_bidirectional_attention": true,
  "use_cache": true,
  "vocab_size": 262144
}*/