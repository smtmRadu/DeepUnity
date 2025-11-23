namespace DeepUnity
{
    namespace MobileLLMModeling
    {
        public static class LlamaConfig
        {
            // silu
            public static int
                EOS_IDX = 2,
                BOS_IDX = 1,
                UNK_IDX = 0,
                VOCAB_SIZE = 32_000,
                HIDDEN_SIZE = 576,
                HEAD_DIM = 64,
                MLP_INTERMEDIATE_SIZE = 1536,


                NUM_LAYERS = 30,
                MAX_POSITION_EMBEDDINGS = 2048, // they say is 128K, the code says 32768.. fuck off.
                ROPE_BASE_FREQUENCY = 10_000,

                HEADS_Q = 9,
                HEADS_KV = 3;


            public static float
                RMS_EPS = 1e-5f,
                ATTN_EXPANSION_FACTOR = 1f;

            // 5:1 ratio
            public static bool
                TIE_EMBEDDING = true;
        }
    }
}

/*
 MobileLLMConfig {
  "architectures": [
    "MobileLLMForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "configuration_mobilellm.MobileLLMConfig",
    "AutoModelForCausalLM": "modeling_mobilellm.MobileLLMForCausalLM"
  },
  "bos_token_id": 1,
  "dtype": "float32",
  "eos_token_id": 2,
  "head_dim": 64,
  "hidden_act": "silu",
  "hidden_size": 576,
  "initializer_range": 0.02,
  "intermediate_size": 1536,
  "layer_sharing": false,
  "max_position_embeddings": 2048,
  "mlp_bias": false,
  "model_type": "mobilellm",
  "num_attention_heads": 9,
  "num_hidden_layers": 30,
  "num_key_value_heads": 3,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 10000.0,
  "share_embedding": true,
  "tie_word_embeddings": false,
  "transformers_version": "4.56.0",
  "use_cache": true,
  "vocab_size": 32000
}*/


/*
 MobileLLMForCausalLM(
  (model): MobileLLMModel(
    (embed_tokens): Embedding(32000, 576)
    (layers): ModuleList(
      (0-29): 30 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=576, out_features=576, bias=False)
          (k_proj): Linear(in_features=576, out_features=192, bias=False)
          (v_proj): Linear(in_features=576, out_features=192, bias=False)
          (o_proj): Linear(in_features=576, out_features=576, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=576, out_features=1536, bias=False)
          (up_proj): Linear(in_features=576, out_features=1536, bias=False)
          (down_proj): Linear(in_features=1536, out_features=576, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm()
        (post_attention_layernorm): LlamaRMSNorm()
      )
    )
    (norm): LlamaRMSNorm()
  )
)*/