namespace DeepUnity
{
    namespace Gemma3Modelling
    {
        public enum GemmaLayerType
        {
            FullAttention,
            SlidingWindowAttention,
        }
        public static class Gemma3Config
        {
            public static int? ATTN_LOGIT_SOFTCAPPING = null;
            public static float? ROPE_SCALING = null;
            public static int
                PAD_IDX = 0,
                EOS_IDX = 1,
                BOS_IDX = 2,
                VOCAB_SIZE = 262144,
                HIDDEN_SIZE = 640,
                MLP_INTERMEDIATE_SIZE = 2048,
                NUM_LAYERS = 18,
                MAX_POSITION_EMBEDDINGS = 32_768, // they say is 128K, the code says 32768.. fuck off.
                ROPE_LOCAL_BASE_FREQUENCY = 10_000,
                ROPE_THETA = 1_000_000,
                HEAD_DIM = 256,
                HEADS_Q = 4,
                HEADS_KV = 1,
                SLIDING_WINDOW = 512;

            public static float
                RMS_EPS = 1e-6f,
                QUERY_PRE_ATTENTION_SCALAR = 256,
                ATTN_EXPANSION_FACTOR = 1.6f;

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
            }; 
            public static bool
                TIE_EMBEDDING = true;
        }
    }
}

// Total params = 268 098 176
/*
 {'training': False,
 '_parameters': {},
 '_buffers': {},
 '_non_persistent_buffers_set': set(),
 '_backward_pre_hooks': OrderedDict(),
 '_backward_hooks': OrderedDict(),
 '_is_full_backward_hook': None,
 '_forward_hooks': OrderedDict(),
 '_forward_hooks_with_kwargs': OrderedDict(),
 '_forward_hooks_always_called': OrderedDict(),
 '_forward_pre_hooks': OrderedDict(),
 '_forward_pre_hooks_with_kwargs': OrderedDict(),
 '_state_dict_hooks': OrderedDict(),
 '_state_dict_pre_hooks': OrderedDict(),
 '_load_state_dict_pre_hooks': OrderedDict(),
 '_load_state_dict_post_hooks': OrderedDict(),
 '_modules': {'model': Gemma3TextModel(
    (embed_tokens): Gemma3TextScaledWordEmbedding(262144, 640, padding_idx=0)
    (layers): ModuleList(
      (0-17): 18 x Gemma3DecoderLayer(
        (self_attn): Gemma3Attention(
          (q_proj): Linear(in_features=640, out_features=1024, bias=False)
          (k_proj): Linear(in_features=640, out_features=256, bias=False)
          (v_proj): Linear(in_features=640, out_features=256, bias=False)
          (o_proj): Linear(in_features=1024, out_features=640, bias=False)
          (q_norm): Gemma3RMSNorm((256,), eps=1e-06)
          (k_norm): Gemma3RMSNorm((256,), eps=1e-06)
        )
        (mlp): Gemma3MLP(
          (gate_proj): Linear(in_features=640, out_features=2048, bias=False)
          (up_proj): Linear(in_features=640, out_features=2048, bias=False)
          (down_proj): Linear(in_features=2048, out_features=640, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (post_attention_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (pre_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
        (post_feedforward_layernorm): Gemma3RMSNorm((640,), eps=1e-06)
      )
    )
    (norm): Gemma3RMSNorm((640,), eps=1e-06)
    (rotary_emb): Gemma3RotaryEmbedding()
    (rotary_emb_local): Gemma3RotaryEmbedding()
  ),
  'lm_head': Linear(in_features=640, out_features=262144, bias=False)},
 'config': Gemma3TextConfig {
   "_sliding_window_pattern": 6,
   "architectures": [
     "Gemma3ForCausalLM"
   ],
   "attention_bias": false,
   "attention_dropout": 0.0,
   "attn_logit_softcapping": null,
   "bos_token_id": 2,
   "eos_token_id": 1,
   "final_logit_softcapping": null,
   "head_dim": 256,
   "hidden_activation": "gelu_pytorch_tanh",
   "hidden_size": 640,
   "initializer_range": 0.02,
   "intermediate_size": 2048,
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
     "full_attention"
   ],
   "max_position_embeddings": 32768,
   "model_type": "gemma3_text",
   "num_attention_heads": 4,
   "num_hidden_layers": 18,
   "num_key_value_heads": 1,
   "pad_token_id": 0,
   "query_pre_attn_scalar": 256,
   "rms_norm_eps": 1e-06,
   "rope_local_base_freq": 10000.0,
   "rope_scaling": null,
   "rope_theta": 1000000.0,
   "sliding_window": 512,
   "torch_dtype": "float32",
   "transformers_version": "4.55.4",
   "use_bidirectional_attention": false,
   "use_cache": true,
   "vocab_size": 262144
 },
 'loss_type': 'ForCausalLM',
 'name_or_path': 'google/gemma-3-270m',
 'warnings_issued': {},
 'generation_config': GenerationConfig {
   "cache_implementation": "hybrid",
   "do_sample": true,
   "top_k": 64,
   "top_p": 0.95
 },
 '_keep_in_fp32_modules': None,
 '_keep_in_fp32_modules_strict': None,
 '_no_split_modules': ['Gemma3DecoderLayer',
  'SiglipVisionEmbeddings',
  'SiglipEncoderLayer',
  'SiglipMultiheadAttentionPoolingHead'],
 'vocab_size': 262144,
 '_pp_plan': {'embed_tokens': (['input_ids'], ['inputs_embeds']),
  'layers': (['hidden_states', 'attention_mask'], ['hidden_states']),
  'norm': (['hidden_states'], ['hidden_states'])},
 '_is_hf_initialized': True}
 
 */