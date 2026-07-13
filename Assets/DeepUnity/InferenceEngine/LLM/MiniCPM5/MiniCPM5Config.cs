namespace DeepUnity
{
    namespace MiniCPM5Modeling
    {
        // MiniCPM5-1B (openbmb/MiniCPM5-1B) — a vanilla llama-architecture decoder:
        // 24 identical full-attention GQA layers (no sliding window, no DeltaNet), 2 RMSNorms per
        // layer (input + post-attention/pre-MLP), full-head split-half RoPE, SiLU GLU MLP,
        // UNTIED embedding/lm_head (unlike Gemma3/Qwen3.5 — the lm_head is a separate matrix).
        // No qk-norm, no attention output gate, no logit softcapping, no MiniCPM mup scaling
        // (the MiniCPM5 checkpoints ship plain LlamaForCausalLM configs).
        public static class MiniCPM5Config
        {
            // Vocab / tokens (special_tokens_map.json + tokenizer.json added_tokens)
            public static int
                VOCAB_SIZE = 130560,
                BOS_TOKEN_ID = 0,            // <s>
                EOS_TOKEN_ID = 1,            // </s>  (also pad)
                ENDOFTEXT_TOKEN_ID = 1,      // pad alias (</s>), keeps parity with the other families
                IM_START_TOKEN_ID = 130072,  // <|im_start|>
                IM_END_TOKEN_ID = 130073;    // <|im_end|>  (chat EOS; generation_config eos = [1, 130073])

            // Model dims (config.json)
            public static int
                HIDDEN_SIZE = 1536,
                MLP_INTERMEDIATE_SIZE = 4608,
                NUM_LAYERS = 24,
                MAX_POSITION_EMBEDDINGS = 131_072,
                HEAD_DIM = 128,
                HEADS_Q = 16,
                HEADS_KV = 2;

            public static float
                RMS_EPS = 1e-6f,
                ROPE_THETA = 5_000_000f;

            public static bool TIE_EMBEDDING = false; // separate lm_head — costs an extra vocab*hidden fp16 matrix
        }
    }

    // Model-agnostic descriptor for MiniCPM5 — forwards to the static MiniCPM5Config above.
    // Sampling defaults mirror the checkpoint's generation_config (t=0.9, top_p=0.95).
    public sealed class MiniCPM5ConfigDescriptor : LLMConfig
    {
        public override int HiddenSize            => MiniCPM5Modeling.MiniCPM5Config.HIDDEN_SIZE;
        public override int VocabSize             => MiniCPM5Modeling.MiniCPM5Config.VOCAB_SIZE;
        public override int NumLayers             => MiniCPM5Modeling.MiniCPM5Config.NUM_LAYERS;
        public override int MaxPositionEmbeddings => MiniCPM5Modeling.MiniCPM5Config.MAX_POSITION_EMBEDDINGS;
        public override int HeadDim               => MiniCPM5Modeling.MiniCPM5Config.HEAD_DIM;
        public override float RmsEps              => MiniCPM5Modeling.MiniCPM5Config.RMS_EPS;
        public override bool TieEmbedding         => MiniCPM5Modeling.MiniCPM5Config.TIE_EMBEDDING;

        public override int EosTokenId            => MiniCPM5Modeling.MiniCPM5Config.EOS_TOKEN_ID;
        public override int PadTokenId            => MiniCPM5Modeling.MiniCPM5Config.EOS_TOKEN_ID;
        public override int BosTokenId            => MiniCPM5Modeling.MiniCPM5Config.BOS_TOKEN_ID;

        public override float DefaultTopP => 0.95f;
    }
}

// Reference (config.json):
// {
//   "architectures": ["LlamaForCausalLM"],
//   "model_type": "llama",
//   "hidden_size": 1536,
//   "intermediate_size": 4608,
//   "num_hidden_layers": 24,
//   "num_attention_heads": 16,
//   "num_key_value_heads": 2,
//   "head_dim": 128,
//   "vocab_size": 130560,
//   "max_position_embeddings": 131072,
//   "rope_theta": 5000000,
//   "rope_scaling": null,
//   "rms_norm_eps": 1e-06,
//   "tie_word_embeddings": false,
//   "hidden_act": "silu",
//   "attention_bias": false (implied — no bias tensors in the checkpoint)
// }
