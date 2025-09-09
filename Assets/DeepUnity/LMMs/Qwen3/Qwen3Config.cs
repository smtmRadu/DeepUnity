namespace DeepUnity
{
    namespace Qwen3Modeling
    {
        public static class Qwen3Config
        {
            // Total params: 596,049,920
            // embedding layer: 151936 x 1024
            // lm head: 151936 x 1024
            // q_proj = 2048, 1024,
            // att_rms_norm = 128

            public static int
                PAD_IDX = 151_645,//151_645,
                VOCAB_SIZE = 151936,
                HIDDEN_SIZE = 1024,
                MLP_INTERMEDIATE_SIZE = 3072,


                ATTN_EXPANSION_FACTOR = 2,
                
                NUM_LAYERS = 18, 
                CONTEXT_LENGTH = 32768,
                ROPE_THETA = 1_000_000,
                HEADS_Q = 16,
                HEADS_KV = 8;
            public static float
                RMS_EPS = 1e-6f;

            public static bool
                TIE_EMBEDDING = true;

        }
    }
}