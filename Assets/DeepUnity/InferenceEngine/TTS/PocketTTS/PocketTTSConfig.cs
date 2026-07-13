namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        // Frozen constants for the Kyutai pocket-tts port (english voice-cloning variant).
        // Source of truth: config/english.yaml + the safetensors shapes (see ../SPEC.md).
        // Everything the C# needs to size buffers and dispatch kernels without re-reading the manifest.
        public static class PocketTTSConfig
        {
            // ---- audio / framing ----
            public const int SAMPLE_RATE = 24000;
            public const float FRAME_RATE = 12.5f;                 // latent frames per second
            public const int SAMPLES_PER_LATENT = 1920;            // SAMPLE_RATE / FRAME_RATE
            public const float ENCODER_FRAME_RATE = 200f;          // SAMPLE_RATE / HOP (24000/120)
            public const int MIMI_STEPS_PER_LATENT = 16;           // ENCODER_FRAME_RATE / FRAME_RATE

            // ---- FlowLM transformer backbone ----
            public const int DIM = 1024;                           // d_model
            public const int TF_LAYERS = 6;
            public const int TF_HEADS = 16;
            public const int HEAD_DIM = 64;                        // DIM / TF_HEADS
            public const int TF_FFN = 4096;                        // hidden_scale 4
            public const float ROPE_THETA = 10000f;                // max_period
            public const int TEXT_VOCAB = 4001;                    // conditioner.embed rows (n_bins 4000 + 1)

            // ---- FlowLM latent + flow head (SimpleMLPAdaLN) ----
            public const int LDIM = 32;                            // Mimi inner_dim = flow latent dim
            public const int FLOW_DIM = 512;                       // flow_net width
            public const int FLOW_DEPTH = 6;                       // res_blocks
            public const int LSD_DECODE_STEPS = 1;                 // Euler steps per frame
            public const float TEMPERATURE = 0.7f;
            public const float EOS_THRESHOLD = -4.0f;              // logprob

            // ---- Mimi codec ----
            public const int MIMI_SEANET_DIM = 512;                // seanet.dimension
            public const int MIMI_N_FILTERS = 64;
            public static readonly int[] MIMI_RATIOS = { 6, 5, 4 }; // decoder upsample ratios (hop 120)
            public const int MIMI_N_RESIDUAL_LAYERS = 1;
            public const int MIMI_KERNEL_SIZE = 7;                 // first conv
            public const int MIMI_LAST_KERNEL_SIZE = 3;            // final conv -> 1ch
            public const int MIMI_RESIDUAL_KERNEL_SIZE = 3;
            public const int MIMI_DILATION_BASE = 2;
            public const int MIMI_COMPRESS = 2;                    // resblock hidden = dim/compress
            // pad_mode = "constant" (zero left-pad) — causal.

            // ---- DummyQuantizer (latent 32 -> 512, Conv1d k1 no bias) ----
            public const int QUANT_IN = 32;                        // = LDIM
            public const int QUANT_OUT = 512;                      // = outer_dim = MIMI_SEANET_DIM

            // ---- Mimi decoder_transformer (ProjectedTransformer) ----
            public const int MIMI_TF_DIM = 512;
            public const int MIMI_TF_LAYERS = 2;
            public const int MIMI_TF_HEADS = 8;
            public const int MIMI_TF_HEAD_DIM = 64;                // MIMI_TF_DIM / MIMI_TF_HEADS
            public const int MIMI_TF_FFN = 2048;                   // dim_feedforward
            public const float MIMI_TF_LAYER_SCALE = 0.01f;
            public const int MIMI_TF_CONTEXT = 250;                // causal attention window (frames)
            public const float MIMI_TF_ROPE_THETA = 10000f;

            // ---- upsample (ConvTrUpsample1d): stride 16, groups 512, kernel 32, no bias ----
            public const int UPSAMPLE_STRIDE = 16;                 // MIMI_STEPS_PER_LATENT
            public const int UPSAMPLE_KERNEL = 32;                 // 2 * stride
            public const int UPSAMPLE_GROUPS = 512;                // depthwise

            // ---- windowed streaming decode (P5.3, long-utterance fix) ----
            // Left-context latents for a windowed Mimi decode. Total left receptive field of the
            // decoder is ~34.5 latents: 2 transformer layers x 250-frame window @200Hz = 31.25
            // + upsample ~2 + SEANet ~1. Measured: CTX>=36 plateaus at wav maxabs ~3e-5 (pure fp
            // rounding from RoPE absolute-phase — rotary attention is mathematically relative, so
            // restarting positions inside a window cancels), corr 1.00000000. 40 adds margin.
            public const int MIMI_DECODE_CTX = 40;

            public const string WEIGHTS_DIR_FP16 = "Assets/Resources/Weights/weights_pockettts_english_fp16";
            public const string WEIGHTS_DIR_INT8 = "Assets/Resources/Weights/weights_pockettts_english_int8";
        }
    }
}
