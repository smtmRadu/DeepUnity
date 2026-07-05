namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // Chatterbox-Turbo constants — mirror the reference implementation exactly.
        // Source of truth: Assets/DeepUnity/TTS/SPEC.md (extracted from resemble-ai/chatterbox +
        // the ResembleAI/chatterbox-turbo checkpoint).
        public static class ChatterboxConfig
        {
            // ---- T3 (GPT2-medium backbone) --------------------------------------------------
            public const int T3_HIDDEN = 1024;
            public const int T3_LAYERS = 24;
            public const int T3_HEADS = 16;              // MHA: heads_kv == heads_q
            public const int T3_HEAD_DIM = 64;
            public const int T3_QKV_DIM = 3 * T3_HIDDEN; // fused c_attn output
            public const int T3_MLP = 4096;
            public const int T3_TEXT_VOCAB = 50276;
            public const int T3_SPEECH_VOCAB = 6563;
            public const int T3_MAX_POSITIONS = 8196;    // wpe rows
            public const float T3_LN_EPS = 1e-5f;

            public const int START_SPEECH_TOKEN = 6561;
            public const int STOP_SPEECH_TOKEN = 6562;
            public const int SPEAKER_EMB_DIM = 256;
            public const int T3_COND_PROMPT_LEN = 375;   // baked-voice speech prompt tokens
            public const int T3_COND_LEN = 1 + T3_COND_PROMPT_LEN;  // + spkr_enc position

            // Turbo sampling defaults (tts_turbo.py generate())
            public const float DEFAULT_TEMPERATURE = 0.8f;
            public const int DEFAULT_TOP_K = 1000;
            public const float DEFAULT_TOP_P = 0.95f;
            public const float DEFAULT_REPETITION_PENALTY = 1.2f;
            public const int MAX_SPEECH_TOKENS = 1000;   // max_gen_len

            // ---- S3Gen flow ------------------------------------------------------------------
            public const int FLOW_VOCAB = 6561;          // speech tokens < this are valid for the flow
            public const int SIL_TOKEN = 4299;           // S3GEN_SIL, appended x3
            public const int ENC_DIM = 512;
            public const int ENC_HEADS = 8;
            public const int ENC_HEAD_DIM = 64;
            public const int ENC_FF = 2048;
            public const int ENC_LAYERS = 6;             // 25Hz encoders
            public const int ENC_UP_LAYERS = 4;          // 50Hz up_encoders
            public const int PRE_LOOKAHEAD = 3;
            public const float ENC_LN_EPS = 1e-12f;      // encoder-layer norms
            public const float EMBED_LN_EPS = 1e-5f;     // embed/up_embed/after_norm
            public const int XVECTOR_DIM = 192;
            public const int MEL_DIM = 80;
            public const int TOKEN_MEL_RATIO = 2;

            // ---- meanflow estimator ----------------------------------------------------------
            public const int EST_IN = 320;               // x|mu|spks|cond
            public const int EST_CH = 256;
            public const int EST_HEADS = 8;
            public const int EST_HEAD_DIM = 64;          // attn inner dim 512
            public const int EST_ATTN_INNER = 512;
            public const int EST_FF = 1024;
            public const int EST_MID_BLOCKS = 12;
            public const int EST_TFMR_PER_BLOCK = 4;
            public const int EST_TIME_DIM = 1024;        // time_embed_dim
            public const int EST_TIME_IN = 320;          // sinusoidal emb dim
            public const int CFM_TIMESTEPS = 2;          // meanflow: t_span [0, .5, 1] -> 2 estimator calls

            // ---- HiFTGenerator vocoder ---------------------------------------------------------
            public const int SAMPLE_RATE = 24000;
            public static readonly int[] UPSAMPLE_RATES = { 8, 5, 3 };
            public static readonly int[] UPSAMPLE_KERNELS = { 16, 11, 7 };
            public static readonly int[] RESBLOCK_KERNELS = { 3, 7, 11 };
            public static readonly int[] RESBLOCK_DILATIONS = { 1, 3, 5 };
            public static readonly int[] SOURCE_RESBLOCK_KERNELS = { 7, 7, 11 };
            public const int VOC_BASE_CH = 512;
            public const int NB_HARMONICS = 8;           // 9 sine rows incl. fundamental
            public const int ISTFT_NFFT = 16;
            public const int ISTFT_HOP = 4;
            public const int SAMPLES_PER_MEL_FRAME = 480; // 8*5*3*4
            public const float NSF_ALPHA = 0.1f;         // sine_amp
            public const float NSF_SIGMA = 0.003f;       // noise_std
            public const float NSF_VOICED_THRESHOLD = 10f;
            public const float AUDIO_LIMIT = 0.99f;
            public const float LRELU_SLOPE = 0.1f;       // upsample-loop leaky (conv_post pre-act uses 0.01)
        }
    }
}
