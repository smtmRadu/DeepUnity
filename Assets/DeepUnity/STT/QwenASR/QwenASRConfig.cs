// Pure C# (no UnityEngine) — shared by the Unity runtime AND the net8.0 parity harness
// (validation/harness). Keep it dependency-free.
namespace DeepUnity
{
    public enum QwenASRSize { B0_6, B1_7 }

    namespace QwenASRModeling
    {
        // Frozen from Qwen/Qwen3-ASR-{0.6B,1.7B}-hf config.json (see SPEC.md §0-§4).
        // Mutating statics matches the one-model-per-session design used by Qwen3_5Config.
        public static class QwenASRConfig
        {
            // ---- tokens (tokenizer.json, verified) ----
            public const int VOCAB_SIZE = 151936;
            public const int ENDOFTEXT_TOKEN_ID = 151643;   // <|endoftext|>  (eos2)
            public const int IM_START_TOKEN_ID = 151644;    // <|im_start|>
            public const int IM_END_TOKEN_ID = 151645;      // <|im_end|>     (eos1, pad)
            public const int AUDIO_START_TOKEN_ID = 151669; // <|audio_start|>
            public const int AUDIO_END_TOKEN_ID = 151670;   // <|audio_end|>
            public const int AUDIO_PAD_TOKEN_ID = 151676;   // <|audio_pad|>  (audio placeholder)
            public const int ASR_TEXT_TOKEN_ID = 151704;    // <asr_text>

            // ---- mel frontend (feature_extraction_qwen3_asr.py) ----
            public const int SAMPLE_RATE = 16000;
            public const int N_FFT = 400;
            public const int HOP = 160;
            public const int N_MELS = 128;
            public const int N_FREQS = 201;                 // 1 + N_FFT/2
            public const int MIN_SAMPLES = 8000;            // clips shorter are zero-padded (no mask fix)
            public const int MEL_CHUNK = 100;               // 2*n_window; mel padded to a multiple

            // ---- audio encoder (audio_config; size-dependent set by ApplySize; defaults 0.6B) ----
            public static int ENC_D_MODEL = 896;
            public static int ENC_LAYERS = 18;
            public static int ENC_HEADS = 14;               // head_dim = d/heads = 64 both sizes
            public static int ENC_FFN = 3584;
            public const int ENC_CONV_CH = 480;             // downsample_hidden_size
            public const int ENC_CONV_FLAT = 7680;          // 480 * 16 freq bins after 3x stride-2
            public const int ENC_POS_LEN = 13;              // tokens per 100-frame chunk (13 Hz)
            public const int ENC_WINDOW_TOKENS = 104;       // 13 * (n_window_infer 800 / 100) = 8 s
            public const float ENC_LN_EPS = 1e-5f;          // nn.LayerNorm default

            // ---- decoder = stock Qwen3 (text_config; size-dependent set by ApplySize) ----
            public static int HIDDEN_SIZE = 1024;
            public static int MLP_INTERMEDIATE_SIZE = 3072;
            public const int NUM_LAYERS = 28;
            public const int HEADS_Q = 16;
            public const int HEADS_KV = 8;
            public const int HEAD_DIM = 128;                // full-dim RoPE (no partial factor)
            public const float ROPE_THETA = 1_000_000f;
            public const float RMS_EPS = 1e-6f;
            public const int MAX_POSITION_EMBEDDINGS = 65536;
            public const bool TIE_EMBEDDING = true;         // no lm_head tensor in the checkpoint

            public static void ApplySize(QwenASRSize size)
            {
                switch (size)
                {
                    case QwenASRSize.B0_6:
                        ENC_D_MODEL = 896;  ENC_LAYERS = 18; ENC_HEADS = 14; ENC_FFN = 3584;
                        HIDDEN_SIZE = 1024; MLP_INTERMEDIATE_SIZE = 3072;
                        break;
                    case QwenASRSize.B1_7:
                        ENC_D_MODEL = 1024; ENC_LAYERS = 24; ENC_HEADS = 16; ENC_FFN = 4096;
                        HIDDEN_SIZE = 2048; MLP_INTERMEDIATE_SIZE = 6144;
                        break;
                }
            }

            public static string SizeLabel(QwenASRSize size) => size == QwenASRSize.B1_7 ? "1.7b" : "0.6b";

            /// <summary>Post-CNN length of a partial mel chunk: l → floor((l−1)/2)+1 three times
            /// (0 stays 0). NB Python floor-division semantics — handled explicitly (SPEC §2.1).</summary>
            public static int Ceil3(int l)
            {
                for (int i = 0; i < 3; i++)
                    l = l <= 0 ? 0 : (l - 1) / 2 + 1;
                return l;
            }

            /// <summary>Audio token count for a clip with <paramref name="validMelFrames"/> valid mel
            /// frames: 13 per full 100-frame chunk + Ceil3(remainder). Validated against the HF
            /// processor in the D0 reference dumps (meta.json expected_audio_tokens_formula).</summary>
            public static int AudioTokenCount(int validMelFrames)
                => (validMelFrames / MEL_CHUNK) * ENC_POS_LEN + Ceil3(validMelFrames % MEL_CHUNK);
        }
    }
}
