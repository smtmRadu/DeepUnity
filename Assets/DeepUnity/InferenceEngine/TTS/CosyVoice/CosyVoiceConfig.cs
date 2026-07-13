namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // Fun-CosyVoice3-0.5B-2512 constants — mirror the reference implementation exactly.
        // Source of truth: Assets/DeepUnity/InferenceEngine/TTS/CosyVoice/SPEC.md, frozen from the released
        // cosyvoice3.yaml + FunAudioLLM/CosyVoice sources + CosyVoice-BlankEN/config.json.
        // Anything marked VERIFY-AT-EXPORT is confirmed against checkpoint tensor shapes by
        // validation/import in A0 before first use.
        public static class CosyVoiceConfig
        {
            // ---- Global ----------------------------------------------------------------------
            public const int SAMPLE_RATE = 24000;
            public const int TOKEN_FRAME_RATE = 25;       // speech tokens per second
            public const int TOKEN_MEL_RATIO = 2;         // 1 token -> 2 mel frames (mel @ 50Hz)
            public const int MEL_DIM = 80;
            public const int SAMPLES_PER_MEL_FRAME = 480; // hop 480 @ 24kHz (= 8*5*3*4)
            public const int CHUNK_TOKENS = 25;           // streaming chunk (1s of audio)
            public const int CHUNK_MEL = CHUNK_TOKENS * TOKEN_MEL_RATIO; // DiT static_chunk_size = 50

            // ---- LM: CosyVoice3LM = Qwen2.5-0.5B backbone + speech head ------------------------
            public const int LM_HIDDEN = 896;
            public const int LM_LAYERS = 24;
            public const int LM_HEADS_Q = 14;
            public const int LM_HEADS_KV = 2;             // GQA 7:1
            public const int LM_HEAD_DIM = 64;            // 896/14
            public const int LM_MLP = 4864;               // SiLU gate/up/down
            public const int LM_TEXT_VOCAB = 151936;      // Qwen2.5 embed_tokens (tied lm head UNUSED here)
            public const float LM_ROPE_THETA = 1000000f;  // FULL RoPE over all 64 dims (no partial factor)
            public const float LM_RMS_EPS = 1e-6f;
            // Qwen2 attention uses bias on q/k/v projections (not o_proj) — VERIFY-AT-EXPORT.

            // Speech-side vocab: FSQ codebook 6561 (=3^8) + 200 extra rows.
            public const int SPEECH_VOCAB = 6561;         // flow-valid tokens are < this
            public const int SPEECH_EMB_ROWS = SPEECH_VOCAB + 200;   // speech_embedding + llm_decoder rows (6761)
            public const int SOS_TOKEN = SPEECH_VOCAB + 0;           // 6561 (embedded via speech_embedding)
            public const int EOS_TOKEN = SPEECH_VOCAB + 1;           // 6562
            public const int TASK_ID_TOKEN = SPEECH_VOCAB + 2;       // 6563
            public const int FILL_TOKEN = SPEECH_VOCAB + 3;          // 6564 (bistream mode only)
            // stop condition: sampled id >= SPEECH_VOCAB (all 200 tail ids are stop ids)

            public const int ENDOFPROMPT_TEXT_ID = 151646;           // <|endofprompt|> in Qwen text vocab

            // RAS sampling (ras_sampling in utils/common.py) — reference defaults
            public const float RAS_TOP_P = 0.8f;
            public const int RAS_TOP_K = 25;
            public const int RAS_WIN_SIZE = 10;           // resample randomly if candidate repeated
            public const float RAS_TAU_R = 0.1f;          //   >= win*tau (=1) times in last 10 tokens
            public const float MIN_TOKEN_TEXT_RATIO = 2f; // EOS suppressed below minLen = 2*textLen
            public const float MAX_TOKEN_TEXT_RATIO = 20f;

            // ---- Flow: CausalMaskedDiffWithDiT -------------------------------------------------
            public const int FLOW_TOKEN_EMB_DIM = 80;     // input_embedding: Embedding(6561, 80)
            public const int SPK_EMBED_DIM = 192;         // campplus x-vector (L2-normalized)
            public const int SPK_PROJ_DIM = 80;           // spk_embed_affine_layer: Linear(192->80)
            public const int PRE_LOOKAHEAD_LEN = 3;       // tokens of future context
            public const int PRE_LOOKAHEAD_CH = 1024;     // conv1 k=4 (80->1024), leaky; conv2 k=3 causal (1024->80); +residual

            // DiT estimator (flow/DiT/dit.py)
            public const int DIT_DIM = 1024;
            public const int DIT_DEPTH = 22;
            public const int DIT_HEADS = 16;
            public const int DIT_HEAD_DIM = 64;           // inner 1024, full MHA (bias=true on q/k/v/out)
            public const int DIT_FF = 2048;               // ff_mult 2, GELU(tanh)
            public const int DIT_IN_CONCAT = 4 * MEL_DIM; // input_embed cat[x, cond, mu, spk] = 320 -> Linear(320->1024)
            public const int DIT_CONVPOS_KERNEL = 31;     // CausalConvPositionEmbedding: 2x (conv k31 g16 leftpad30 + Mish), +residual
            public const int DIT_CONVPOS_GROUPS = 16;
            public const int DIT_TIME_FREQ_DIM = 256;     // SinusPositionEmbedding(scale=1000) -> MLP 256->1024->1024 (SiLU)
            public const float DIT_LN_EPS = 1e-6f;        // all LayerNorms affine-free
            // AdaLN-Zero per block: SiLU(t) -> Linear(1024->6144) -> shift/scale/gate (msa) + shift/scale/gate (mlp)
            // Final: SiLU(t) -> Linear(1024->2048) -> scale/shift; proj_out Linear(1024->80)
            // RoPE: x_transformers RotaryEmbedding(dim_head=64), theta 10000 — VERIFY-AT-PARITY

            // CFM solver (flow_matching.py)
            public const int CFM_TIMESTEPS = 10;          // Euler steps (n_timesteps=10 at inference)
            public const float CFM_SIGMA_MIN = 1e-6f;
            public const float CFM_INFERENCE_CFG_RATE = 0.7f; // dxdt = 1.7*cond - 0.7*uncond (batch-2 estimator)
            // t_scheduler 'cosine': t = 1 - cos(t * pi/2) over linspace(0,1,11)
            // Noise: FIXED buffer randn(1,80,15000) @ torch seed 0 — exported as a weight for parity.
            public const int FIXED_NOISE_FRAMES = 15000;  // 50*300 (~5 min cap)
            public const int FLOW_CACHE_OVERLAP = 34;     // z/mu cache: prompt frames + last 34 frames of prev chunk

            // ---- A6-max streaming (single-pass causal flow + windowed vocoder) -----------------
            // DiT K/V cache cap in mel frames (incl. prompt): 10 steps x 22 layers x 2 CFG halves
            // x 2 (K,V) x cap x 1024 fp16 ~= 1.8 MB/frame. 768 frames ~= 1.38 GB and covers ~12 s
            // of generated audio per clause; longer utterances fall back to the full re-solve.
            public const int FLOW_STREAM_KV_MAX_FRAMES = 768;
            // Windowed re-vocode left context. The HiFT main branch's total causal receptive
            // field is ~22 mel frames (resblocks k11 d5 dominate); 64 leaves ~3x margin so the
            // window's left-edge zero-pad artifacts sit fully outside the emitted band.
            // (Raised 40 -> 64 after the fast path measured a 0.045 boundary jump vs 0.015 legacy.)
            public const int VOC_STREAM_OVERLAP_MEL = 64;
            // Streaming seam cross-fade: each non-finalize chunk holds back this many samples
            // (20 ms) and the next chunk raised-cosine-blends its own recomputation over them,
            // so the first emitted sample of every chunk continues the PREVIOUS vocode exactly —
            // kills residual seam discontinuities regardless of their source (fp16-KV mel deltas,
            // window edges). Applies only to the windowed streaming path.
            public const int VOC_STREAM_FADE = 480;

            // ---- Vocoder: CausalHiFTGenerator ---------------------------------------------------
            public static readonly int[] UPSAMPLE_RATES = { 8, 5, 3 };
            public static readonly int[] UPSAMPLE_KERNELS = { 16, 11, 7 };
            public static readonly int[] RESBLOCK_KERNELS = { 3, 7, 11 };
            public static readonly int[] RESBLOCK_DILATIONS = { 1, 3, 5 };
            public static readonly int[] SOURCE_RESBLOCK_KERNELS = { 7, 7, 11 };
            public const int VOC_BASE_CH = 512;
            public const int NB_HARMONICS = 8;
            public const int ISTFT_NFFT = 16;
            public const int ISTFT_HOP = 4;
            public const float NSF_ALPHA = 0.1f;
            public const float NSF_SIGMA = 0.003f;
            public const float NSF_VOICED_THRESHOLD = 10f;
            public const float AUDIO_LIMIT = 0.99f;
            public const float LRELU_SLOPE = 0.1f;
            public const int CONV_PRE_LOOK_RIGHT = 4;     // causal variant's bounded right-context — VERIFY-AT-EXPORT
            // f0_predictor: CausalConvRNNF0Predictor (in 80, cond 512) — structure read at A1 from generator source
        }
    }
}
