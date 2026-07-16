using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace KokoroModeling
    {
        // Kokoro-82M GPU forward (KokoroCS.compute), mirroring KokoroCPU.Forward stage-by-stage —
        // the oracle's call sequence IS this dispatch list. fp16 weights / fp32 activations.
        //
        // GPU/CPU split (SPEC §9): everything conv/attention-shaped runs on GPU; the 6 biLSTMs,
        // the duration head + alignment build, and the NSF phase pipeline (+ its RNG) run on
        // KokoroCPU via worker Tasks at three handoff points:
        //   GPU PLBERT+benc -> readback d_en [T,512] -> CPU DurationEncoder/head/shared-LSTM
        //     -> upload xf [512,F] -> GPU F0/N stacks -> readback F0 [2F] (NSF input)
        //   GPU TextEncoder convs -> readback [512,T] -> CPU tenc biLSTM -> upload t_en [512,T]
        //   CPU NsfHar (overlaps the GPU decoder) -> upload har [S] -> GPU STFT/generator/iSTFT.
        // FastKernels2 (#33) reorders the SCHEDULE only (identical math): the tenc convs are
        // issued BEFORE PLBERT and their readback requested early, so the tenc biLSTM runs
        // concurrently with the predictor's CPU stage instead of serially after the F0/N
        // stacks; the F0 readback is requested before the N stack is issued (earlier NSF
        // start). BertMs therefore includes the (cheap) tenc dispatches under v2.
        //
        // Layouts match KokoroCPU exactly: PLBERT buffers are [T,C] row-major, everything
        // conv-side is [C,T] channel-major (channel concat = contiguous CopySlice).
        //
        // Scratch is grow-only; the generator buffers dominate (6 × ~[128, 120F+1] floats — for a
        // 20 s chunk ≈ 45 MB each). Chunks are bounded by the 510-phoneme packer; F above ~1050
        // (~26 s) would exceed the 65535-threadgroup dispatch limit of the flat kernels and is
        // warned about (never produced by KokoroTTS.Chunk).
        //
        // NOTE for the orchestrator: KokoroCS must be registered in DeepUnityMeta.cs at merge
        // (shared file — not edited by this workstream); until then the shader is loaded via
        // Resources.Load here.
        public class KokoroModel : IDisposable
        {
            readonly ComputeShader cs;
            readonly KokoroWeights weights;
            readonly KokoroCPU cpu;      // CPU-side stage host (biLSTMs, duration, NSF) + oracle

            int kEmbAlbert, kEmbText, kLinear, kLinearQ8, kLN, kAttn, kAct, kConv, kConvT, kIN,
                kGather, kSnake, kStft, kIstft, kCopySlice, kAdd, kAddScale, kScale;
            int kLinearT2, kLinearT2Q8, kConvTile, kConvTFast, kINStats;
            int kConvTile2, kLNCoop;

            /// <summary>Routes the optimized kernels (LinearTileBias2[Q8], Conv1DTile with the
            /// fused AdaIN+Snake/LeakyReLU X-prologue, InstanceNormStats, ConvTranspose1DFast).
            /// Static so probes can bisect old-vs-new without replumbing: false = the original
            /// per-output kernels and unfused block sequences, byte-for-byte the
            /// pre-optimization dispatch list. Both paths keep the oracle's per-output float
            /// accumulation order (differences are fma-contraction noise, ~1e-6).</summary>
            public static bool FastKernels = true;

            /// <summary>#33 deep-opt layer on top of FastKernels (no effect while FastKernels is
            /// false). Routes Conv1DTile2 (4t x 4oc register blocking + fused residual/AddScale
            /// writeback), LayerNormCoop (group-per-position LN, bert residual adds fused),
            /// activation prologues on ConvT/conv_post/tenc, and the reordered pipeline
            /// (TextEncoder branch issued BEFORE PLBERT so its readback + CPU biLSTM overlap
            /// the predictor's CPU stage; F0 readback requested before the N stack). Per-output
            /// conv accumulation order is unchanged (bit-comparable to v1 modulo fma noise);
            /// LayerNormCoop tree-reduces (reordered sums, gated at maxabs &lt; 1e-3 like the
            /// #31 GEMVs). Probes bisect three ways: v2 / v1 (this off) / legacy (both off).</summary>
            public static bool FastKernels2 = true;

            // persistent small buffers
            ComputeBuffer idsBuf, styleSdBuf, styleSpBuf, styleG1Buf, styleG2Buf;
            ComputeBuffer statsBuf;      // InstanceNormStats out [2C], C <= 1090 (fused AdaIN)
            // PLBERT / TextEncoder scratch (sized by T <= 512, allocated once)
            ComputeBuffer bA, bB, bC, qB, kB, vB, atB, ffB, embA, embB, dEnBuf, teA, teB, tEnBuf;
            // predictor / decoder scratch (sized by F, grow-only)
            ComputeBuffer xfBuf, fA, fB, fT1, fT2, fT3, F0Buf, NBuf, f0cBuf, ncBuf;
            ComputeBuffer asrBuf, asrResBuf, decCat, decA, dT1, dT2, dT3;
            ComputeBuffer gatherBuf;     // uint time indices (frame2tok / reflection pad)
            // generator scratch (sized by F, the big ones)
            ComputeBuffer gX, gSrc, gT1, gT2, gAcc, gTmp, harBuf, harCatBuf, wavBuf;

            int curF;
            const float RSQRT2 = 0.70710678f;

            public bool IsReady => weights.IsReady;

            // ---- parity hooks (validation/KokoroKernelProbe) ---------------------------------
            /// <summary>Override the predicted durations with the reference dump (CHECKLIST B2-D
            /// provision: keeps downstream shapes comparable when rounding differs).</summary>
            public int[] InjectPredDur;
            /// <summary>When true, ForwardYielding fills LastStages with per-stage readbacks in
            /// KokoroCPU.Stages layout (probe only — costs sync readbacks).</summary>
            public bool CaptureStages;
            public KokoroCPU.Stages LastStages;

            /// <summary>Wall-clock stage timings of the last forward (ms). EndToEnd includes the
            /// final wav readback = true ids-in -> samples-out latency. PredCpuMs/TencCpuMs are
            /// the CPU biLSTM worker-task waits inside PredictorMs; NsfWaitMs is how long the
            /// generator stalled on the CPU NSF source (0 when it fully overlapped the decoder).</summary>
            public float BertMs, PredictorMs, DecoderMs, GeneratorMs, EndToEndMs;
            public float PredCpuMs, TencCpuMs, NsfWaitMs;

            public KokoroModel(KokoroWeights weights, KokoroCPU cpu)
            {
                this.weights = weights;
                this.cpu = cpu;
                cs = Resources.Load<ComputeShader>("ComputeShaders/KokoroCS");
                if (cs == null)
                {
                    ConsoleMessage.Error("Compute Shader 'KokoroCS' was not found in Resources/ComputeShaders.");
                    return;
                }

                kEmbAlbert = cs.FindKernel("EmbedAlbert");
                kEmbText = cs.FindKernel("EmbedText");
                kLinear = cs.FindKernel("LinearBias");
                kLinearQ8 = cs.FindKernel("LinearBiasQ8");
                kLN = cs.FindKernel("LayerNormAffine");
                kAttn = cs.FindKernel("AttentionBi");
                kAct = cs.FindKernel("Activate");
                kConv = cs.FindKernel("Conv1D");
                kConvT = cs.FindKernel("ConvTranspose1D");
                kIN = cs.FindKernel("InstanceNormStyle");
                kGather = cs.FindKernel("GatherTime");
                kSnake = cs.FindKernel("Snake");
                kStft = cs.FindKernel("Stft20");
                kIstft = cs.FindKernel("Istft20");
                kCopySlice = cs.FindKernel("CopySlice");
                kAdd = cs.FindKernel("AddBuf");
                kAddScale = cs.FindKernel("AddScale");
                kScale = cs.FindKernel("ScaleBuf");
                kLinearT2 = cs.FindKernel("LinearTileBias2");
                kLinearT2Q8 = cs.FindKernel("LinearTileBias2Q8");
                kConvTile = cs.FindKernel("Conv1DTile");
                kConvTFast = cs.FindKernel("ConvTranspose1DFast");
                kINStats = cs.FindKernel("InstanceNormStats");
                kConvTile2 = cs.FindKernel("Conv1DTile2");
                kLNCoop = cs.FindKernel("LayerNormCoop");

                const int T = 512;                        // ALBERT max_position_embeddings
                idsBuf = New(T);
                gatherBuf = New(1);                       // grown by EnsureFrameScratch
                styleSdBuf = New(128); styleSpBuf = New(128);
                styleG1Buf = New(2 * 1090); styleG2Buf = New(2 * 1090);
                statsBuf = New(2 * 1090);
                bA = New(T * 768); bB = New(T * 768); bC = New(T * 768);
                qB = New(T * 768); kB = New(T * 768); vB = New(T * 768); atB = New(T * 768);
                ffB = New(T * 2048); embA = New(T * 128); embB = New(T * 128);
                dEnBuf = New(T * 512); teA = New(512 * T); teB = New(512 * T); tEnBuf = New(512 * T);
            }

            static ComputeBuffer New(int count) => new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            static int Div256(int n) => (n + 255) / 256;

            static void Grow(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = New(count);
            }

            void EnsureFrameScratch(int F)
            {
                if (F <= curF) return;
                curF = F;
                if (120 * F + 1 > 65535 * 256 / 128)
                    ConsoleMessage.Warning($"Kokoro chunk too long (F={F}): generator dispatches " +
                                           "exceed the 65535-threadgroup limit. Shorten the chunk.");
                int F2 = 2 * F, frames = 120 * F + 1;
                Grow(ref xfBuf, 512 * F);
                Grow(ref fA, 512 * F2); Grow(ref fB, 512 * F2);
                Grow(ref fT1, 512 * F2); Grow(ref fT2, 512 * F2); Grow(ref fT3, 512 * F2);
                Grow(ref F0Buf, F2); Grow(ref NBuf, F2); Grow(ref f0cBuf, F); Grow(ref ncBuf, F);
                Grow(ref asrBuf, 512 * F); Grow(ref asrResBuf, 64 * F);
                Grow(ref decCat, 1090 * F); Grow(ref decA, 1024 * F);
                Grow(ref dT1, 1090 * F2); Grow(ref dT2, 1090 * F2); Grow(ref dT3, 1090 * F2);
                Grow(ref gatherBuf, frames);
                int gen = 128 * frames;                    // >= 256*20F and 512*2F
                Grow(ref gX, gen); Grow(ref gSrc, gen); Grow(ref gT1, gen); Grow(ref gT2, gen);
                Grow(ref gAcc, gen); Grow(ref gTmp, gen);
                Grow(ref harBuf, 600 * F); Grow(ref harCatBuf, 22 * frames); Grow(ref wavBuf, 600 * F);
            }

            // ================= generic op helpers (public: graded by KokoroKernelProbe) =========
            /// <summary>Upload token ids for EmbedAlbert/EmbedText (probe hook; ForwardYielding
            /// calls it itself).</summary>
            public void UploadIds(int[] ids)
            {
                uint[] u = new uint[ids.Length];
                for (int i = 0; i < ids.Length; i++) u[i] = (uint)ids[i];
                idsBuf.SetData(u, 0, 0, ids.Length);
            }

            public void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int I, int O,
                               int act = 0, bool bias = true, float leaky = 0f)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", I); cs.SetInt("out_dim", O);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetFloat("leaky_slope", leaky);
                // per-TENSOR quant pick: int8 exports carry a ".w.scales" sibling (INT8_NOTES.md)
                bool q8 = weights.Has(name + ".w.scales");
                // tiled GEMM for the PLBERT/benc matmuls; the tile kernels need in_dim % 32 == 0
                // and only pay off with multiple tokens (StyleFc T=1 stays on the naive kernel)
                bool tile = FastKernels && T >= 8 && (I & 31) == 0;
                int k = q8 ? (tile ? kLinearT2Q8 : kLinearQ8) : (tile ? kLinearT2 : kLinear);
                cs.SetBuffer(k, "X", x);
                cs.SetBuffer(k, "W", weights.Get(name + ".w"));
                cs.SetBuffer(k, "W_bias", weights.Get(bias ? name + ".b" : name + ".w"));
                cs.SetBuffer(k, "Y", y);
                if (q8) cs.SetBuffer(k, "W_scales", weights.Get(name + ".w.scales"));
                if (tile) cs.Dispatch(k, (O + 63) / 64, (T + 31) / 32, 1);
                else cs.Dispatch(k, 1, (T + 7) / 8, (O + 31) / 32);
            }

            /// <summary>AdaIN style FC: [gamma|beta] = fc(s), Linear(128 -> 2C). Returns 2C.</summary>
            public int StyleFc(string name, ComputeBuffer styleVec, ComputeBuffer gbOut)
            {
                int O = weights.Shape(name + ".b")[0];
                Linear(name, styleVec, gbOut, 1, 128, O);
                return O;
            }

            /// <summary>Strided LayerNorm: bert rows [T,C] -> (posStride C, chStride 1);
            /// tenc channels of [C,T] -> (posStride 1, chStride T). eps 1e-12 / 1e-5.
            /// FastKernels2 routes the cooperative group-per-position kernel (the thread-per-
            /// position v1 left the GPU nearly idle at bert/tenc chunk sizes).</summary>
            public void LayerNorm(string name, ComputeBuffer x, ComputeBuffer y, int positions,
                                  int C, float eps, int posStride, int chStride)
            {
                cs.SetInt("seq_len", positions); cs.SetInt("norm_dim", C);
                cs.SetFloat("norm_eps", eps);
                cs.SetInt("ln_pos_stride", posStride); cs.SetInt("ln_ch_stride", chStride);
                int k = FastKernels && FastKernels2 ? kLNCoop : kLN;
                cs.SetBuffer(k, "norm_input", x);
                cs.SetBuffer(k, "norm_output", y);
                cs.SetBuffer(k, "ln_gamma", weights.Get(name + ".w"));
                cs.SetBuffer(k, "ln_beta", weights.Get(name + ".b"));
                if (k == kLNCoop)
                {
                    cs.SetInt("ln_add", 0);
                    cs.SetBuffer(k, "buf_b", x);              // never read at ln_add 0
                    cs.Dispatch(k, positions, 1, 1);
                }
                else
                    cs.Dispatch(k, Div256(positions), 1, 1);
            }

            /// <summary>FastKernels2-only: LayerNormCoop with the preceding residual AddBuf
            /// fused (out = LN(x + res)) — the two per-layer bert residual sites. Same float
            /// add as the standalone AddBuf; x/res are left unmutated.</summary>
            public void LayerNormAdd(string name, ComputeBuffer x, ComputeBuffer res,
                                     ComputeBuffer y, int positions, int C, float eps,
                                     int posStride, int chStride)
            {
                cs.SetInt("seq_len", positions); cs.SetInt("norm_dim", C);
                cs.SetFloat("norm_eps", eps);
                cs.SetInt("ln_pos_stride", posStride); cs.SetInt("ln_ch_stride", chStride);
                cs.SetInt("ln_add", 1);
                cs.SetBuffer(kLNCoop, "norm_input", x);
                cs.SetBuffer(kLNCoop, "buf_b", res);
                cs.SetBuffer(kLNCoop, "norm_output", y);
                cs.SetBuffer(kLNCoop, "ln_gamma", weights.Get(name + ".w"));
                cs.SetBuffer(kLNCoop, "ln_beta", weights.Get(name + ".b"));
                cs.Dispatch(kLNCoop, positions, 1, 1);
            }

            public void EmbedAlbert(ComputeBuffer y, int T)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", 128);
                cs.SetBuffer(kEmbAlbert, "token_ids", idsBuf);
                cs.SetBuffer(kEmbAlbert, "emb_word", weights.Get("bert/emb/word.w"));
                cs.SetBuffer(kEmbAlbert, "emb_pos", weights.Get("bert/emb/pos.w"));
                cs.SetBuffer(kEmbAlbert, "emb_tok", weights.Get("bert/emb/tok.w"));
                cs.SetBuffer(kEmbAlbert, "Y", y);
                cs.Dispatch(kEmbAlbert, Div256(T * 128), 1, 1);
            }

            public void EmbedText(ComputeBuffer y, int T)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", 512);
                cs.SetBuffer(kEmbText, "token_ids", idsBuf);
                cs.SetBuffer(kEmbText, "emb_word", weights.Get("tenc/embedding.w"));
                cs.SetBuffer(kEmbText, "Y", y);
                cs.Dispatch(kEmbText, Div256(T * 512), 1, 1);
            }

            public void AttentionBi(ComputeBuffer q, ComputeBuffer k, ComputeBuffer v,
                                    ComputeBuffer o, int T, int heads, int headDim)
            {
                cs.SetInt("seq_len", T); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", headDim);
                cs.SetFloat("attn_scale", 1f / Mathf.Sqrt(headDim));
                cs.SetBuffer(kAttn, "Q", q); cs.SetBuffer(kAttn, "K", k); cs.SetBuffer(kAttn, "V", v);
                cs.SetBuffer(kAttn, "AttendedValues", o);
                cs.Dispatch(kAttn, T, heads, 1);
            }

            /// <summary>In-place activation. act: 1 gelu_new, 2 leaky (slope!), 3 tanh.</summary>
            public void Activate(ComputeBuffer buf, int count, int act, float leaky = 0f)
            {
                cs.SetInt("buffer_size", count); cs.SetInt("activation_type", act);
                cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kAct, "inout_buf", buf);
                cs.Dispatch(kAct, Div256(count), 1, 1);
            }

            /// <summary>Channel-major Conv1d [Cin,Tin] -> [Cout,Tout]. FastKernels routes
            /// stride-1 convs with K &lt;= 11, (K-1)*dil &lt;= 50 and Cout &gt;= 16 to the
            /// register-blocked tile kernel (Conv1DTile2 under FastKernels2, else Conv1DTile;
            /// raw-X prologue); the rest (strided F0/N_conv, noise_conv0 k12s6, 1-channel
            /// projs) keep the naive per-output kernel. inMode 3 (FastKernels2 callers only)
            /// fuses a producer LeakyReLU(inSlope) into the X load.</summary>
            public void Conv(string name, ComputeBuffer x, ComputeBuffer y, int Cin, int Tin,
                             int Cout, int Tout, int k, int stride, int pad, int dil, bool bias = true,
                             int inMode = 0, float inSlope = 0f)
            {
                cs.SetInt("seq_len", Tout); cs.SetInt("in_len", Tin);
                cs.SetInt("in_dim", Cin); cs.SetInt("out_dim", Cout);
                cs.SetInt("conv_kernel", k); cs.SetInt("conv_stride", stride);
                cs.SetInt("conv_dilation", dil); cs.SetInt("pad_left", pad);
                cs.SetInt("has_bias", bias ? 1 : 0);
                bool tile = FastKernels && stride == 1 && k <= 11 && (k - 1) * dil <= 50 && Cout >= 16;
                bool v2 = tile && FastKernels2;
                int kk = v2 ? kConvTile2 : tile ? kConvTile : kConv;
                cs.SetBuffer(kk, "X", x);
                cs.SetBuffer(kk, "W", weights.Get(name + ".w"));
                cs.SetBuffer(kk, "W_bias", weights.Get(bias ? name + ".b" : name + ".w"));
                cs.SetBuffer(kk, "Y", y);
                if (tile)
                {
                    cs.SetInt("conv_in_mode", inMode);     // 0 raw (never-read prologue slots)
                    if (inMode == 3) cs.SetFloat("leaky_slope", inSlope);
                    cs.SetBuffer(kk, "in_stats", statsBuf);
                    cs.SetBuffer(kk, "style_gb", statsBuf);
                    cs.SetBuffer(kk, "snake_alpha", weights.Get(name + ".w"));
                    if (v2)
                    {
                        cs.SetInt("conv_out_mode", 0);
                        cs.SetBuffer(kk, "buf_b", statsBuf);    // never read at out_mode 0
                        cs.Dispatch(kk, (Tout + 127) / 128, (Cout + 31) / 32, 1);
                    }
                    else
                        cs.Dispatch(kk, (Tout + 127) / 128, (Cout + 15) / 16, 1);
                }
                else
                    cs.Dispatch(kk, Div256(Tout * Cout), 1, 1);
            }

            /// <summary>FastKernels-only: Conv1DTile with the producing AdaIN block's ops fused
            /// into the X load — mode 1 = AdaIN+Snake (Generator resblocks, alphaName set),
            /// mode 2 = AdaIN+LeakyReLU 0.2 (AdainResBlk1d). Requires InStats(x) in statsBuf
            /// and the style fc output in styleGB; x itself is read-only (no copy needed).</summary>
            void ConvFused(string name, ComputeBuffer x, ComputeBuffer y, int Cin, int T, int Cout,
                           int k, int pad, int dil, int mode, ComputeBuffer styleGB, string alphaName)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_len", T);
                cs.SetInt("in_dim", Cin); cs.SetInt("out_dim", Cout);
                cs.SetInt("conv_kernel", k); cs.SetInt("conv_stride", 1);
                cs.SetInt("conv_dilation", dil); cs.SetInt("pad_left", pad);
                cs.SetInt("has_bias", 1);
                cs.SetInt("conv_in_mode", mode); cs.SetFloat("leaky_slope", 0.2f);
                cs.SetBuffer(kConvTile, "X", x);
                cs.SetBuffer(kConvTile, "W", weights.Get(name + ".w"));
                cs.SetBuffer(kConvTile, "W_bias", weights.Get(name + ".b"));
                cs.SetBuffer(kConvTile, "Y", y);
                cs.SetBuffer(kConvTile, "in_stats", statsBuf);
                cs.SetBuffer(kConvTile, "style_gb", styleGB);
                cs.SetBuffer(kConvTile, "snake_alpha",
                             weights.Get(alphaName ?? name + ".w"));   // mode 2: never read
                cs.Dispatch(kConvTile, (T + 127) / 128, (Cout + 15) / 16, 1);
            }

            /// <summary>FastKernels2-only: Conv1DTile2 with producer ops fused into the X load
            /// (inMode 1 AdaIN+Snake / 2 AdaIN+LeakyReLU 0.2 / 3 plain LeakyReLU(slope)) and
            /// the CONSUMER's elementwise op fused into the writeback: outMode 1 = Y += conv
            /// (the SnakeResBlock residual AddBuf), outMode 2 = Y = (conv + addBuf)*outScale
            /// (the AdainBlock (res+short)*rsqrt2 AddScale). Same float expressions and
            /// per-element order as the standalone dispatches they replace. inMode 1/2 need
            /// InStats(x) in statsBuf and the style fc output in styleGB.</summary>
            void ConvFused2(string name, ComputeBuffer x, ComputeBuffer y, int Cin, int T, int Cout,
                            int k, int pad, int dil, int inMode, ComputeBuffer styleGB,
                            string alphaName, int outMode = 0, ComputeBuffer addBuf = null,
                            float outScale = 1f, float slope = 0.2f)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_len", T);
                cs.SetInt("in_dim", Cin); cs.SetInt("out_dim", Cout);
                cs.SetInt("conv_kernel", k); cs.SetInt("conv_stride", 1);
                cs.SetInt("conv_dilation", dil); cs.SetInt("pad_left", pad);
                cs.SetInt("has_bias", 1);
                cs.SetInt("conv_in_mode", inMode); cs.SetFloat("leaky_slope", slope);
                cs.SetInt("conv_out_mode", outMode); cs.SetFloat("scale_val", outScale);
                cs.SetBuffer(kConvTile2, "X", x);
                cs.SetBuffer(kConvTile2, "W", weights.Get(name + ".w"));
                cs.SetBuffer(kConvTile2, "W_bias", weights.Get(name + ".b"));
                cs.SetBuffer(kConvTile2, "Y", y);
                cs.SetBuffer(kConvTile2, "in_stats", statsBuf);
                cs.SetBuffer(kConvTile2, "style_gb", styleGB ?? statsBuf);
                cs.SetBuffer(kConvTile2, "snake_alpha",
                             weights.Get(alphaName ?? name + ".w"));   // mode 2/3: never read
                cs.SetBuffer(kConvTile2, "buf_b", addBuf ?? statsBuf); // out_mode < 2: never read
                cs.Dispatch(kConvTile2, (T + 127) / 128, (Cout + 31) / 32, 1);
            }

            /// <summary>FastKernels-only: per-channel InstanceNorm mean/rstd of x [C,T] into
            /// statsBuf [2C] (same reduction/eps as InstanceNormStyle; the affine + activation
            /// happen in the consuming ConvFused prologue).</summary>
            void InStats(ComputeBuffer x, int C, int T)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", C); cs.SetFloat("norm_eps", 1e-5f);
                cs.SetBuffer(kINStats, "norm_input", x);
                cs.SetBuffer(kINStats, "norm_output", statsBuf);
                cs.Dispatch(kINStats, C, 1, 1);
            }

            /// <summary>Channel-major ConvTranspose1d; groups=Cin for the AdainBlock depthwise
            /// pool (k3 s2 p1 outPad1 -> Tout=2Tin), groups=1 for the Generator ups.
            /// FastKernels uses the residue-major mapping (warp lanes share the stride
            /// residue -> no k-loop divergence, coalesced X) — bit-exact same math.
            /// inMode (FastKernels2 callers only): 2 = AdaIN+LeakyReLU 0.2 on each X read
            /// (needs InStats in statsBuf + styleGB), 3 = plain LeakyReLU(inSlope).</summary>
            public void ConvT(string name, ComputeBuffer x, ComputeBuffer y, int Cin, int Tin,
                              int Cout, int Tout, int k, int stride, int pad, int groups,
                              int inMode = 0, float inSlope = 0f, ComputeBuffer styleGB = null)
            {
                cs.SetInt("seq_len", Tout); cs.SetInt("in_len", Tin);
                cs.SetInt("in_dim", Cin); cs.SetInt("out_dim", Cout);
                cs.SetInt("conv_kernel", k); cs.SetInt("conv_stride", stride);
                cs.SetInt("pad_left", pad); cs.SetInt("conv_groups", groups);
                cs.SetInt("has_bias", 1);
                int kk = FastKernels ? kConvTFast : kConvT;
                cs.SetBuffer(kk, "X", x);
                cs.SetBuffer(kk, "W", weights.Get(name + ".w"));
                cs.SetBuffer(kk, "W_bias", weights.Get(name + ".b"));
                cs.SetBuffer(kk, "Y", y);
                if (FastKernels)
                {
                    cs.SetInt("conv_in_mode", inMode);     // ALWAYS set (stale-uniform guard)
                    if (inMode != 0) cs.SetFloat("leaky_slope", inMode == 2 ? 0.2f : inSlope);
                    cs.SetBuffer(kk, "in_stats", statsBuf);          // mode 0/3: never read
                    cs.SetBuffer(kk, "style_gb", styleGB ?? statsBuf);
                    cs.Dispatch(kk, Div256(Cout * ((Tout + stride - 1) / stride) * stride), 1, 1);
                }
                else
                    cs.Dispatch(kk, Div256(Tout * Cout), 1, 1);
            }

            /// <summary>AdaIN: per-channel InstanceNorm over time + (1+g)x+b from style_gb [2C].
            /// In-place on [C,T]. eps 1e-5.</summary>
            public void InstanceNormStyle(ComputeBuffer x, ComputeBuffer styleGB, int C, int T)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", C); cs.SetFloat("norm_eps", 1e-5f);
                cs.SetBuffer(kIN, "inout_buf", x);
                cs.SetBuffer(kIN, "style_gb", styleGB);
                cs.Dispatch(kIN, C, 1, 1);
            }

            /// <summary>Nearest x2 time upsample (AdainBlock shortcut), [C,Tin] -> [C,2Tin].</summary>
            public void GatherUp2(ComputeBuffer x, ComputeBuffer y, int C, int Tin, int Tout)
            {
                cs.SetInt("seq_len", Tout); cs.SetInt("in_len", Tin); cs.SetInt("in_dim", C);
                cs.SetInt("gather_mode", 1);
                cs.SetBuffer(kGather, "X", x); cs.SetBuffer(kGather, "Y", y);
                cs.SetBuffer(kGather, "gather_idx", gatherBuf);   // unused in mode 1
                cs.Dispatch(kGather, Div256(Tout * C), 1, 1);
            }

            /// <summary>Index-buffer time gather (frame2tok aln expand / reflection pad).</summary>
            public void GatherIdx(ComputeBuffer x, ComputeBuffer y, ComputeBuffer idx,
                                  int C, int Tin, int Tout)
            {
                cs.SetInt("seq_len", Tout); cs.SetInt("in_len", Tin); cs.SetInt("in_dim", C);
                cs.SetInt("gather_mode", 0);
                cs.SetBuffer(kGather, "X", x); cs.SetBuffer(kGather, "Y", y);
                cs.SetBuffer(kGather, "gather_idx", idx);
                cs.Dispatch(kGather, Div256(Tout * C), 1, 1);
            }

            /// <summary>Snake x + sin^2(ax)/a, per-channel alpha, in-place on [C,T].</summary>
            public void SnakeAct(string alphaName, ComputeBuffer buf, int C, int T)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", C);
                cs.SetBuffer(kSnake, "inout_buf", buf);
                cs.SetBuffer(kSnake, "snake_alpha", weights.Get(alphaName));
                cs.Dispatch(kSnake, Div256(T * C), 1, 1);
            }

            /// <summary>STFT n_fft 20 hop 5: har [S] -> har_cat [22, frames] (mag;angle).</summary>
            public void Stft(ComputeBuffer har, ComputeBuffer harCat, int S, int frames)
            {
                cs.SetInt("sample_len", S); cs.SetInt("n_frames", frames);
                cs.SetBuffer(kStft, "X", har); cs.SetBuffer(kStft, "Y", harCat);
                cs.Dispatch(kStft, Div256(frames * 11), 1, 1);
            }

            /// <summary>iSTFT head: conv_post out [22, frames] -> wav [(frames-1)*5].</summary>
            public void Istft(ComputeBuffer spec, ComputeBuffer wav, int frames, int Sout)
            {
                cs.SetInt("sample_len", Sout); cs.SetInt("n_frames", frames);
                cs.SetBuffer(kIstft, "X", spec); cs.SetBuffer(kIstft, "Y", wav);
                cs.Dispatch(kIstft, Div256(Sout), 1, 1);
            }

            public void CopySliceOp(ComputeBuffer dst, int dstOff, ComputeBuffer src, int srcOff, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetInt("copy_dst_offset", dstOff); cs.SetInt("copy_src_offset", srcOff);
                cs.SetBuffer(kCopySlice, "buf_a", dst); cs.SetBuffer(kCopySlice, "buf_b", src);
                cs.Dispatch(kCopySlice, Div256(count), 1, 1);
            }

            public void AddOp(ComputeBuffer dstA, ComputeBuffer srcB, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kAdd, "buf_a", dstA); cs.SetBuffer(kAdd, "buf_b", srcB);
                cs.Dispatch(kAdd, Div256(count), 1, 1);
            }

            public void AddScaleOp(ComputeBuffer dstA, ComputeBuffer srcB, int count, float s)
            {
                cs.SetInt("buffer_size", count); cs.SetFloat("scale_val", s);
                cs.SetBuffer(kAddScale, "buf_a", dstA); cs.SetBuffer(kAddScale, "buf_b", srcB);
                cs.Dispatch(kAddScale, Div256(count), 1, 1);
            }

            public void ScaleOp(ComputeBuffer buf, int count, float s)
            {
                cs.SetInt("buffer_size", count); cs.SetFloat("scale_val", s);
                cs.SetBuffer(kScale, "inout_buf", buf);
                cs.Dispatch(kScale, Div256(count), 1, 1);
            }

            // ================= composite blocks (KokoroCPU.AdainBlock / SnakeResBlock) ==========
            /// <summary>AdainResBlk1d on [Cin,T] -> outBuf [Cout, up?2T:T]; returns (Cout,Tout).
            /// Scratch t1..t3 each hold >= Cin*(up?2T:T) floats; x is left intact.</summary>
            // ---------------- frame-slicing budget ------------------------------------------
            // In play mode one frame must never absorb a whole stage of dispatches: Tick()
            // accumulates the approximate MACs issued since the last yield and the pipeline
            // yields whenever a slice's worth has been queued. Editor probes pump MoveNext()
            // in a tight loop, so the extra yield points cost them nothing.
            /// <summary>Approx MACs of GPU work per frame-slice (~2-3 ms on a 4060). Lower =
            /// smoother but more frames per clause; long.MaxValue = never split.</summary>
            public long SliceMacs = 250_000_000;
            long macAcc;
            bool Tick(long macs) { macAcc += macs; if (macAcc < SliceMacs) return false; macAcc = 0; return true; }

            public IEnumerator AdainBlockY(string p, ComputeBuffer x, ComputeBuffer outBuf, int Cin,
                                           int T, bool up, ComputeBuffer styleVec,
                                           ComputeBuffer t1, ComputeBuffer t2, ComputeBuffer t3,
                                           Action<int, int> dims = null)
            {
                int Cout = weights.Shape(p + "/conv1.w")[0];
                int Tr = up ? 2 * T : T;
                if (FastKernels && FastKernels2 && !up)
                {
                    // v2: v1's fused prologues + the (res+short)*rsqrt2 AddScale fused into
                    // conv2's writeback (outMode 2) — the shortcut is computed BEFORE conv2
                    StyleFc(p + "/norm1_fc", styleVec, styleG1Buf);
                    InStats(x, Cin, T);
                    ConvFused2(p + "/conv1", x, t3, Cin, T, Cout, 3, 1, 1, 2, styleG1Buf, null);
                    if (Tick(3L * Cin * Cout * Tr)) yield return null;
                    ComputeBuffer scf = x;
                    if (weights.Has(p + "/conv1x1.w"))
                    {
                        Conv(p + "/conv1x1", x, t1, Cin, Tr, Cout, Tr, 1, 1, 0, 1, bias: false);
                        if (Tick((long)Cin * Cout * Tr)) yield return null;
                        scf = t1;
                    }
                    StyleFc(p + "/norm2_fc", styleVec, styleG2Buf);
                    InStats(t3, Cout, Tr);
                    ConvFused2(p + "/conv2", t3, outBuf, Cout, Tr, Cout, 3, 1, 1, 2, styleG2Buf,
                               null, outMode: 2, addBuf: scf, outScale: RSQRT2);
                    if (Tick(3L * Cout * Cout * Tr)) yield return null;
                    dims?.Invoke(Cout, Tr);
                    yield break;
                }
                if (FastKernels && FastKernels2)   // v2 up: AdaIN1+lrelu fused into the pool's
                {                                  // X read; shortcut fused into conv2's writeback
                    StyleFc(p + "/norm1_fc", styleVec, styleG1Buf);
                    InStats(x, Cin, T);
                    ConvT(p + "/pool", x, t2, Cin, T, Cin, Tr, 3, 2, 1, Cin,
                          inMode: 2, styleGB: styleG1Buf);              // depthwise
                    Conv(p + "/conv1", t2, t3, Cin, Tr, Cout, Tr, 3, 1, 1, 1);
                    if (Tick(3L * Cin * Cout * Tr)) yield return null;
                    GatherUp2(x, t1, Cin, T, Tr);                       // t2 free after conv1
                    ComputeBuffer scf = t1;
                    if (weights.Has(p + "/conv1x1.w"))
                    {
                        Conv(p + "/conv1x1", t1, t2, Cin, Tr, Cout, Tr, 1, 1, 0, 1, bias: false);
                        if (Tick((long)Cin * Cout * Tr)) yield return null;
                        scf = t2;
                    }
                    StyleFc(p + "/norm2_fc", styleVec, styleG2Buf);
                    InStats(t3, Cout, Tr);
                    ConvFused2(p + "/conv2", t3, outBuf, Cout, Tr, Cout, 3, 1, 1, 2, styleG2Buf,
                               null, outMode: 2, addBuf: scf, outScale: RSQRT2);
                    if (Tick(3L * Cout * Cout * Tr)) yield return null;
                    dims?.Invoke(Cout, Tr);
                    yield break;
                }
                if (FastKernels && !up)
                {
                    // fused path: AdaIN affine + lrelu live in the ConvFused X prologue, fed by
                    // InstanceNormStats — x is never copied or mutated (shortcut reads it raw)
                    StyleFc(p + "/norm1_fc", styleVec, styleG1Buf);
                    InStats(x, Cin, T);
                    ConvFused(p + "/conv1", x, t3, Cin, T, Cout, 3, 1, 1, 2, styleG1Buf, null);
                    if (Tick(3L * Cin * Cout * Tr)) yield return null;
                    StyleFc(p + "/norm2_fc", styleVec, styleG2Buf);
                    InStats(t3, Cout, Tr);
                    ConvFused(p + "/conv2", t3, outBuf, Cout, Tr, Cout, 3, 1, 1, 2, styleG2Buf, null);
                    if (Tick(3L * Cout * Cout * Tr)) yield return null;
                    ComputeBuffer scf = x;
                    if (weights.Has(p + "/conv1x1.w"))
                    {
                        Conv(p + "/conv1x1", x, t3, Cin, Tr, Cout, Tr, 1, 1, 0, 1, bias: false);
                        if (Tick((long)Cin * Cout * Tr)) yield return null;
                        scf = t3;
                    }
                    AddScaleOp(outBuf, scf, Cout * Tr, RSQRT2);                  // (res+short)/sqrt2
                    dims?.Invoke(Cout, Tr);
                    yield break;
                }
                if (FastKernels)   // up: pool consumes the activation -> only conv2 fuses
                {
                    CopySliceOp(t1, 0, x, 0, Cin * T);
                    StyleFc(p + "/norm1_fc", styleVec, styleG1Buf);
                    InstanceNormStyle(t1, styleG1Buf, Cin, T);
                    Activate(t1, Cin * T, 2, 0.2f);
                    ConvT(p + "/pool", t1, t2, Cin, T, Cin, Tr, 3, 2, 1, Cin);   // depthwise
                    Conv(p + "/conv1", t2, t3, Cin, Tr, Cout, Tr, 3, 1, 1, 1);
                    if (Tick(3L * Cin * Cout * Tr)) yield return null;
                    StyleFc(p + "/norm2_fc", styleVec, styleG2Buf);
                    InStats(t3, Cout, Tr);
                    ConvFused(p + "/conv2", t3, outBuf, Cout, Tr, Cout, 3, 1, 1, 2, styleG2Buf, null);
                    if (Tick(3L * Cout * Cout * Tr)) yield return null;
                    GatherUp2(x, t1, Cin, T, Tr);                                // t1 free by now
                    ComputeBuffer scf = t1;
                    if (weights.Has(p + "/conv1x1.w"))
                    {
                        Conv(p + "/conv1x1", t1, t3, Cin, Tr, Cout, Tr, 1, 1, 0, 1, bias: false);
                        if (Tick((long)Cin * Cout * Tr)) yield return null;
                        scf = t3;
                    }
                    AddScaleOp(outBuf, scf, Cout * Tr, RSQRT2);
                    dims?.Invoke(Cout, Tr);
                    yield break;
                }
                // ---- legacy path (FastKernels == false): the original dispatch list ----------
                // residual: conv2(lrelu(AdaIN2(conv1(pool?(lrelu(AdaIN1(x)))))))
                CopySliceOp(t1, 0, x, 0, Cin * T);
                StyleFc(p + "/norm1_fc", styleVec, styleG1Buf);
                InstanceNormStyle(t1, styleG1Buf, Cin, T);
                Activate(t1, Cin * T, 2, 0.2f);
                ComputeBuffer r = t1;
                if (up)
                {
                    ConvT(p + "/pool", t1, t2, Cin, T, Cin, Tr, 3, 2, 1, Cin);   // depthwise
                    r = t2;
                }
                Conv(p + "/conv1", r, t3, Cin, Tr, Cout, Tr, 3, 1, 1, 1);
                if (Tick(3L * Cin * Cout * Tr)) yield return null;
                StyleFc(p + "/norm2_fc", styleVec, styleG2Buf);
                InstanceNormStyle(t3, styleG2Buf, Cout, Tr);
                Activate(t3, Cout * Tr, 2, 0.2f);
                Conv(p + "/conv2", t3, outBuf, Cout, Tr, Cout, Tr, 3, 1, 1, 1);
                if (Tick(3L * Cout * Cout * Tr)) yield return null;
                // shortcut: nearest x2 then conv1x1 (no bias) when shapes change
                ComputeBuffer sc = x;
                if (up) { GatherUp2(x, t1, Cin, T, Tr); sc = t1; }               // t1 free by now
                if (weights.Has(p + "/conv1x1.w"))
                {
                    Conv(p + "/conv1x1", sc, t3, Cin, Tr, Cout, Tr, 1, 1, 0, 1, bias: false);
                    if (Tick((long)Cin * Cout * Tr)) yield return null;
                    sc = t3;                                                     // t3 free by now
                }
                AddScaleOp(outBuf, sc, Cout * Tr, RSQRT2);                       // (res+short)/sqrt2
                dims?.Invoke(Cout, Tr);
            }

            /// <summary>Generator AdaINResBlock1 (AdaIN+Snake+dilated convs, dil 1/3/5) —
            /// in-place on x [C,T]. Scratch t1,t2 >= C*T floats.</summary>
            public IEnumerator SnakeResBlockY(string p, ComputeBuffer x, int C, int T,
                                              ComputeBuffer styleVec, ComputeBuffer t1, ComputeBuffer t2)
            {
                int K = weights.Shape(p + "/c1_0.w")[2];
                int[] dil = { 1, 3, 5 };
                for (int j = 0; j < 3; j++)
                {
                    if (FastKernels && FastKernels2)
                    {
                        // v2: v1's fused AdaIN+Snake prologues, and conv2 accumulates straight
                        // into x (outMode 1 == the AddBuf it replaces) — t1 never touched
                        StyleFc($"{p}/ada1_{j}_fc", styleVec, styleG1Buf);
                        InStats(x, C, T);
                        ConvFused2($"{p}/c1_{j}", x, t2, C, T, C, K, (K * dil[j] - dil[j]) / 2,
                                   dil[j], 1, styleG1Buf, $"{p}/a1_{j}");
                        if (Tick((long)K * C * C * T)) yield return null;
                        StyleFc($"{p}/ada2_{j}_fc", styleVec, styleG1Buf);
                        InStats(t2, C, T);
                        ConvFused2($"{p}/c2_{j}", t2, x, C, T, C, K, (K - 1) / 2,
                                   1, 1, styleG1Buf, $"{p}/a2_{j}", outMode: 1);
                        if (Tick((long)K * C * C * T)) yield return null;
                        continue;
                    }
                    if (FastKernels)
                    {
                        // AdaIN affine + Snake fused into the conv X prologue (mode 1); x stays
                        // read-only so the residual copy disappears too
                        StyleFc($"{p}/ada1_{j}_fc", styleVec, styleG1Buf);
                        InStats(x, C, T);
                        ConvFused($"{p}/c1_{j}", x, t2, C, T, C, K, (K * dil[j] - dil[j]) / 2,
                                  dil[j], 1, styleG1Buf, $"{p}/a1_{j}");
                        if (Tick((long)K * C * C * T)) yield return null;
                        StyleFc($"{p}/ada2_{j}_fc", styleVec, styleG1Buf);
                        InStats(t2, C, T);
                        ConvFused($"{p}/c2_{j}", t2, t1, C, T, C, K, (K - 1) / 2,
                                  1, 1, styleG1Buf, $"{p}/a2_{j}");
                        AddOp(x, t1, C * T);
                        if (Tick((long)K * C * C * T)) yield return null;
                        continue;
                    }
                    CopySliceOp(t1, 0, x, 0, C * T);
                    StyleFc($"{p}/ada1_{j}_fc", styleVec, styleG1Buf);
                    InstanceNormStyle(t1, styleG1Buf, C, T);
                    SnakeAct($"{p}/a1_{j}", t1, C, T);
                    Conv($"{p}/c1_{j}", t1, t2, C, T, C, T, K, 1, (K * dil[j] - dil[j]) / 2, dil[j]);
                    if (Tick((long)K * C * C * T)) yield return null;
                    StyleFc($"{p}/ada2_{j}_fc", styleVec, styleG1Buf);
                    InstanceNormStyle(t2, styleG1Buf, C, T);
                    SnakeAct($"{p}/a2_{j}", t2, C, T);
                    Conv($"{p}/c2_{j}", t2, t1, C, T, C, T, K, 1, (K - 1) / 2, 1);
                    AddOp(x, t1, C * T);
                    if (Tick((long)K * C * C * T)) yield return null;
                }
            }

            // ================= readback helpers =================================================
            /// <summary>FastKernels2 pipeline: a readback REQUESTED early (right after its
            /// producing dispatches) and resolved later — the GPU keeps executing queued work
            /// and the copy lands as soon as the producer drains, instead of after everything
            /// queued since. Falls back to a deferred sync GetData when async is unsupported.</summary>
            struct Pending
            {
                public UnityEngine.Rendering.AsyncGPUReadbackRequest req;
                public bool async;
                public ComputeBuffer buf;
                public int count;
            }

            Pending BeginReadback(ComputeBuffer buf, int count)
            {
                var p = new Pending { buf = buf, count = count };
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    p.req = UnityEngine.Rendering.AsyncGPUReadback.Request(buf, count * 4, 0);
                    p.async = true;
                }
                return p;
            }

            IEnumerator ResolvePending(Pending p, Action<float[]> sink)
            {
                float[] arr = new float[p.count];
                if (p.async)
                {
                    while (!p.req.done) yield return null;
                    if (!p.req.hasError)
                    {
                        p.req.GetData<float>().CopyTo(arr);
                        sink(arr);
                        yield break;
                    }
                }
                p.buf.GetData(arr, 0, 0, p.count);
                sink(arr);
            }

            IEnumerator ReadbackYielding(ComputeBuffer buf, int count, Action<float[]> sink)
            {
                float[] arr = new float[count];
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(buf, count * 4, 0);
                    while (!req.done) yield return null;
                    if (!req.hasError)
                    {
                        req.GetData<float>().CopyTo(arr);
                        sink(arr);
                        yield break;
                    }
                }
                buf.GetData(arr, 0, 0, count);
                sink(arr);
            }

            float[] ReadNow(ComputeBuffer buf, int count)   // sync (CaptureStages only)
            {
                float[] arr = new float[count];
                buf.GetData(arr, 0, 0, count);
                return arr;
            }

            // ================= full forward =====================================================
            /// <summary>Mirror of KokoroCPU.Forward: ids (with $ bounds) + voicepack row [256]
            /// -> 24 kHz wav via onWav. randU01/randN01 feed the NSF RNG (probes inject dumps).
            /// Runs GPU dispatches on the caller's (main) thread, CPU stages on worker Tasks.</summary>
            public IEnumerator ForwardYielding(int[] ids, float[] refS, float speed,
                                               Func<int, float[]> randU01, Func<int, float[]> randN01,
                                               Action<float[]> onWav)
            {
                int T = ids.Length;
                var swAll = System.Diagnostics.Stopwatch.StartNew();
                var stages = CaptureStages ? new KokoroCPU.Stages { T = T } : null;

                float[] sd = new float[128], sp = new float[128];
                Array.Copy(refS, 0, sd, 0, 128);
                Array.Copy(refS, 128, sp, 0, 128);

                UploadIds(ids);
                styleSdBuf.SetData(sd);
                styleSpBuf.SetData(sp);
                bool v2 = FastKernels && FastKernels2;

                // ---------------- TextEncoder convs FIRST (v2) ----------------------------------
                // The tenc branch depends only on ids: issuing its convs + readback BEFORE
                // PLBERT lets the tenc CPU biLSTM run concurrently with the predictor's CPU
                // stage (legacy ran it serially after the F0/N stacks, with the GPU idle).
                macAcc = 0;
                Pending teRb = default;
                if (v2)
                {
                    EmbedText(teA, T);                                            // [512,T]
                    for (int i = 0; i < 3; i++)
                    {
                        // i>0: the previous iteration's LeakyReLU is fused into this conv's
                        // X prologue; the last activation stays explicit for the readback
                        Conv($"tenc/cnn{i}/conv", teA, teB, 512, T, 512, T, 5, 1, 2, 1,
                             inMode: i > 0 ? 3 : 0, inSlope: 0.2f);
                        LayerNorm($"tenc/cnn{i}/ln", teB, teA, T, 512, 1e-5f, 1, T);
                        if (i == 2) Activate(teA, 512 * T, 2, 0.2f);
                        if (Tick(5L * 512 * 512 * T)) yield return null;
                    }
                    teRb = BeginReadback(teA, 512 * T);
                }

                // ---------------- PLBERT (GPU): embed -> LN -> map -> 12x shared layer ----------
                EmbedAlbert(embA, T);
                LayerNorm("bert/emb/ln", embA, embB, T, 128, 1e-12f, 128, 1);
                Linear("bert/map", embB, bA, T, 128, 768);
                for (int layer = 0; layer < 12; layer++)
                {
                    Linear("bert/layer/attn_q", bA, qB, T, 768, 768);
                    Linear("bert/layer/attn_k", bA, kB, T, 768, 768);
                    Linear("bert/layer/attn_v", bA, vB, T, 768, 768);
                    if (Tick(3L * T * 768 * 768)) yield return null;
                    AttentionBi(qB, kB, vB, atB, T, 12, 64);
                    Linear("bert/layer/attn_o", atB, bB, T, 768, 768);
                    if (v2)   // residual add fused into the cooperative LN (2 fewer dispatches)
                        LayerNormAdd("bert/layer/attn_ln", bB, bA, bC, T, 768, 1e-12f, 768, 1);
                    else
                    {
                        AddOp(bB, bA, T * 768);
                        LayerNorm("bert/layer/attn_ln", bB, bC, T, 768, 1e-12f, 768, 1);
                    }
                    if (Tick(2L * T * T * 768 + (long)T * 768 * 768)) yield return null;
                    Linear("bert/layer/ffn", bC, ffB, T, 768, 2048, act: 1);     // gelu_new fused
                    if (Tick((long)T * 768 * 2048)) yield return null;
                    Linear("bert/layer/ffn_out", ffB, bB, T, 2048, 768);
                    if (v2)
                        LayerNormAdd("bert/layer/ln", bB, bC, bA, T, 768, 1e-12f, 768, 1);
                    else
                    {
                        AddOp(bB, bC, T * 768);
                        LayerNorm("bert/layer/ln", bB, bA, T, 768, 1e-12f, 768, 1);
                    }
                    if (Tick((long)T * 2048 * 768)) yield return null;
                }
                if (stages != null) stages.bertDur = ReadNow(bA, T * 768);
                Linear("benc", bA, dEnBuf, T, 768, 512);                          // d_en rows [T,512]

                // ---------------- CPU: DurationEncoder + head + alignment + shared biLSTM -------
                float[] dEnRows = null;
                var rb1 = ReadbackYielding(dEnBuf, T * 512, a => dEnRows = a);
                while (rb1.MoveNext()) yield return rb1.Current;
                BertMs = (float)swAll.Elapsed.TotalMilliseconds;

                float[] d = null, duration = null, sh = null;
                int[] predDur = null, frame2tok = null;
                int F = 0;
                Exception cpuErr = null;
                var predTask = Task.Run(() =>
                {
                    try
                    {
                        d = cpu.DurationEncode(dEnRows, T, sp);                   // [T,640]
                        (duration, predDur) = cpu.DurationHead(d, T, speed);
                        if (InjectPredDur != null) predDur = InjectPredDur;
                        foreach (int pd in predDur) F += pd;
                        frame2tok = new int[F];
                        for (int t = 0, f = 0; t < T; t++)
                            for (int k = 0; k < predDur[t]; k++) frame2tok[f++] = t;
                        // en rows [F,640] = d rows expanded by the alignment; shared biLSTM
                        float[] enRows = new float[F * 640];
                        for (int f = 0; f < F; f++)
                            Array.Copy(d, frame2tok[f] * 640, enRows, f * 640, 640);
                        sh = cpu.BiLstm(enRows, F, 640, "pred/shared");           // [F,512]
                    }
                    catch (Exception e) { cpuErr = e; }
                });
                // v2: the tenc readback was requested before PLBERT — resolve it now (already
                // in flight) and run the tenc biLSTM concurrently with the predictor task
                float[] teCT = null, tEnCT = null;
                Task tencTask = null;
                Action tencWork = () =>
                {
                    try
                    {
                        float[] rows = KokoroCPU.Transpose(teCT, 512, T);         // [T,512]
                        float[] tEnRows = cpu.BiLstm(rows, T, 512, "tenc/lstm");
                        tEnCT = KokoroCPU.Transpose(tEnRows, T, 512);             // [512,T]
                    }
                    catch (Exception e) { cpuErr = e; }
                };
                if (v2)
                {
                    var rbT = ResolvePending(teRb, a => teCT = a);
                    while (rbT.MoveNext()) yield return rbT.Current;
                    tencTask = Task.Run(tencWork);
                }
                var swCpu = System.Diagnostics.Stopwatch.StartNew();
                while (!predTask.IsCompleted) yield return null;
                PredCpuMs = (float)swCpu.Elapsed.TotalMilliseconds;
                if (cpuErr != null)
                {
                    ConsoleMessage.Warning($"Kokoro CPU predictor stage failed: {cpuErr.Message}");
                    onWav?.Invoke(null);
                    yield break;
                }
                if (stages != null)
                {
                    stages.dEn = KokoroCPU.Transpose(dEnRows, T, 512);
                    stages.d = d; stages.duration = duration; stages.predDur = predDur; stages.F = F;
                    stages.en = new float[640 * F];
                    for (int c = 0; c < 640; c++)
                        for (int f = 0; f < F; f++) stages.en[c * F + f] = d[frame2tok[f] * 640 + c];
                }

                EnsureFrameScratch(F);
                int F2 = 2 * F;
                xfBuf.SetData(KokoroCPU.Transpose(sh, F, 512), 0, 0, 512 * F);    // [512,F]

                // ---------------- F0/N predictors (GPU AdainResBlk stacks) ----------------------
                Pending f0Rb = default;
                foreach (string fam in new[] { "F0", "N" })
                {
                    var p0 = AdainBlockY($"pred/{fam}_0", xfBuf, fA, 512, F, false, styleSpBuf, fT1, fT2, fT3);
                    while (p0.MoveNext()) yield return p0.Current;
                    var p1 = AdainBlockY($"pred/{fam}_1", fA, fB, 512, F, true, styleSpBuf, fT1, fT2, fT3);
                    while (p1.MoveNext()) yield return p1.Current;
                    var p2 = AdainBlockY($"pred/{fam}_2", fB, fA, 256, F2, false, styleSpBuf, fT1, fT2, fT3);
                    while (p2.MoveNext()) yield return p2.Current;
                    Conv($"pred/{fam}_proj", fA, fam == "F0" ? F0Buf : NBuf, 256, F2, 1, F2, 1, 1, 0, 1);
                    if (Tick((long)256 * F2)) yield return null;
                    // v2: request the F0 readback BEFORE the N stack is issued — it lands as
                    // soon as the F0 family drains and the NSF task starts that much earlier
                    if (v2 && fam == "F0") f0Rb = BeginReadback(F0Buf, F2);
                }

                // F0 readback feeds the CPU NSF source (started now, overlaps the GPU decoder)
                float[] f0Cpu = null;
                if (v2)
                {
                    var rf = ResolvePending(f0Rb, a => f0Cpu = a);
                    while (rf.MoveNext()) yield return rf.Current;
                }
                else
                {
                    var rb2 = ReadbackYielding(F0Buf, F2, a => f0Cpu = a);
                    while (rb2.MoveNext()) yield return rb2.Current;
                }
                float[] har = null;
                var nsfTask = Task.Run(() =>
                {
                    try { har = cpu.NsfHar(f0Cpu, F2, randU01, randN01); }
                    catch (Exception e) { cpuErr = e; }
                });
                if (stages != null) { stages.F0 = f0Cpu; stages.N = ReadNow(NBuf, F2); }

                // ---------------- TextEncoder (legacy position; v2 ran it before PLBERT) --------
                if (!v2)
                {
                    EmbedText(teA, T);                                            // [512,T]
                    for (int i = 0; i < 3; i++)
                    {
                        Conv($"tenc/cnn{i}/conv", teA, teB, 512, T, 512, T, 5, 1, 2, 1);
                        LayerNorm($"tenc/cnn{i}/ln", teB, teA, T, 512, 1e-5f, 1, T);  // over channels
                        Activate(teA, 512 * T, 2, 0.2f);
                        if (Tick(5L * 512 * 512 * T)) yield return null;
                    }
                    var rb3 = ReadbackYielding(teA, 512 * T, a => teCT = a);
                    while (rb3.MoveNext()) yield return rb3.Current;
                    tencTask = Task.Run(tencWork);
                }
                swCpu.Restart();
                while (!tencTask.IsCompleted) yield return null;
                TencCpuMs = (float)swCpu.Elapsed.TotalMilliseconds;
                if (cpuErr != null)
                {
                    ConsoleMessage.Warning($"Kokoro CPU tenc stage failed: {cpuErr.Message}");
                    onWav?.Invoke(null);
                    yield break;
                }
                tEnBuf.SetData(tEnCT, 0, 0, 512 * T);
                if (stages != null) stages.tEn = tEnCT;

                // asr = t_en @ aln — frame2tok gather on GPU
                uint[] f2t = new uint[F];
                for (int f = 0; f < F; f++) f2t[f] = (uint)frame2tok[f];
                gatherBuf.SetData(f2t, 0, 0, F);
                GatherIdx(tEnBuf, asrBuf, gatherBuf, 512, T, F);                  // [512,F]
                if (stages != null) stages.asr = ReadNow(asrBuf, 512 * F);
                PredictorMs = (float)swAll.Elapsed.TotalMilliseconds - BertMs;

                // ---------------- Decoder trunk (GPU) -------------------------------------------
                Conv("dec/F0_conv", F0Buf, f0cBuf, 1, F2, 1, F, 3, 2, 1, 1);      // 80Hz -> 40Hz
                Conv("dec/N_conv", NBuf, ncBuf, 1, F2, 1, F, 3, 2, 1, 1);
                CopySliceOp(decCat, 0, asrBuf, 0, 512 * F);                       // cat[asr,F0,N]
                CopySliceOp(decCat, 512 * F, f0cBuf, 0, F);
                CopySliceOp(decCat, 513 * F, ncBuf, 0, F);
                var de = AdainBlockY("dec/encode", decCat, decA, 514, F, false, styleSdBuf, dT1, dT2, dT3);
                while (de.MoveNext()) yield return de.Current;
                Conv("dec/asr_res", asrBuf, asrResBuf, 512, F, 64, F, 1, 1, 0, 1);
                if (Tick((long)512 * 64 * F)) yield return null;
                int C = 1024, L = F;
                for (int b = 0; b < 4; b++)
                {
                    CopySliceOp(decCat, 0, decA, 0, C * L);                       // cat[x,asr_res,F0,N]
                    CopySliceOp(decCat, 1024 * L, asrResBuf, 0, 64 * L);
                    CopySliceOp(decCat, 1088 * L, f0cBuf, 0, L);
                    CopySliceOp(decCat, 1089 * L, ncBuf, 0, L);
                    var db = AdainBlockY($"dec/decode{b}", decCat, decA, 1090, L, b == 3,
                                         styleSdBuf, dT1, dT2, dT3, (c2, l2) => { C = c2; L = l2; });
                    while (db.MoveNext()) yield return db.Current;
                }
                // decA = dec_x [512, 2F]
                if (stages != null) stages.decX = ReadNow(decA, 512 * F2);
                DecoderMs = (float)swAll.Elapsed.TotalMilliseconds - BertMs - PredictorMs;

                // ---------------- Generator (GPU; NSF har from CPU) -----------------------------
                swCpu.Restart();
                while (!nsfTask.IsCompleted) yield return null;
                NsfWaitMs = (float)swCpu.Elapsed.TotalMilliseconds;
                if (cpuErr != null)
                {
                    ConsoleMessage.Warning($"Kokoro CPU NSF stage failed: {cpuErr.Message}");
                    onWav?.Invoke(null);
                    yield break;
                }
                int S = 600 * F, frames = 120 * F + 1;
                harBuf.SetData(har, 0, 0, S);
                Stft(harBuf, harCatBuf, S, frames);                               // [22, frames]
                if (Tick(20L * 22 * frames)) yield return null;

                // trunk stage 0: 512ch @2F -> 256ch @20F
                int L0 = 20 * F;
                if (!v2)
                {
                    CopySliceOp(gX, 0, decA, 0, 512 * F2);
                    Activate(gX, 512 * F2, 2, 0.1f);                              // in-loop slope 0.1
                }
                Conv("dec/gen/noise_conv0", harCatBuf, gSrc, 22, frames, 256, L0, 12, 6, 3, 1);
                if (Tick(12L * 22 * 256 * L0)) yield return null;
                var nr0 = SnakeResBlockY("dec/gen/noise_res0", gSrc, 256, L0, styleSdBuf, gT1, gT2);
                while (nr0.MoveNext()) yield return nr0.Current;
                if (v2)  // lrelu(0.1) fused into the ups0 X reads — decA copy disappears
                    ConvT("dec/gen/ups0", decA, gTmp, 512, F2, 256, L0, 20, 10, 5, 1,
                          inMode: 3, inSlope: 0.1f);
                else
                    ConvT("dec/gen/ups0", gX, gTmp, 512, F2, 256, L0, 20, 10, 5, 1);
                AddOp(gTmp, gSrc, 256 * L0);                                      // x += x_source
                if (Tick(2L * 512 * 256 * L0)) yield return null;
                for (int j = 0; j < 3; j++)
                {
                    CopySliceOp(gX, 0, gTmp, 0, 256 * L0);
                    var rb = SnakeResBlockY($"dec/gen/rb{j}", gX, 256, L0, styleSdBuf, gT1, gT2);
                    while (rb.MoveNext()) yield return rb.Current;
                    if (j == 0) CopySliceOp(gAcc, 0, gX, 0, 256 * L0);
                    else AddOp(gAcc, gX, 256 * L0);
                }
                ScaleOp(gAcc, 256 * L0, 1f / 3f);                                 // resblock mean

                // trunk stage 1: 256ch @20F -> 128ch @120F+1
                if (!v2) Activate(gAcc, 256 * L0, 2, 0.1f);
                Conv("dec/gen/noise_conv1", harCatBuf, gSrc, 22, frames, 128, frames, 1, 1, 0, 1);
                if (Tick(22L * 128 * frames)) yield return null;
                var nr1 = SnakeResBlockY("dec/gen/noise_res1", gSrc, 128, frames, styleSdBuf, gT1, gT2);
                while (nr1.MoveNext()) yield return nr1.Current;
                int L1 = 120 * F;
                if (v2)  // lrelu(0.1) fused into the ups1 X reads
                    ConvT("dec/gen/ups1", gAcc, gX, 256, L0, 128, L1, 12, 6, 3, 1,
                          inMode: 3, inSlope: 0.1f);
                else
                    ConvT("dec/gen/ups1", gAcc, gX, 256, L0, 128, L1, 12, 6, 3, 1);
                if (Tick(12L * 256 * 128 * L0)) yield return null;
                // reflection_pad (1,0): prepend index-1 sample (SPEC §12.8) via idx gather
                uint[] rp = new uint[frames];
                rp[0] = 1;
                for (int t = 1; t < frames; t++) rp[t] = (uint)(t - 1);
                gatherBuf.SetData(rp, 0, 0, frames);
                GatherIdx(gX, gTmp, gatherBuf, 128, L1, frames);                  // [128, 120F+1]
                AddOp(gTmp, gSrc, 128 * frames);                                  // x += x_source
                for (int j = 0; j < 3; j++)
                {
                    CopySliceOp(gX, 0, gTmp, 0, 128 * frames);
                    var rb = SnakeResBlockY($"dec/gen/rb{3 + j}", gX, 128, frames, styleSdBuf, gT1, gT2);
                    while (rb.MoveNext()) yield return rb.Current;
                    if (j == 0) CopySliceOp(gAcc, 0, gX, 0, 128 * frames);
                    else AddOp(gAcc, gX, 128 * frames);
                }
                ScaleOp(gAcc, 128 * frames, 1f / 3f);

                if (v2)  // DEFAULT slope 0.01 here (SPEC §12.5) — fused into conv_post's X reads
                    Conv("dec/gen/conv_post", gAcc, harCatBuf, 128, frames, 22, frames, 7, 1, 3, 1,
                         inMode: 3, inSlope: 0.01f);
                else
                {
                    Activate(gAcc, 128 * frames, 2, 0.01f);                       // DEFAULT slope here!
                    Conv("dec/gen/conv_post", gAcc, harCatBuf, 128, frames, 22, frames, 7, 1, 3, 1);
                }
                if (Tick(7L * 128 * 22 * frames)) yield return null;
                Istft(harCatBuf, wavBuf, frames, S);                              // reuse harCat as spec
                GeneratorMs = (float)swAll.Elapsed.TotalMilliseconds - BertMs - PredictorMs - DecoderMs;

                float[] wav = null;
                var rb4 = ReadbackYielding(wavBuf, S, a => wav = a);
                while (rb4.MoveNext()) yield return rb4.Current;
                EndToEndMs = (float)swAll.Elapsed.TotalMilliseconds;
                if (stages != null) { stages.wav = wav; LastStages = stages; }
                onWav?.Invoke(wav);
            }

            public void Dispose()
            {
                idsBuf?.Release(); styleSdBuf?.Release(); styleSpBuf?.Release();
                styleG1Buf?.Release(); styleG2Buf?.Release(); statsBuf?.Release();
                bA?.Release(); bB?.Release(); bC?.Release(); qB?.Release(); kB?.Release();
                vB?.Release(); atB?.Release(); ffB?.Release(); embA?.Release(); embB?.Release();
                dEnBuf?.Release(); teA?.Release(); teB?.Release(); tEnBuf?.Release();
                xfBuf?.Release(); fA?.Release(); fB?.Release();
                fT1?.Release(); fT2?.Release(); fT3?.Release();
                F0Buf?.Release(); NBuf?.Release(); f0cBuf?.Release(); ncBuf?.Release();
                asrBuf?.Release(); asrResBuf?.Release(); decCat?.Release(); decA?.Release();
                dT1?.Release(); dT2?.Release(); dT3?.Release(); gatherBuf?.Release();
                gX?.Release(); gSrc?.Release(); gT1?.Release(); gT2?.Release();
                gAcc?.Release(); gTmp?.Release();
                harBuf?.Release(); harCatBuf?.Release(); wavBuf?.Release();
            }
        }
    }
}
