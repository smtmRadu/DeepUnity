using System;
using System.Collections;
using UnityEngine;

namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // Chatterbox-Turbo S3Gen: speech tokens (25Hz) -> mel (50Hz, meanflow 2-step euler) ->
        // 24kHz waveform (HiFTGenerator NSF + iSTFTNet). Full-GPU via ChatterboxS3GenCS.compute.
        // Graph per SPEC.md §3/§6/§7. Batch=1, offline (finalize=True), masks all-ones.
        public class S3GenModel : IDisposable
        {
            const int D = ChatterboxConfig.ENC_DIM;        // 512
            const int MEL = ChatterboxConfig.MEL_DIM;      // 80
            const int ECH = ChatterboxConfig.EST_CH;       // 256
            const int EIN = ChatterboxConfig.EST_IN;       // 320
            const int TDIM = ChatterboxConfig.EST_TIME_DIM;// 1024

            readonly ComputeShader cs;
            readonly ChatterboxWeights weights;

            int kTokenEmbed, kLinear, kLayerNorm, kConv, kConvT, kRelAttn, kBidirAttn, kTimeEmb;
            int kSnake, kActivate, kAddBroadcast, kPack, kSliceCh, kRepeat, kGauss, kEuler;
            int kCumsum, kSineMerge, kSTFT, kMagPhase, kISTFT, kTrimFade;
            int kZero, kCopy, kCopySlice, kAdd, kScale;

            // persistent small buffers
            ComputeBuffer tokenIdsBuf, spkInBuf, spkProjBuf, timeInBuf, timeVecBuf, timeVecBuf2,
                          timeCatBuf, tEmbBuf, rEmbBuf, tMixBuf, tMlpBuf, phaseVecBuf;
            // per-synthesis scratch (grow-only)
            ComputeBuffer tokBuf, posBuf, posProjBuf, qBuf, kBuf, vBuf, attnBuf, ffBuf, encA, encB;
            ComputeBuffer muBuf, condBuf, xBuf, dxdtBuf, estIn, estA, estB, estC, skipBuf;
            ComputeBuffer melBuf, f0A, f0B, f0UpBuf, thetaBuf, noiseBuf, srcBuf, sstftBuf;
            ComputeBuffer vA, vB, vC, vD, rbT1, rbT2, rbAcc, wavBuf;

            int curTok, curMel, curSamp, curVoc;

            public bool IsReady => weights.IsReady;

            // ---- parity-test hooks (validation/ChatterboxParityProbe) ----------------------------
            // When set, these exact tensors are injected instead of GPU-generated randomness so the
            // output is bit-comparable against the Python reference dump. Layouts: flow noise
            // [T_mel, 80] ([T,C]); NSF noise [S, 9]; phases [9]. Null = normal random path.
            public float[] InjectFlowNoise, InjectNsfNoise, InjectNsfPhases;
            /// <summary>Conds subfolder in the manifest ("conds" default; "conds_<voice>" for
            /// alternative voices exported by validation/make_voice.py).</summary>
            public string CondsPrefix = "conds";
            // Post-run debug readback targets (buffers stay valid until the next synthesis)
            public ComputeBuffer DebugMu => muBuf;
            public ComputeBuffer DebugMel => melBuf;
            public ComputeBuffer DebugDxdt => dxdtBuf;
            public ComputeBuffer DebugF0 => f0B;
            public ComputeBuffer DebugSource => srcBuf;

            /// <summary>Wall-clock stage timings of the last SynthesizeYielding call (ms). CPU-side
            /// wall time including the coroutine frame yields; EndToEnd includes the final GPU
            /// readback, so it is the true tokens-in -> samples-out latency.</summary>
            public float EncoderMs, EstimatorMs, VocoderMs, ReadbackMs, EndToEndMs;

            public S3GenModel(ChatterboxWeights weights)
            {
                this.weights = weights;
                cs = DeepUnityMeta.ChatterboxS3GenCS;

                kTokenEmbed = cs.FindKernel("TokenEmbed");
                kLinear = cs.FindKernel("LinearBias");
                kLayerNorm = cs.FindKernel("LayerNormT");
                kConv = cs.FindKernel("Conv1D");
                kConvT = cs.FindKernel("ConvTranspose1D");
                kRelAttn = cs.FindKernel("RelPosAttention");
                kBidirAttn = cs.FindKernel("BidirAttention");
                kTimeEmb = cs.FindKernel("SinusTimeEmb");
                kSnake = cs.FindKernel("SnakeAct");
                kActivate = cs.FindKernel("Activate");
                kAddBroadcast = cs.FindKernel("AddBroadcastCh");
                kPack = cs.FindKernel("PackChannels");
                kSliceCh = cs.FindKernel("SliceChannels");
                kRepeat = cs.FindKernel("RepeatTime");
                kGauss = cs.FindKernel("GaussNoise");
                kEuler = cs.FindKernel("EulerStep");
                kCumsum = cs.FindKernel("CumsumPhase");
                kSineMerge = cs.FindKernel("SineMerge");
                kSTFT = cs.FindKernel("STFT16");
                kMagPhase = cs.FindKernel("MagPhase");
                kISTFT = cs.FindKernel("ISTFT16");
                kTrimFade = cs.FindKernel("TrimFade");
                kZero = cs.FindKernel("ZeroBuffer");
                kCopy = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAdd = cs.FindKernel("AddResidual");
                kScale = cs.FindKernel("ScaleBuf");

                tokenIdsBuf = new ComputeBuffer(4096, 4, ComputeBufferType.Structured);
                spkInBuf = new ComputeBuffer(ChatterboxConfig.XVECTOR_DIM, 4, ComputeBufferType.Structured);
                spkProjBuf = new ComputeBuffer(MEL, 4, ComputeBufferType.Structured);
                timeInBuf = new ComputeBuffer(ChatterboxConfig.EST_TIME_IN, 4, ComputeBufferType.Structured);
                timeVecBuf = new ComputeBuffer(TDIM, 4, ComputeBufferType.Structured);
                timeVecBuf2 = new ComputeBuffer(TDIM, 4, ComputeBufferType.Structured);
                timeCatBuf = new ComputeBuffer(2 * TDIM, 4, ComputeBufferType.Structured);
                tEmbBuf = new ComputeBuffer(TDIM, 4, ComputeBufferType.Structured);
                rEmbBuf = new ComputeBuffer(TDIM, 4, ComputeBufferType.Structured);
                tMixBuf = new ComputeBuffer(TDIM, 4, ComputeBufferType.Structured);
                tMlpBuf = new ComputeBuffer(ECH * 2, 4, ComputeBufferType.Structured);
                phaseVecBuf = new ComputeBuffer(9, 4, ComputeBufferType.Structured);
            }

            static int Div256(int n) => (n + 255) / 256;

            static void Grow(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            void EnsureScratch(int tokLen, int melLen, int genMelLen)
            {
                int samples = genMelLen * ChatterboxConfig.SAMPLES_PER_MEL_FRAME;
                int frames = samples / ChatterboxConfig.ISTFT_HOP + 1;
                // vocoder stage element maxima ([T, C] products): T0=g, 512 | 8g,256 | 40g,128 | 120g+1,64
                int voc = Math.Max(Math.Max(genMelLen * 512, 8 * genMelLen * 256),
                          Math.Max(40 * genMelLen * 128, (120 * genMelLen + 1) * 64));

                if (tokLen <= curTok && melLen <= curMel && samples <= curSamp && voc <= curVoc) return;
                curTok = Math.Max(tokLen, curTok); curMel = Math.Max(melLen, curMel);
                curSamp = Math.Max(samples, curSamp); curVoc = Math.Max(voc, curVoc);

                int T = curTok, M = curMel, S = curSamp, VMAX = curVoc;
                int T2 = 2 * T;                       // encoder after x2 upsample (>= mel length)
                Grow(ref tokBuf, T2 * D);             // reused pre/post upsample
                Grow(ref posBuf, (2 * T2 - 1) * D);
                Grow(ref posProjBuf, (2 * T2 - 1) * D);
                Grow(ref qBuf, T2 * D); Grow(ref kBuf, T2 * D); Grow(ref vBuf, T2 * D);
                Grow(ref attnBuf, T2 * D);
                Grow(ref ffBuf, T2 * Math.Max(ChatterboxConfig.ENC_FF, ChatterboxConfig.EST_FF));
                Grow(ref encA, T2 * D); Grow(ref encB, T2 * D);
                Grow(ref muBuf, M * MEL);
                Grow(ref condBuf, M * MEL);
                Grow(ref xBuf, M * MEL);
                Grow(ref dxdtBuf, M * MEL);
                Grow(ref estIn, M * EIN);
                Grow(ref estA, M * Math.Max(ChatterboxConfig.EST_ATTN_INNER, EIN));
                Grow(ref estB, M * Math.Max(ChatterboxConfig.EST_ATTN_INNER, EIN));
                Grow(ref estC, M * Math.Max(ChatterboxConfig.EST_ATTN_INNER, EIN));
                Grow(ref skipBuf, M * ECH);
                Grow(ref melBuf, M * MEL);
                Grow(ref f0A, M * 512); Grow(ref f0B, M * 512);
                Grow(ref f0UpBuf, S);
                Grow(ref thetaBuf, S * 9);
                Grow(ref noiseBuf, S * 9);
                Grow(ref srcBuf, S);
                Grow(ref sstftBuf, (S / 4 + 1) * 18);
                Grow(ref vA, VMAX); Grow(ref vB, VMAX); Grow(ref vC, VMAX); Grow(ref vD, VMAX);
                Grow(ref rbT1, VMAX); Grow(ref rbT2, VMAX); Grow(ref rbAcc, VMAX);
                Grow(ref wavBuf, S);
            }

            // ---------------- generic op helpers ([T, C] layout everywhere) ----------------------
            void Linear(string wName, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim,
                        int act = 0, bool bias = true, float leaky = 0.01f)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kLinear, "X", x);
                cs.SetBuffer(kLinear, "W", weights.Get(wName + ".w"));
                cs.SetBuffer(kLinear, "W_bias", weights.Get(bias ? wName + ".b" : wName + ".w"));
                cs.SetBuffer(kLinear, "Y", y);
                cs.Dispatch(kLinear, 1, (T + 7) / 8, (outDim + 31) / 32);
            }

            void LN(string name, ComputeBuffer x, ComputeBuffer y, int T, int dim, float eps)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", dim); cs.SetFloat("norm_eps", eps);
                cs.SetBuffer(kLayerNorm, "norm_input", x);
                cs.SetBuffer(kLayerNorm, "norm_output", y);
                cs.SetBuffer(kLayerNorm, "ln_gamma", weights.Get(name + ".w"));
                cs.SetBuffer(kLayerNorm, "ln_beta", weights.Get(name + ".b"));
                cs.Dispatch(kLayerNorm, Div256(T), 1, 1);
            }

            void Conv(string wName, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                      int inCh, int outCh, int kernel, int stride, int dilation, int padLeft,
                      int act = 0, float leaky = 0.01f)
            {
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen);
                cs.SetInt("in_dim", inCh); cs.SetInt("out_dim", outCh);
                cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("conv_dilation", dilation); cs.SetInt("pad_left", padLeft);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", 1);
                cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kConv, "X", x);
                cs.SetBuffer(kConv, "W", weights.Get(wName + ".w"));
                cs.SetBuffer(kConv, "W_bias", weights.Get(wName + ".b"));
                cs.SetBuffer(kConv, "Y", y);
                cs.Dispatch(kConv, Div256(outLen * outCh), 1, 1);
            }

            void ConvT(string wName, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                       int inCh, int outCh, int kernel, int stride, int pad)
            {
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen);
                cs.SetInt("in_dim", inCh); cs.SetInt("out_dim", outCh);
                cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("pad_left", pad); cs.SetInt("has_bias", 1);
                cs.SetBuffer(kConvT, "X", x);
                cs.SetBuffer(kConvT, "W", weights.Get(wName + ".w"));
                cs.SetBuffer(kConvT, "W_bias", weights.Get(wName + ".b"));
                cs.SetBuffer(kConvT, "Y", y);
                cs.Dispatch(kConvT, Div256(outLen * outCh), 1, 1);
            }

            void CopyOp(ComputeBuffer dst, ComputeBuffer src, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kCopy, "buf_a", dst); cs.SetBuffer(kCopy, "buf_b", src);
                cs.Dispatch(kCopy, Div256(count), 1, 1);
            }

            void CopySliceOp(ComputeBuffer dst, int dstOff, ComputeBuffer src, int srcOff, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetInt("copy_dst_offset", dstOff); cs.SetInt("copy_src_offset", srcOff);
                cs.SetBuffer(kCopySlice, "buf_a", dst); cs.SetBuffer(kCopySlice, "buf_b", src);
                cs.Dispatch(kCopySlice, Div256(count), 1, 1);
            }

            void AddOp(ComputeBuffer dstA, ComputeBuffer srcB, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kAdd, "buf_a", dstA); cs.SetBuffer(kAdd, "buf_b", srcB);
                cs.Dispatch(kAdd, Div256(count), 1, 1);
            }

            void ZeroOp(ComputeBuffer dst, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kZero, "buf_a", dst);
                cs.Dispatch(kZero, Div256(count), 1, 1);
            }

            void ScaleOp(ComputeBuffer buf, int count, float s)
            {
                cs.SetInt("buffer_size", count); cs.SetFloat("scale_val", s);
                cs.SetBuffer(kScale, "inout_buf", buf);
                cs.Dispatch(kScale, Div256(count), 1, 1);
            }

            void ActivateOp(ComputeBuffer buf, int count, int act, float leaky = 0.01f)
            {
                cs.SetInt("buffer_size", count); cs.SetInt("activation_type", act);
                cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kActivate, "inout_buf", buf);
                cs.Dispatch(kActivate, Div256(count), 1, 1);
            }

            void PackOp(ComputeBuffer dst, int dstDim, int dstOff, ComputeBuffer src, int T, int srcDim)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", srcDim);
                cs.SetInt("pack_dst_dim", dstDim); cs.SetInt("pack_dst_off", dstOff);
                cs.SetBuffer(kPack, "buf_a", dst); cs.SetBuffer(kPack, "buf_b", src);
                cs.Dispatch(kPack, Div256(T * srcDim), 1, 1);
            }

            void RepeatOp(ComputeBuffer src, ComputeBuffer dst, int outT, int ch, int factor)
            {
                cs.SetInt("seq_len", outT); cs.SetInt("in_dim", ch); cs.SetInt("factor", factor);
                cs.SetBuffer(kRepeat, "X", src); cs.SetBuffer(kRepeat, "Y", dst);
                cs.Dispatch(kRepeat, Div256(outT * ch), 1, 1);
            }

            // ---------------- encoder (UpsampleConformerEncoder) --------------------------------
            void UploadRelPos(int T)
            {
                // EspnetRelPositionalEncoding table: rel positions [T-1 .. -(T-1)], interleaved sin/cos
                int rows = 2 * T - 1;
                float[] pe = new float[rows * D];
                for (int r = 0; r < rows; r++)
                {
                    int rel = (T - 1) - r;              // row 0 = most positive (key far left)
                    for (int i = 0; i < D / 2; i++)
                    {
                        double div = Math.Exp(-Math.Log(10000.0) * (2.0 * i) / D);
                        double ang = rel * div;
                        pe[r * D + 2 * i] = (float)Math.Sin(ang);
                        pe[r * D + 2 * i + 1] = (float)Math.Cos(ang);
                    }
                }
                posBuf.SetData(pe, 0, 0, rows * D);
            }

            void EncoderLayer(string p, ComputeBuffer x, int T, float lnEps)
            {
                // x += attn(LN_mha(x)) — rel-pos MHA
                LN(p + ".norm_mha", x, encB, T, D, lnEps);
                Linear(p + ".attn.linear_q", encB, qBuf, T, D, D);
                Linear(p + ".attn.linear_k", encB, kBuf, T, D, D);
                Linear(p + ".attn.linear_v", encB, vBuf, T, D, D);
                // p = linear_pos(pos_emb) (no bias) over the whole rel table
                cs.SetInt("seq_len", 2 * T - 1); cs.SetInt("in_dim", D); cs.SetInt("out_dim", D);
                cs.SetInt("activation_type", 0); cs.SetInt("has_bias", 0);
                cs.SetBuffer(kLinear, "X", posBuf);
                cs.SetBuffer(kLinear, "W", weights.Get(p + ".attn.linear_pos.w"));
                cs.SetBuffer(kLinear, "W_bias", weights.Get(p + ".attn.linear_pos.w"));
                cs.SetBuffer(kLinear, "Y", posProjBuf);
                cs.Dispatch(kLinear, 1, (2 * T - 1 + 7) / 8, (D + 31) / 32);

                cs.SetInt("seq_len", T);
                cs.SetInt("num_heads", ChatterboxConfig.ENC_HEADS);
                cs.SetInt("head_dim", ChatterboxConfig.ENC_HEAD_DIM);
                cs.SetFloat("scale", 1f / Mathf.Sqrt(ChatterboxConfig.ENC_HEAD_DIM));
                cs.SetBuffer(kRelAttn, "Q", qBuf); cs.SetBuffer(kRelAttn, "K", kBuf);
                cs.SetBuffer(kRelAttn, "V", vBuf); cs.SetBuffer(kRelAttn, "P", posProjBuf);
                cs.SetBuffer(kRelAttn, "pos_bias_u", weights.Get(p + ".attn.pos_bias_u"));
                cs.SetBuffer(kRelAttn, "pos_bias_v", weights.Get(p + ".attn.pos_bias_v"));
                cs.SetBuffer(kRelAttn, "AttendedValues", attnBuf);
                cs.Dispatch(kRelAttn, T, ChatterboxConfig.ENC_HEADS, 1);

                Linear(p + ".attn.linear_out", attnBuf, encB, T, D, D);
                AddOp(x, encB, T * D);

                // x += FF(LN_ff(x)) — Linear(512->2048) silu -> Linear(2048->512)
                LN(p + ".norm_ff", x, encB, T, D, lnEps);
                Linear(p + ".ff.w1", encB, ffBuf, T, D, ChatterboxConfig.ENC_FF, act: 1);
                Linear(p + ".ff.w2", ffBuf, encB, T, ChatterboxConfig.ENC_FF, D);
                AddOp(x, encB, T * D);
            }

            /// <summary>tokens (prompt+gen+SIL) -> mu [2T, 80] in muBuf. Returns mel length 2T.</summary>
            IEnumerator EncodeYielding(int[] tokens)
            {
                int T = tokens.Length;
                uint[] arr = new uint[T];
                for (int i = 0; i < T; i++) arr[i] = (uint)tokens[i];
                tokenIdsBuf.SetData(arr);

                cs.SetInt("seq_len", T); cs.SetInt("in_dim", D);
                cs.SetBuffer(kTokenEmbed, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kTokenEmbed, "emb_weights", weights.Get("s3gen/enc/input_embedding"));
                cs.SetBuffer(kTokenEmbed, "embed_output", tokBuf);
                cs.Dispatch(kTokenEmbed, Div256(T * D), 1, 1);

                // embed: Linear -> LN(1e-5) -> x*sqrt(512); rel-pos table for T
                Linear("s3gen/enc/embed.linear", tokBuf, encA, T, D, D);
                LN("s3gen/enc/embed.ln", encA, tokBuf, T, D, ChatterboxConfig.EMBED_LN_EPS);
                ScaleOp(tokBuf, T * D, Mathf.Sqrt(D));
                UploadRelPos(T);
                yield return null;

                // pre_lookahead: right-pad3 conv k4 leaky(0.01) -> left-pad2 conv k3 -> +residual
                Conv("s3gen/enc/prelook.conv1", tokBuf, encA, T, T, D, D, 4, 1, 1, 0, act: 4, leaky: 0.01f);
                Conv("s3gen/enc/prelook.conv2", encA, encB, T, T, D, D, 3, 1, 1, 2);
                AddOp(tokBuf, encB, T * D);
                yield return null;

                for (int i = 0; i < ChatterboxConfig.ENC_LAYERS; i++)
                {
                    EncoderLayer($"s3gen/enc/enc{i}", tokBuf, T, ChatterboxConfig.ENC_LN_EPS);
                    yield return null;
                }

                // up_layer: nearest x2 -> left-pad4 conv k5
                int T2 = 2 * T;
                RepeatOp(tokBuf, encA, T2, D, 2);
                Conv("s3gen/enc/up_layer.conv", encA, tokBuf, T2, T2, D, D, 5, 1, 1, 4);
                yield return null;

                // up_embed + 4 up-encoders at 50Hz
                Linear("s3gen/enc/up_embed.linear", tokBuf, encA, T2, D, D);
                LN("s3gen/enc/up_embed.ln", encA, tokBuf, T2, D, ChatterboxConfig.EMBED_LN_EPS);
                ScaleOp(tokBuf, T2 * D, Mathf.Sqrt(D));
                UploadRelPos(T2);
                yield return null;

                for (int i = 0; i < ChatterboxConfig.ENC_UP_LAYERS; i++)
                {
                    EncoderLayer($"s3gen/enc/upenc{i}", tokBuf, T2, ChatterboxConfig.ENC_LN_EPS);
                    yield return null;
                }

                LN("s3gen/enc/after_norm", tokBuf, encA, T2, D, ChatterboxConfig.EMBED_LN_EPS);
                Linear("s3gen/enc/encoder_proj", encA, muBuf, T2, D, MEL);
                yield return null;
            }

            // ---------------- meanflow estimator --------------------------------------------------
            void TimeEmbed(float t, ComputeBuffer outEmb)
            {
                cs.SetInt("out_dim", ChatterboxConfig.EST_TIME_IN);
                cs.SetFloat("t_scalar", t);
                cs.SetBuffer(kTimeEmb, "Y", timeInBuf);
                cs.Dispatch(kTimeEmb, Div256(ChatterboxConfig.EST_TIME_IN), 1, 1);
                Linear("s3gen/est/time_mlp1", timeInBuf, timeVecBuf, 1, ChatterboxConfig.EST_TIME_IN, TDIM, act: 1);
                Linear("s3gen/est/time_mlp2", timeVecBuf, outEmb, 1, TDIM, TDIM);
            }

            void CausalResnet(string p, ComputeBuffer x, ComputeBuffer y, int T, int inCh)
            {
                // h = Mish(LN(CausalConv_k3(x))); h += tmlp(Mish(tMix)); h = Mish(LN(CausalConv_k3(h)));
                // y = h + Conv_k1(x)
                Conv(p + ".block1.conv", x, estA, T, T, inCh, ECH, 3, 1, 1, 2);
                LN(p + ".block1.ln", estA, estB, T, ECH, 1e-5f);
                ActivateOp(estB, T * ECH, 3);
                // time inject: tmlp = Linear(Mish(tMix)) [256]
                CopyOp(timeVecBuf2, tMixBuf, TDIM);
                ActivateOp(timeVecBuf2, TDIM, 3);
                Linear(p + ".tmlp", timeVecBuf2, tMlpBuf, 1, TDIM, ECH);
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", ECH);
                cs.SetBuffer(kAddBroadcast, "inout_buf", estB);
                cs.SetBuffer(kAddBroadcast, "buf_b", tMlpBuf);
                cs.Dispatch(kAddBroadcast, Div256(T * ECH), 1, 1);

                Conv(p + ".block2.conv", estB, estA, T, T, ECH, ECH, 3, 1, 1, 2);
                LN(p + ".block2.ln", estA, y, T, ECH, 1e-5f);
                ActivateOp(y, T * ECH, 3);
                Conv(p + ".res_conv", x, estA, T, T, inCh, ECH, 1, 1, 1, 0);
                AddOp(y, estA, T * ECH);
            }

            void EstTransformer(string p, ComputeBuffer x, int T)
            {
                // x += to_out(attn(LN(x))): to_q/k/v 256->512 no bias, 8x64, bidirectional
                LN(p + ".norm1", x, estA, T, ECH, 1e-5f);
                Linear(p + ".to_q", estA, qBuf, T, ECH, 512, bias: false);
                Linear(p + ".to_k", estA, kBuf, T, ECH, 512, bias: false);
                Linear(p + ".to_v", estA, vBuf, T, ECH, 512, bias: false);
                cs.SetInt("seq_len", T);
                cs.SetInt("num_heads", ChatterboxConfig.EST_HEADS);
                cs.SetInt("head_dim", ChatterboxConfig.EST_HEAD_DIM);
                cs.SetFloat("scale", 1f / Mathf.Sqrt(ChatterboxConfig.EST_HEAD_DIM));
                cs.SetBuffer(kBidirAttn, "Q", qBuf); cs.SetBuffer(kBidirAttn, "K", kBuf);
                cs.SetBuffer(kBidirAttn, "V", vBuf);
                cs.SetBuffer(kBidirAttn, "AttendedValues", attnBuf);
                cs.Dispatch(kBidirAttn, T, ChatterboxConfig.EST_HEADS, 1);
                Linear(p + ".to_out", attnBuf, estA, T, 512, ECH);
                AddOp(x, estA, T * ECH);

                // x += ff_out(gelu(ff_in(LN(x))))
                LN(p + ".norm3", x, estA, T, ECH, 1e-5f);
                Linear(p + ".ff_in", estA, ffBuf, T, ECH, ChatterboxConfig.EST_FF, act: 2);
                Linear(p + ".ff_out", ffBuf, estA, T, ChatterboxConfig.EST_FF, ECH);
                AddOp(x, estA, T * ECH);
            }

            /// <summary>One estimator forward: dxdt(x, mu, spks, cond, t, r) -> dxdtBuf [T, 80].</summary>
            IEnumerator EstimatorYielding(int T, float t, float r)
            {
                // meanflow time embedding: tMix = mixer(cat[emb(t), emb(r)])
                TimeEmbed(t, tEmbBuf);
                TimeEmbed(r, rEmbBuf);
                CopySliceOp(timeCatBuf, 0, tEmbBuf, 0, TDIM);
                CopySliceOp(timeCatBuf, TDIM, rEmbBuf, 0, TDIM);
                Linear("s3gen/est/time_mixer", timeCatBuf, tMixBuf, 1, 2 * TDIM, TDIM, bias: false);

                // pack input [x | mu | spks_rep | cond] -> estIn [T, 320]
                PackOp(estIn, EIN, 0, xBuf, T, MEL);
                PackOp(estIn, EIN, MEL, muBuf, T, MEL);
                RepeatOp(spkProjBuf, estA, T, MEL, T);          // broadcast spks over time
                PackOp(estIn, EIN, 2 * MEL, estA, T, MEL);
                PackOp(estIn, EIN, 3 * MEL, condBuf, T, MEL);
                yield return null;

                // down block: resnet(320->256) + 4 tfmr; save skip; causal conv k3
                CausalResnet("s3gen/est/down0.res", estIn, estC, T, EIN);
                for (int j = 0; j < 4; j++) EstTransformer($"s3gen/est/down0.tfmr{j}", estC, T);
                CopyOp(skipBuf, estC, T * ECH);
                Conv("s3gen/est/down0.conv", estC, estB, T, T, ECH, ECH, 3, 1, 1, 2);
                CopyOp(estC, estB, T * ECH);
                yield return null;

                // 12 mid blocks
                for (int m = 0; m < ChatterboxConfig.EST_MID_BLOCKS; m++)
                {
                    CausalResnet($"s3gen/est/mid{m}.res", estC, estB, T, ECH);
                    CopyOp(estC, estB, T * ECH);
                    for (int j = 0; j < 4; j++) EstTransformer($"s3gen/est/mid{m}.tfmr{j}", estC, T);
                    yield return null;
                }

                // up block: cat[x, skip] (512) -> resnet(512->256) + 4 tfmr + causal conv.
                // Packed input goes into attnBuf (NOT estA): CausalResnet writes estA internally,
                // and a conv reading and writing the same buffer in one dispatch is a GPU race.
                PackOp(attnBuf, 2 * ECH, 0, estC, T, ECH);
                PackOp(attnBuf, 2 * ECH, ECH, skipBuf, T, ECH);
                CausalResnet("s3gen/est/up0.res", attnBuf, estC, T, 2 * ECH);
                for (int j = 0; j < 4; j++) EstTransformer($"s3gen/est/up0.tfmr{j}", estC, T);
                Conv("s3gen/est/up0.conv", estC, estB, T, T, ECH, ECH, 3, 1, 1, 2);
                yield return null;

                // final: CausalBlock (conv k3 -> LN -> mish) -> conv k1 -> dxdt
                Conv("s3gen/est/final_block.conv", estB, estA, T, T, ECH, ECH, 3, 1, 1, 2);
                LN("s3gen/est/final_block.ln", estA, estC, T, ECH, 1e-5f);
                ActivateOp(estC, T * ECH, 3);
                Conv("s3gen/est/final_proj", estC, dxdtBuf, T, T, ECH, MEL, 1, 1, 1, 0);
                yield return null;
            }

            // ---------------- vocoder (HiFTGenerator) --------------------------------------------
            // Internal scratch is rbT1/rbT2/rbAcc ONLY, so x/outSum may be any of the vA..vD buffers.
            void ResBlock(string p, ComputeBuffer x, ComputeBuffer outSum, int T, int ch, int kernel, bool accumulate)
            {
                // 3x: x += conv2_j( snake2( conv1_j( snake1(x) ) ) )   (dilations 1,3,5; conv2 dil 1)
                CopyOp(rbAcc, x, T * ch);
                for (int j = 0; j < 3; j++)
                {
                    int dil = ChatterboxConfig.RESBLOCK_DILATIONS[j];
                    CopyOp(rbT1, rbAcc, T * ch);
                    SnakeOp(p + $".a1_{j}", rbT1, T, ch);
                    Conv(p + $".c1_{j}", rbT1, rbT2, T, T, ch, ch, kernel, 1, dil, (kernel * dil - dil) / 2);
                    SnakeOp(p + $".a2_{j}", rbT2, T, ch);
                    Conv(p + $".c2_{j}", rbT2, rbT1, T, T, ch, ch, kernel, 1, 1, (kernel - 1) / 2);
                    AddOp(rbAcc, rbT1, T * ch);
                }
                if (accumulate) AddOp(outSum, rbAcc, T * ch);
                else CopyOp(outSum, rbAcc, T * ch);
            }

            void SnakeOp(string alphaName, ComputeBuffer buf, int T, int ch)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", ch);
                cs.SetBuffer(kSnake, "inout_buf", buf);
                cs.SetBuffer(kSnake, "snake_alpha", weights.Get(alphaName));
                cs.Dispatch(kSnake, Div256(T * ch), 1, 1);
            }

            /// <summary>melBuf [Tg, 80] -> 24kHz waveform in wavBuf ([480*Tg]).</summary>
            IEnumerator VocoderYielding(int Tg, int noiseSeed)
            {
                int S = Tg * ChatterboxConfig.SAMPLES_PER_MEL_FRAME;

                // ---- F0 predictor: 5x conv k3 p1 + ELU -> linear + abs
                Conv("s3gen/voc/f0.conv0", melBuf, f0A, Tg, Tg, MEL, 512, 3, 1, 1, 1, act: 7);
                for (int j = 1; j < 5; j++)
                {
                    Conv($"s3gen/voc/f0.conv{j}", f0A, f0B, Tg, Tg, 512, 512, 3, 1, 1, 1, act: 7);
                    (f0A, f0B) = (f0B, f0A);
                }
                Linear("s3gen/voc/f0.cls", f0A, f0B, Tg, 512, 1, act: 5);
                RepeatOp(f0B, f0UpBuf, S, 1, ChatterboxConfig.SAMPLES_PER_MEL_FRAME);
                yield return null;

                // ---- NSF source
                cs.SetInt("sample_len", S);
                cs.SetBuffer(kCumsum, "f0_up", f0UpBuf);
                cs.SetBuffer(kCumsum, "theta_out", thetaBuf);
                cs.Dispatch(kCumsum, 1, 1, 1);

                // per-harmonic random phase (h0 = 0), CPU RNG; injectable for parity tests
                float[] ph = InjectNsfPhases ?? new float[9];
                if (InjectNsfPhases == null)
                    for (int h = 1; h < 9; h++) ph[h] = UnityEngine.Random.Range(-Mathf.PI, Mathf.PI);
                phaseVecBuf.SetData(ph);

                if (InjectNsfNoise != null)
                {
                    noiseBuf.SetData(InjectNsfNoise, 0, 0, Math.Min(InjectNsfNoise.Length, S * 9));
                }
                else
                {
                    cs.SetInt("buffer_size", S * 9);
                    cs.SetInt("gauss_offset", 0);
                    cs.SetInt("rng_seed", noiseSeed);
                    cs.SetBuffer(kGauss, "buf_a", noiseBuf);
                    cs.Dispatch(kGauss, Div256(S * 9), 1, 1);
                }

                cs.SetInt("sample_len", S);
                cs.SetFloat("sine_amp", ChatterboxConfig.NSF_ALPHA);
                cs.SetFloat("noise_std", ChatterboxConfig.NSF_SIGMA);
                cs.SetFloat("voiced_threshold", ChatterboxConfig.NSF_VOICED_THRESHOLD);
                cs.SetBuffer(kSineMerge, "f0_up", f0UpBuf);
                cs.SetBuffer(kSineMerge, "theta_in", thetaBuf);
                cs.SetBuffer(kSineMerge, "phase_vec", phaseVecBuf);
                cs.SetBuffer(kSineMerge, "nsf_noise", noiseBuf);
                cs.SetBuffer(kSineMerge, "nsf_w", weights.Get("s3gen/voc/nsf_linear.w"));
                cs.SetBuffer(kSineMerge, "nsf_b", weights.Get("s3gen/voc/nsf_linear.b"));
                cs.SetBuffer(kSineMerge, "Y", srcBuf);
                cs.Dispatch(kSineMerge, Div256(S), 1, 1);
                yield return null;

                // ---- source STFT: [n_frames, 18]
                int nFrames = S / ChatterboxConfig.ISTFT_HOP + 1;
                cs.SetInt("n_frames", nFrames);
                cs.SetInt("sample_len", S);
                cs.SetBuffer(kSTFT, "X", srcBuf);
                cs.SetBuffer(kSTFT, "Y", sstftBuf);
                cs.Dispatch(kSTFT, Div256(nFrames * 9), 1, 1);
                yield return null;

                // ---- main branch
                int[] ups = ChatterboxConfig.UPSAMPLE_RATES;
                int[] ker = ChatterboxConfig.UPSAMPLE_KERNELS;
                int[] chs = { 512, 256, 128, 64 };
                // source_downs conv params per stage (SPEC §7): (k,s,p) = (30,15,7), (6,3,1), (1,1,0)
                int[] sdK = { 30, 6, 1 }; int[] sdS = { 15, 3, 1 }; int[] sdP = { 7, 1, 0 };

                Conv("s3gen/voc/conv_pre", melBuf, vA, Tg, Tg, MEL, 512, 7, 1, 1, 3);
                int curLen = Tg;
                for (int i = 0; i < 3; i++)
                {
                    ActivateOp(vA, curLen * chs[i], 4, ChatterboxConfig.LRELU_SLOPE);
                    int outLen = curLen * ups[i];       // (L-1)*s - 2p + k with p=(k-u)/2 -> L*u
                    ConvT($"s3gen/voc/ups{i}", vA, vB, outLen, curLen, chs[i], chs[i + 1], ker[i], ups[i], (ker[i] - ups[i]) / 2);
                    curLen = outLen;
                    (vA, vB) = (vB, vA);

                    if (i == 2)
                    {
                        // ReflectionPad1d((1,0)) on [T,C]: prepend row 1
                        CopySliceOp(vB, chs[3], vA, 0, curLen * chs[3]);
                        CopySliceOp(vB, 0, vA, chs[3], chs[3]);
                        curLen += 1;
                        (vA, vB) = (vB, vA);
                    }

                    // fusion: x += source_resblock(source_down(s_stft))
                    int sdOut = (nFrames + 2 * sdP[i] - sdK[i]) / sdS[i] + 1;
                    if (sdOut != curLen)
                        ConsoleMessage.Warning($"HiFT source fusion length mismatch at stage {i}: {sdOut} vs {curLen}");
                    Conv($"s3gen/voc/sdown{i}", sstftBuf, vB, curLen, nFrames, 18, chs[i + 1], sdK[i], sdS[i], 1, sdP[i]);
                    ResBlock($"s3gen/voc/srb{i}", vB, vC, curLen, chs[i + 1], ChatterboxConfig.SOURCE_RESBLOCK_KERNELS[i], accumulate: false);
                    AddOp(vA, vC, curLen * chs[i + 1]);
                    yield return null;

                    // x = mean of 3 resblocks (kernels 3,7,11)
                    ZeroOp(vD, curLen * chs[i + 1]);
                    for (int j = 0; j < 3; j++)
                    {
                        ResBlock($"s3gen/voc/rb{i * 3 + j}", vA, vD, curLen, chs[i + 1],
                                 ChatterboxConfig.RESBLOCK_KERNELS[j], accumulate: true);
                        yield return null;
                    }
                    CopyOp(vA, vD, curLen * chs[i + 1]);
                    ScaleOp(vA, curLen * chs[i + 1], 1f / 3f);
                }

                // conv_post: leaky(0.01 default) -> k7 p3 -> 18ch, then mag/phase -> iSTFT -> fade
                ActivateOp(vA, curLen * 64, 4, 0.01f);
                Conv("s3gen/voc/conv_post", vA, sstftBuf, curLen, curLen, 64, 18, 7, 1, 1, 3);

                cs.SetInt("n_frames", curLen);
                cs.SetBuffer(kMagPhase, "inout_buf", sstftBuf);
                cs.Dispatch(kMagPhase, Div256(curLen * 9), 1, 1);

                int outSamples = (curLen - 1) * ChatterboxConfig.ISTFT_HOP;
                cs.SetInt("sample_len", outSamples);
                cs.SetInt("n_frames", curLen);
                cs.SetFloat("audio_limit", ChatterboxConfig.AUDIO_LIMIT);
                cs.SetBuffer(kISTFT, "X", sstftBuf);
                cs.SetBuffer(kISTFT, "Y", wavBuf);
                cs.Dispatch(kISTFT, Div256(outSamples), 1, 1);

                cs.SetInt("sample_len", outSamples);
                cs.SetBuffer(kTrimFade, "inout_buf", wavBuf);
                cs.Dispatch(kTrimFade, 4, 1, 1);
                yield return null;
            }

            // ---------------- public entry --------------------------------------------------------
            /// <summary>
            /// speechTokens: T3 output filtered to &lt; 6561 (SIL x3 appended here). Baked-voice
            /// conds come from the weights folder. onWav receives the 24kHz mono float samples.
            /// </summary>
            public IEnumerator SynthesizeYielding(int[] speechTokens, Action<float[]> onWav, int seed = 0)
            {
                // tokens = prompt_token | gen | SIL SIL SIL
                int[] prompt = weights.ReadInts($"{CondsPrefix}/prompt_token");
                int[] tokens = new int[prompt.Length + speechTokens.Length + 3];
                Array.Copy(prompt, tokens, prompt.Length);
                Array.Copy(speechTokens, 0, tokens, prompt.Length, speechTokens.Length);
                for (int i = 0; i < 3; i++) tokens[prompt.Length + speechTokens.Length + i] = ChatterboxConfig.SIL_TOKEN;

                int T = tokens.Length;
                int melTotal = 2 * T;
                int promptMel = 2 * prompt.Length;
                int genMel = melTotal - promptMel;
                EnsureScratch(T, melTotal, genMel);

                // spks: spk_affine(normalize(x-vector))
                float[] xv = weights.ReadFloats($"{CondsPrefix}/embedding");
                float norm = 0f; foreach (float v in xv) norm += v * v;
                norm = Mathf.Sqrt(Mathf.Max(norm, 1e-12f));
                for (int i = 0; i < xv.Length; i++) xv[i] /= norm;
                spkInBuf.SetData(xv);
                Linear("s3gen/enc/spk_affine", spkInBuf, spkProjBuf, 1, ChatterboxConfig.XVECTOR_DIM, MEL);

                // cond feat: prompt mel then zeros
                ZeroOp(condBuf, melTotal * MEL);
                float[] pf = weights.ReadFloats($"{CondsPrefix}/prompt_feat");
                condBuf.SetData(pf, 0, 0, pf.Length);

                var swAll = System.Diagnostics.Stopwatch.StartNew();

                // encoder -> mu
                var enc = EncodeYielding(tokens);
                while (enc.MoveNext()) yield return enc.Current;
                EncoderMs = (float)swAll.Elapsed.TotalMilliseconds;

                // meanflow euler: x = N(0,1); 2 steps (t,r) = (0,.5), (.5,1); no CFG (distilled)
                if (InjectFlowNoise != null)
                {
                    xBuf.SetData(InjectFlowNoise, 0, 0, Math.Min(InjectFlowNoise.Length, melTotal * MEL));
                }
                else
                {
                    cs.SetInt("buffer_size", melTotal * MEL);
                    cs.SetInt("gauss_offset", 0);
                    cs.SetInt("rng_seed", seed == 0 ? UnityEngine.Random.Range(1, int.MaxValue) : seed);
                    cs.SetBuffer(kGauss, "buf_a", xBuf);
                    cs.Dispatch(kGauss, Div256(melTotal * MEL), 1, 1);
                }

                float[] tSpan = { 0f, 0.5f, 1f };
                for (int s = 0; s < ChatterboxConfig.CFM_TIMESTEPS; s++)
                {
                    var est = EstimatorYielding(melTotal, tSpan[s], tSpan[s + 1]);
                    while (est.MoveNext()) yield return est.Current;

                    cs.SetInt("buffer_size", melTotal * MEL);
                    cs.SetFloat("dt_val", tSpan[s + 1] - tSpan[s]);
                    cs.SetBuffer(kEuler, "inout_buf", xBuf);
                    cs.SetBuffer(kEuler, "buf_b", dxdtBuf);
                    cs.Dispatch(kEuler, Div256(melTotal * MEL), 1, 1);
                }

                EstimatorMs = (float)swAll.Elapsed.TotalMilliseconds - EncoderMs;

                // mel = x[promptMel:, :]
                CopySliceOp(melBuf, 0, xBuf, promptMel * MEL, genMel * MEL);

                var voc = VocoderYielding(genMel, seed == 0 ? UnityEngine.Random.Range(1, int.MaxValue) : seed + 1);
                while (voc.MoveNext()) yield return voc.Current;
                VocoderMs = (float)swAll.Elapsed.TotalMilliseconds - EncoderMs - EstimatorMs;

                // async readback of the final waveform. NOTE: dispatches are async — the GPU drains
                // the queued work HERE, so ReadbackMs carries most of the true GPU time and
                // EndToEndMs (set in BOTH paths) is the honest tokens-in -> samples-out wall time.
                int outSamples = 120 * genMel * ChatterboxConfig.ISTFT_HOP;   // (frames-1)*hop
                float[] wav = new float[outSamples];
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(wavBuf, outSamples * 4, 0);
                    while (!req.done) yield return null;
                    if (!req.hasError)
                    {
                        req.GetData<float>().CopyTo(wav);
                        ReadbackMs = (float)swAll.Elapsed.TotalMilliseconds - EncoderMs - EstimatorMs - VocoderMs;
                        EndToEndMs = (float)swAll.Elapsed.TotalMilliseconds;
                        onWav?.Invoke(wav);
                        yield break;
                    }
                }
                wavBuf.GetData(wav, 0, 0, outSamples);
                ReadbackMs = (float)swAll.Elapsed.TotalMilliseconds - EncoderMs - EstimatorMs - VocoderMs;
                EndToEndMs = (float)swAll.Elapsed.TotalMilliseconds;
                onWav?.Invoke(wav);
            }

            public void Dispose()
            {
                tokenIdsBuf?.Release(); spkInBuf?.Release(); spkProjBuf?.Release();
                timeInBuf?.Release(); timeVecBuf?.Release(); timeVecBuf2?.Release(); timeCatBuf?.Release();
                tEmbBuf?.Release(); rEmbBuf?.Release(); tMixBuf?.Release(); tMlpBuf?.Release(); phaseVecBuf?.Release();
                tokBuf?.Release(); posBuf?.Release(); posProjBuf?.Release();
                qBuf?.Release(); kBuf?.Release(); vBuf?.Release(); attnBuf?.Release(); ffBuf?.Release();
                encA?.Release(); encB?.Release();
                muBuf?.Release(); condBuf?.Release(); xBuf?.Release(); dxdtBuf?.Release();
                estIn?.Release(); estA?.Release(); estB?.Release(); estC?.Release(); skipBuf?.Release();
                melBuf?.Release(); f0A?.Release(); f0B?.Release(); f0UpBuf?.Release();
                thetaBuf?.Release(); noiseBuf?.Release(); srcBuf?.Release(); sstftBuf?.Release();
                vA?.Release(); vB?.Release(); vC?.Release(); vD?.Release();
                rbT1?.Release(); rbT2?.Release(); rbAcc?.Release(); wavBuf?.Release();
            }
        }
    }
}
