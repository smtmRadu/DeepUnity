using System;
using System.Collections;
using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace QwenASRModeling
    {
        // GPU forward orchestration for Qwen3-ASR (mel -> encoder -> projector -> Qwen3 decoder
        // -> greedy). Dispatch idioms mirror Qwen3_5Model; every kernel's math is the parity-
        // validated QwenASRCPU twin (validation/harness — ALL GATES PASS vs the python dumps).
        //
        // v1 scope: fp16 weights (int8 keyword plumbed via readW/wScale, untested), FP32 KV cache,
        // greedy decode only (do_sample=false is the model's own generation_config).
        public class QwenASRModel : IDisposable
        {
            public readonly QwenASRWeights weights;
            ComputeShader cs;

            // kernel ids
            int kDft, kLogMel, kGMax, kMelNorm;
            int kConvMel, kConv, kConvOutPos;
            int kMat, kLNorm, kEncScores, kEncSoftmax, kEncAttend;
            int kEmbed, kCopyTo, kRmsHidden, kRmsHead, kRope, kWriteCache, kFlash;
            int kGateUp, kDown, kLm1, kArgMax, kZero, kCopy, kAddRes;

            // frontend/encoder scratch
            ComputeBuffer framesBuf, dftCos, dftSin, powerBuf, melBuf, gmaxBuf;
            ComputeBuffer conv1Buf, conv2Buf, conv3Buf;
            ComputeBuffer encX, encNorm, encQ, encK, encV, encScores, encAtt, encFfn, projBuf, projTmp;

            // decoder scratch + KV
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf, qBuf, kBuf, vBuf, qNormBuf, kNormBuf;
            ComputeBuffer attendedBuf, attnOutBuf, mlpInterBuf, lastHiddenBuf, normSingleBuf;
            ComputeBuffer logitsBuf, argmaxBuf, tokenIdsBuf;
            ComputeBuffer ropeCos, ropeSin;
            ComputeBuffer[] kCache, vCache;
            int cachedLen;

            readonly int D, EncLayers, EncHeads, Ffn, H, Interm;
            readonly int cacheCapacity;
            int curSeqAlloc, curTokAlloc;

            volatile bool ropeReady, dftReady;
            volatile bool _ropeComputed, _dftComputed;
            uint[] _ropeCosData, _ropeSinData;
            float[] _dftCosData, _dftSinData;

            public bool IsReady => weights.IsReady && ropeReady && dftReady;

            public QwenASRModel(string paramsPath, int cacheCapacity = 1024)
            {
                D = QwenASRConfig.ENC_D_MODEL; EncLayers = QwenASRConfig.ENC_LAYERS;
                EncHeads = QwenASRConfig.ENC_HEADS; Ffn = QwenASRConfig.ENC_FFN;
                H = QwenASRConfig.HIDDEN_SIZE; Interm = QwenASRConfig.MLP_INTERMEDIATE_SIZE;
                this.cacheCapacity = cacheCapacity;

                // NOTE: QwenASRCS must be registered in DeepUnityMeta (see CHECKLIST.md — the
                // registry file is owned by the main workstream; this workstream may not edit it).
                cs = Resources.Load<ComputeShader>("ComputeShaders/QwenASRCS");
                if (cs == null)
                    throw new InvalidOperationException("QwenASRCS.compute not found in Resources/ComputeShaders.");
                CacheKernelIds();

                weights = new QwenASRWeights(paramsPath);

                int kvw = QwenASRConfig.HEADS_KV * QwenASRConfig.HEAD_DIM;
                kCache = new ComputeBuffer[QwenASRConfig.NUM_LAYERS];
                vCache = new ComputeBuffer[QwenASRConfig.NUM_LAYERS];
                for (int i = 0; i < QwenASRConfig.NUM_LAYERS; i++)
                {
                    kCache[i] = new ComputeBuffer(cacheCapacity * kvw, 4, ComputeBufferType.Structured);
                    vCache[i] = new ComputeBuffer(cacheCapacity * kvw, 4, ComputeBufferType.Structured);
                }

                logitsBuf = new ComputeBuffer(QwenASRConfig.VOCAB_SIZE, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(H, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(H, 4, ComputeBufferType.Structured);
                gmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);

                PrecomputeTablesAsync();
            }

            void CacheKernelIds()
            {
                kDft        = cs.FindKernel("MelDftPower");
                kLogMel     = cs.FindKernel("MelLogMel");
                kGMax       = cs.FindKernel("MelGlobalMax");
                kMelNorm    = cs.FindKernel("MelNormalize");
                kConvMel    = cs.FindKernel("Conv2dMelChunk");
                kConv       = cs.FindKernel("Conv2dS2");
                kConvOutPos = cs.FindKernel("ConvOutPos");
                kMat        = cs.FindKernel("MatMulBiasAct");
                kLNorm      = cs.FindKernel("LayerNorm");
                kEncScores  = cs.FindKernel("EncScores");
                kEncSoftmax = cs.FindKernel("EncSoftmaxRows");
                kEncAttend  = cs.FindKernel("EncAttend");
                kEmbed      = cs.FindKernel("EmbeddingLookup");
                kCopyTo     = cs.FindKernel("CopyToOffset");
                kRmsHidden  = cs.FindKernel("RmsNormHidden");
                kRmsHead    = cs.FindKernel("RmsNormHead");
                kRope       = cs.FindKernel("ApplyRope");
                kWriteCache = cs.FindKernel("WriteCacheFull");
                kFlash      = cs.FindKernel("FlashAttention");
                kGateUp     = cs.FindKernel("GateUp");
                kDown       = cs.FindKernel("Down");
                kLm1        = cs.FindKernel("LmHeadPredict1Vec");
                kArgMax     = cs.FindKernel("ArgMax");
                kZero       = cs.FindKernel("ZeroBuffer");
                kCopy       = cs.FindKernel("CopyBuffer");
                kAddRes     = cs.FindKernel("AddResidual");
            }

            // RoPE fp16 tables (θ=1e6, hd2=64) + DFT basis (201x400 cos/sin, hann NOT folded — the
            // frames are windowed CPU-side) computed on a background thread, uploaded when ready.
            void PrecomputeTablesAsync()
            {
                int maxSeq = Mathf.Max(cacheCapacity, 2048);
                int hd2 = QwenASRConfig.HEAD_DIM / 2;
                int packedLen = maxSeq * hd2 / 2;
                ropeCos = new ComputeBuffer(packedLen, 4, ComputeBufferType.Structured);
                ropeSin = new ComputeBuffer(packedLen, 4, ComputeBufferType.Structured);
                dftCos = new ComputeBuffer(QwenASRConfig.N_FREQS * QwenASRConfig.N_FFT, 4, ComputeBufferType.Structured);
                dftSin = new ComputeBuffer(QwenASRConfig.N_FREQS * QwenASRConfig.N_FFT, 4, ComputeBufferType.Structured);

                _ = Task.Run(() =>
                {
                    float[] invFreq = new float[hd2];
                    for (int i = 0; i < hd2; i++)
                        invFreq[i] = MathF.Pow(QwenASRConfig.ROPE_THETA, -2f * i / QwenASRConfig.HEAD_DIM);
                    uint[] c = new uint[packedLen];
                    uint[] s = new uint[packedLen];
                    for (int pos = 0; pos < maxSeq; pos++)
                    {
                        int baseU = pos * (hd2 / 2);
                        for (int j = 0; j < hd2 / 2; j++)
                        {
                            float a0 = pos * invFreq[2 * j], a1 = pos * invFreq[2 * j + 1];
                            c[baseU + j] = (uint)F32ToF16(MathF.Cos(a0)) | ((uint)F32ToF16(MathF.Cos(a1)) << 16);
                            s[baseU + j] = (uint)F32ToF16(MathF.Sin(a0)) | ((uint)F32ToF16(MathF.Sin(a1)) << 16);
                        }
                    }
                    _ropeCosData = c; _ropeSinData = s;
                    _ropeComputed = true;

                    int NF = QwenASRConfig.N_FREQS, NFFT = QwenASRConfig.N_FFT;
                    float[] dc = new float[NF * NFFT], ds = new float[NF * NFFT];
                    for (int f = 0; f < NF; f++)
                        for (int j = 0; j < NFFT; j++)
                        {
                            double a = 2.0 * Math.PI * f * j / NFFT;
                            dc[f * NFFT + j] = (float)Math.Cos(a);
                            ds[f * NFFT + j] = (float)Math.Sin(a);
                        }
                    _dftCosData = dc; _dftSinData = ds;
                    _dftComputed = true;
                });
                DeepUnityDispatcher.Run(UploadTablesWhenReady());
            }

            IEnumerator UploadTablesWhenReady()
            {
                while (!_ropeComputed) yield return null;
                ropeCos.SetData(_ropeCosData); ropeSin.SetData(_ropeSinData);
                _ropeCosData = null; _ropeSinData = null;
                ropeReady = true;
                while (!_dftComputed) yield return null;
                dftCos.SetData(_dftCosData); dftSin.SetData(_dftSinData);
                _dftCosData = null; _dftSinData = null;
                dftReady = true;
            }

            static ushort F32ToF16(float value)   // same managed f32->f16 as Qwen3_5Model
            {
                int i = BitConverter.SingleToInt32Bits(value);
                int s = (i >> 16) & 0x00008000;
                int e = ((i >> 23) & 0x000000ff) - (127 - 15);
                int m = i & 0x007fffff;
                if (e <= 0)
                {
                    if (e < -10) return (ushort)s;
                    m |= 0x00800000;
                    int t = 14 - e, a = (1 << (t - 1)) - 1, b = (m >> t) & 1;
                    m = (m + a + b) >> t;
                    return (ushort)(s | m);
                }
                if (e == 0xff - (127 - 15))
                {
                    if (m == 0) return (ushort)(s | 0x7c00);
                    m >>= 13;
                    return (ushort)(s | 0x7c00 | m | (m == 0 ? 1 : 0));
                }
                m = m + 0x00000fff + ((m >> 13) & 1);
                if ((m & 0x00800000) != 0) { m = 0; e += 1; }
                if (e > 30) return (ushort)(s | 0x7c00);
                return (ushort)(s | (e << 10) | (m >> 13));
            }

            static int Div256(int n) => (n + 255) / 256;

            void Realloc(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            void BindScalesGeneric(int kernel, ComputeBuffer scales, ComputeBuffer fallback)
            {
                // INT8 keyword variant reads W_scales; fp16 never references it, but D3D11 wants a
                // binding either way for the warmup pass — bind the weight itself as a stand-in.
                cs.SetBuffer(kernel, "W_scales", scales != null ? scales : fallback);
            }

            // Y = act(X @ W^T (+ b)) via the generic kernel. bias may be null.
            void MatMul(ComputeBuffer x, int rows, int inDim, string wName, int outDim,
                        string bName, ComputeBuffer y, int act = 0)
            {
                ComputeBuffer w = weights.Get(wName);
                ComputeBuffer scales = weights.Has(wName + ".scales") ? weights.Get(wName + ".scales") : null;
                cs.SetInt("rows", rows);
                cs.SetInt("in_dim", inDim);
                cs.SetInt("out_dim", outDim);
                cs.SetInt("act_mode", act);
                cs.SetInt("has_bias", bName != null ? 1 : 0);
                cs.SetBuffer(kMat, "X", x);
                cs.SetBuffer(kMat, "W", w);
                BindScalesGeneric(kMat, scales, w);
                cs.SetBuffer(kMat, "B", bName != null ? weights.Get(bName) : w);
                cs.SetBuffer(kMat, "Y", y);
                cs.Dispatch(kMat, 1, (rows + 7) / 8, (outDim + 31) / 32);
            }

            void LayerNorm(ComputeBuffer x, int rows, int dim, string gName, string bName, ComputeBuffer y)
            {
                cs.SetInt("rows", rows);
                cs.SetInt("in_dim", dim);
                cs.SetFloat("norm_eps", QwenASRConfig.ENC_LN_EPS);
                cs.SetBuffer(kLNorm, "norm_input", x);
                cs.SetBuffer(kLNorm, "norm_output", y);
                cs.SetBuffer(kLNorm, "norm_gamma", weights.Get(gName));
                cs.SetBuffer(kLNorm, "norm_beta", weights.Get(bName));
                cs.Dispatch(kLNorm, Div256(rows), 1, 1);
            }

            void AddResidual(ComputeBuffer dst, ComputeBuffer src, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kAddRes, "buf_a", dst);
                cs.SetBuffer(kAddRes, "buf_b", src);
                cs.Dispatch(kAddRes, Div256(count), 1, 1);
            }

            void Copy(ComputeBuffer dst, ComputeBuffer src, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kCopy, "buf_a", dst);
                cs.SetBuffer(kCopy, "buf_b", src);
                cs.Dispatch(kCopy, Div256(count), 1, 1);
            }

            // =================================================================== §1-§3 audio path
            /// <summary>Dispatches mel → conv frontend → windowed encoder → projector for a whole
            /// utterance. Yields between heavy stages. Result: projBuf holds [nTokens, H].</summary>
            public IEnumerator EncodeAudioYielding(float[] samples, int[] nTokensOut)
            {
                // ---- CPU prep: min-pad, reflect pad, windowed frames [T, 400]
                int n = Math.Max(samples.Length, QwenASRConfig.MIN_SAMPLES);
                int T = n / QwenASRConfig.HOP;
                int Tpad = (T + 99) / 100 * 100;
                int chunks = Tpad / 100;
                int nTok = QwenASRConfig.AudioTokenCount(T);
                nTokensOut[0] = nTok;

                const int NFFT = QwenASRConfig.N_FFT;
                int pad = NFFT / 2;
                float[] x = new float[n];                    // min-length zero pad first (SPEC §1)
                Array.Copy(samples, x, Math.Min(samples.Length, n));
                float[] xp = new float[n + 2 * pad];         // then reflect — identical to QwenASRCPU.Mel
                for (int i = 0; i < pad; i++) xp[i] = x[pad - i];
                Array.Copy(x, 0, xp, pad, n);
                for (int j = 1; j <= pad; j++) xp[pad + n - 1 + j] = x[n - 1 - j];
                float[] frames = new float[T * NFFT];
                for (int t = 0; t < T; t++)
                {
                    int s0 = t * QwenASRConfig.HOP;
                    for (int j = 0; j < NFFT; j++)
                        frames[t * NFFT + j] = xp[s0 + j] * (0.5f * (1f - MathF.Cos(2f * MathF.PI * j / NFFT)));
                }

                EnsureAudioScratch(T, Tpad, chunks, nTok);
                framesBuf.SetData(frames, 0, 0, T * NFFT);
                yield return null;

                // ---- mel
                cs.SetInt("n_frames", T);
                cs.SetInt("frames_padded", Tpad);
                cs.SetInt("n_freqs", QwenASRConfig.N_FREQS);
                cs.SetInt("n_fft", NFFT);
                cs.SetInt("n_mels", QwenASRConfig.N_MELS);

                cs.SetInt("buffer_size", QwenASRConfig.N_MELS * Tpad);   // zero (pad columns stay 0)
                cs.SetBuffer(kZero, "buf_a", melBuf);
                cs.Dispatch(kZero, Div256(QwenASRConfig.N_MELS * Tpad), 1, 1);

                cs.SetBuffer(kDft, "frames", framesBuf);
                cs.SetBuffer(kDft, "dft_cos", dftCos);
                cs.SetBuffer(kDft, "dft_sin", dftSin);
                cs.SetBuffer(kDft, "power_buf", powerBuf);
                cs.Dispatch(kDft, (T + 7) / 8, (QwenASRConfig.N_FREQS + 31) / 32, 1);

                cs.SetBuffer(kLogMel, "power_buf", powerBuf);
                cs.SetBuffer(kLogMel, "mel_filters", weights.Get("frontend/mel_filters"));
                cs.SetBuffer(kLogMel, "mel_buf", melBuf);
                cs.Dispatch(kLogMel, (T + 7) / 8, (QwenASRConfig.N_MELS + 31) / 32, 1);

                cs.SetBuffer(kGMax, "mel_buf", melBuf);
                cs.SetBuffer(kGMax, "gmax_buf", gmaxBuf);
                cs.Dispatch(kGMax, 1, 1, 1);

                cs.SetBuffer(kMelNorm, "mel_buf", melBuf);
                cs.SetBuffer(kMelNorm, "gmax_buf", gmaxBuf);
                cs.Dispatch(kMelNorm, Div256(QwenASRConfig.N_MELS * T), 1, 1);
                yield return null;

                // ---- conv frontend (per chunk)
                cs.SetInt("n_chunks", chunks);
                cs.SetInt("conv_in_c", 1); cs.SetInt("conv_in_h", 128); cs.SetInt("conv_in_w", 100);
                cs.SetInt("conv_out_c", QwenASRConfig.ENC_CONV_CH);
                cs.SetBuffer(kConvMel, "mel_buf", melBuf);
                cs.SetBuffer(kConvMel, "conv_w", weights.Get("enc/conv2d1.w"));
                cs.SetBuffer(kConvMel, "conv_b", weights.Get("enc/conv2d1.b"));
                cs.SetBuffer(kConvMel, "conv_out_buf", conv1Buf);
                cs.Dispatch(kConvMel, (64 * 50 + 63) / 64, (QwenASRConfig.ENC_CONV_CH + 3) / 4, chunks);

                cs.SetInt("conv_in_c", QwenASRConfig.ENC_CONV_CH);
                cs.SetInt("conv_in_h", 64); cs.SetInt("conv_in_w", 50);
                cs.SetBuffer(kConv, "conv_in", conv1Buf);
                cs.SetBuffer(kConv, "conv_w", weights.Get("enc/conv2d2.w"));
                cs.SetBuffer(kConv, "conv_b", weights.Get("enc/conv2d2.b"));
                cs.SetBuffer(kConv, "conv_out_buf", conv2Buf);
                cs.Dispatch(kConv, (32 * 25 + 63) / 64, (QwenASRConfig.ENC_CONV_CH + 3) / 4, chunks);

                cs.SetInt("conv_in_h", 32); cs.SetInt("conv_in_w", 25);
                cs.SetBuffer(kConv, "conv_in", conv2Buf);
                cs.SetBuffer(kConv, "conv_w", weights.Get("enc/conv2d3.w"));
                cs.SetBuffer(kConv, "conv_b", weights.Get("enc/conv2d3.b"));
                cs.SetBuffer(kConv, "conv_out_buf", conv3Buf);
                cs.Dispatch(kConv, (16 * 13 + 63) / 64, (QwenASRConfig.ENC_CONV_CH + 3) / 4, chunks);
                yield return null;

                int rem = T % 100;
                cs.SetInt("n_tokens", nTok);
                cs.SetInt("enc_d", D);
                cs.SetInt("full_chunks", T / 100);
                cs.SetInt("last_keep", QwenASRConfig.Ceil3(rem));
                cs.SetBuffer(kConvOutPos, "conv_out_buf", conv3Buf);
                cs.SetBuffer(kConvOutPos, "conv_out_w", weights.Get("enc/conv_out.w"));
                cs.SetBuffer(kConvOutPos, "pos_emb", weights.Get("enc/pos_emb"));
                cs.SetBuffer(kConvOutPos, "enc_x", encX);
                cs.Dispatch(kConvOutPos, (D + 31) / 32, 1, chunks);
                yield return null;

                // ---- encoder layers (pre-LN, 104-token windowed attention)
                cs.SetInt("enc_heads", EncHeads);
                cs.SetInt("enc_window", QwenASRConfig.ENC_WINDOW_TOKENS);
                float encScale = 1f / MathF.Sqrt(D / EncHeads);
                for (int li = 0; li < EncLayers; li++)
                {
                    string lp = $"enc/layer_{li}/";
                    LayerNorm(encX, nTok, D, lp + "ln1.w", lp + "ln1.b", encNorm);
                    MatMul(encNorm, nTok, D, lp + "attn_q.w", D, lp + "attn_q.b", encQ);
                    MatMul(encNorm, nTok, D, lp + "attn_k.w", D, lp + "attn_k.b", encK);
                    MatMul(encNorm, nTok, D, lp + "attn_v.w", D, lp + "attn_v.b", encV);

                    cs.SetInt("n_tokens", nTok);
                    cs.SetFloat("scale", encScale);
                    cs.SetBuffer(kEncScores, "Q_e", encQ);
                    cs.SetBuffer(kEncScores, "K_e", encK);
                    cs.SetBuffer(kEncScores, "enc_scores", encScores);
                    cs.Dispatch(kEncScores, (nTok + 3) / 4, (nTok + 31) / 32, (EncHeads + 3) / 4);

                    cs.SetBuffer(kEncSoftmax, "enc_scores", encScores);
                    cs.Dispatch(kEncSoftmax, Div256(EncHeads * nTok), 1, 1);

                    cs.SetBuffer(kEncAttend, "enc_scores", encScores);
                    cs.SetBuffer(kEncAttend, "V_e", encV);
                    cs.SetBuffer(kEncAttend, "enc_att", encAtt);
                    cs.Dispatch(kEncAttend, (D / EncHeads + 63) / 64, (nTok + 3) / 4, EncHeads);

                    MatMul(encAtt, nTok, D, lp + "attn_out.w", D, lp + "attn_out.b", encNorm);
                    AddResidual(encX, encNorm, nTok * D);

                    LayerNorm(encX, nTok, D, lp + "ln2.w", lp + "ln2.b", encNorm);
                    MatMul(encNorm, nTok, D, lp + "fc1.w", Ffn, lp + "fc1.b", encFfn, act: 1);
                    MatMul(encFfn, nTok, Ffn, lp + "fc2.w", D, lp + "fc2.b", encNorm);
                    AddResidual(encX, encNorm, nTok * D);
                    yield return null;
                }

                // ---- ln_post + projector
                LayerNorm(encX, nTok, D, "enc/ln_post.w", "enc/ln_post.b", encNorm);
                MatMul(encNorm, nTok, D, "proj/linear_1.w", D, "proj/linear_1.b", projTmp, act: 1);
                MatMul(projTmp, nTok, D, "proj/linear_2.w", H, "proj/linear_2.b", projBuf);
                yield return null;
            }

            void EnsureAudioScratch(int T, int Tpad, int chunks, int nTok)
            {
                Realloc(ref framesBuf, T * QwenASRConfig.N_FFT);
                Realloc(ref powerBuf, T * QwenASRConfig.N_FREQS);
                Realloc(ref melBuf, QwenASRConfig.N_MELS * Tpad);
                Realloc(ref conv1Buf, chunks * QwenASRConfig.ENC_CONV_CH * 64 * 50);
                Realloc(ref conv2Buf, chunks * QwenASRConfig.ENC_CONV_CH * 32 * 25);
                Realloc(ref conv3Buf, chunks * QwenASRConfig.ENC_CONV_CH * 16 * 13);
                if (nTok <= curTokAlloc) return;
                curTokAlloc = nTok;
                Realloc(ref encX, nTok * D);
                Realloc(ref encNorm, nTok * Math.Max(D, H));
                Realloc(ref encQ, nTok * D);
                Realloc(ref encK, nTok * D);
                Realloc(ref encV, nTok * D);
                Realloc(ref encScores, EncHeads * nTok * nTok);
                Realloc(ref encAtt, nTok * D);
                Realloc(ref encFfn, nTok * Ffn);
                Realloc(ref projTmp, nTok * D);
                Realloc(ref projBuf, nTok * H);
            }

            void EnsureDecoderScratch(int seqLen)
            {
                if (seqLen <= curSeqAlloc) return;
                curSeqAlloc = seqLen;
                int qDim = QwenASRConfig.HEADS_Q * QwenASRConfig.HEAD_DIM;
                int kvDim = QwenASRConfig.HEADS_KV * QwenASRConfig.HEAD_DIM;
                Realloc(ref hiddenBuf, seqLen * H);
                Realloc(ref skipBuf, seqLen * H);
                Realloc(ref normOutBuf, seqLen * H);
                Realloc(ref qBuf, seqLen * qDim);
                Realloc(ref kBuf, seqLen * kvDim);
                Realloc(ref vBuf, seqLen * kvDim);
                Realloc(ref qNormBuf, seqLen * qDim);
                Realloc(ref kNormBuf, seqLen * kvDim);
                Realloc(ref attendedBuf, seqLen * qDim);
                Realloc(ref attnOutBuf, seqLen * H);
                Realloc(ref mlpInterBuf, seqLen * Interm);
                Realloc(ref tokenIdsBuf, seqLen);
            }

            // =================================================================== §4 decoder
            void DispatchDecoderLayers(int seqLen)
            {
                const int hq = QwenASRConfig.HEADS_Q, hkv = QwenASRConfig.HEADS_KV, hd = QwenASRConfig.HEAD_DIM;
                int qDim = hq * hd, kvDim = hkv * hd;
                int hidTotal = seqLen * H;
                float attnScale = MathF.Pow(hd, -0.5f);

                for (int li = 0; li < QwenASRConfig.NUM_LAYERS; li++)
                {
                    string lp = $"dec/layer_{li}/";
                    Copy(skipBuf, hiddenBuf, hidTotal);

                    cs.SetInt("seq_len", seqLen);
                    cs.SetInt("hidden_size", H);
                    cs.SetFloat("norm_eps", QwenASRConfig.RMS_EPS);
                    cs.SetBuffer(kRmsHidden, "norm_input", hiddenBuf);
                    cs.SetBuffer(kRmsHidden, "norm_output", normOutBuf);
                    cs.SetBuffer(kRmsHidden, "norm_gamma", weights.Get(lp + "input_ln"));
                    cs.Dispatch(kRmsHidden, Div256(seqLen), 1, 1);

                    MatMul(normOutBuf, seqLen, H, lp + "q_proj", qDim, null, qBuf);
                    MatMul(normOutBuf, seqLen, H, lp + "k_proj", kvDim, null, kBuf);
                    MatMul(normOutBuf, seqLen, H, lp + "v_proj", kvDim, null, vBuf);

                    // QK-norm BEFORE RoPE (SPEC §4)
                    cs.SetInt("num_vectors", seqLen * hq);
                    cs.SetInt("head_dim", hd);
                    cs.SetBuffer(kRmsHead, "norm_input", qBuf);
                    cs.SetBuffer(kRmsHead, "norm_output", qNormBuf);
                    cs.SetBuffer(kRmsHead, "norm_gamma", weights.Get(lp + "q_norm"));
                    cs.Dispatch(kRmsHead, Div256(seqLen * hq), 1, 1);

                    cs.SetInt("num_vectors", seqLen * hkv);
                    cs.SetBuffer(kRmsHead, "norm_input", kBuf);
                    cs.SetBuffer(kRmsHead, "norm_output", kNormBuf);
                    cs.SetBuffer(kRmsHead, "norm_gamma", weights.Get(lp + "k_norm"));
                    cs.Dispatch(kRmsHead, Div256(seqLen * hkv), 1, 1);

                    int hd2 = hd / 2;
                    cs.SetInt("seq_len", seqLen);
                    cs.SetInt("rope_rot_dim", hd);
                    cs.SetInt("position_offset", cachedLen);
                    cs.SetInt("rope_num_heads", hq);
                    cs.SetBuffer(kRope, "rope_buf", qNormBuf);
                    cs.SetBuffer(kRope, "rope_cos", ropeCos);
                    cs.SetBuffer(kRope, "rope_sin", ropeSin);
                    cs.Dispatch(kRope, (seqLen * hq * hd2 + 127) / 128, 1, 1);

                    cs.SetInt("rope_num_heads", hkv);
                    cs.SetBuffer(kRope, "rope_buf", kNormBuf);
                    cs.Dispatch(kRope, (seqLen * hkv * hd2 + 127) / 128, 1, 1);

                    cs.SetInt("num_heads_kv", hkv);
                    cs.SetInt("cache_len", cachedLen);
                    cs.SetBuffer(kWriteCache, "kv_new", kNormBuf);
                    cs.SetBuffer(kWriteCache, "kv_cache", kCache[li]);
                    cs.Dispatch(kWriteCache, Div256(seqLen * hkv * hd), 1, 1);
                    cs.SetBuffer(kWriteCache, "kv_new", vBuf);
                    cs.SetBuffer(kWriteCache, "kv_cache", vCache[li]);
                    cs.Dispatch(kWriteCache, Div256(seqLen * hkv * hd), 1, 1);

                    cs.SetInt("seq_len_q", seqLen);
                    cs.SetInt("seq_len_k", cachedLen + seqLen);
                    cs.SetInt("num_heads_q", hq);
                    cs.SetFloat("scale", attnScale);
                    cs.SetBuffer(kFlash, "Q", qNormBuf);
                    cs.SetBuffer(kFlash, "K", kCache[li]);
                    cs.SetBuffer(kFlash, "V", vCache[li]);
                    cs.SetBuffer(kFlash, "AttendedValues", attendedBuf);
                    cs.Dispatch(kFlash, seqLen, hq, 1);

                    MatMul(attendedBuf, seqLen, qDim, lp + "o_proj", H, null, attnOutBuf);
                    AddResidual(attnOutBuf, skipBuf, hidTotal);
                    Copy(hiddenBuf, attnOutBuf, hidTotal);
                    Copy(skipBuf, hiddenBuf, hidTotal);

                    cs.SetInt("seq_len", seqLen);
                    cs.SetBuffer(kRmsHidden, "norm_input", hiddenBuf);
                    cs.SetBuffer(kRmsHidden, "norm_output", normOutBuf);
                    cs.SetBuffer(kRmsHidden, "norm_gamma", weights.Get(lp + "post_attn_ln"));
                    cs.Dispatch(kRmsHidden, Div256(seqLen), 1, 1);
                    Copy(hiddenBuf, normOutBuf, hidTotal);

                    cs.SetInt("intermediate_size", Interm);
                    cs.SetBuffer(kGateUp, "input", hiddenBuf);
                    cs.SetBuffer(kGateUp, "mlp_gate_w", weights.Get(lp + "mlp_gate"));
                    cs.SetBuffer(kGateUp, "mlp_up_w", weights.Get(lp + "mlp_up"));
                    cs.SetBuffer(kGateUp, "mlp_gate_scales", weights.Has(lp + "mlp_gate.scales") ? weights.Get(lp + "mlp_gate.scales") : weights.Get(lp + "mlp_gate"));
                    cs.SetBuffer(kGateUp, "mlp_up_scales", weights.Has(lp + "mlp_up.scales") ? weights.Get(lp + "mlp_up.scales") : weights.Get(lp + "mlp_up"));
                    cs.SetBuffer(kGateUp, "intermediate", mlpInterBuf);
                    cs.Dispatch(kGateUp, (Interm + 63) / 64, (seqLen + 7) / 8, 1);

                    cs.SetBuffer(kDown, "input", hiddenBuf);
                    cs.SetBuffer(kDown, "mlp_down_w", weights.Get(lp + "mlp_down"));
                    cs.SetBuffer(kDown, "mlp_down_scales", weights.Has(lp + "mlp_down.scales") ? weights.Get(lp + "mlp_down.scales") : weights.Get(lp + "mlp_down"));
                    cs.SetBuffer(kDown, "intermediate", mlpInterBuf);
                    cs.Dispatch(kDown, (H + 63) / 64, (seqLen + 7) / 8, 1);

                    AddResidual(hiddenBuf, skipBuf, hidTotal);
                }
                cachedLen += seqLen;
            }

            void DispatchFinalLogits(int seqLen)
            {
                cs.SetInt("buffer_size", H);
                cs.SetInt("copy_src_offset", (seqLen - 1) * H);
                cs.SetInt("copy_dst_offset", 0);
                cs.SetBuffer(kCopyTo, "buf_a", lastHiddenBuf);
                cs.SetBuffer(kCopyTo, "buf_b", hiddenBuf);
                cs.Dispatch(kCopyTo, Div256(H), 1, 1);

                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", H);
                cs.SetFloat("norm_eps", QwenASRConfig.RMS_EPS);
                cs.SetBuffer(kRmsHidden, "norm_input", lastHiddenBuf);
                cs.SetBuffer(kRmsHidden, "norm_output", normSingleBuf);
                cs.SetBuffer(kRmsHidden, "norm_gamma", weights.Get("dec/norm"));
                cs.Dispatch(kRmsHidden, 1, 1, 1);

                DispatchLmHeadSharded();
            }

            // The tied embed/lm_head lives in 16 row-shards (dec/embed_tokens/part_k, fp16,
            // 9496 x H each). One LmHeadPredict1Vec dispatch per shard fills its logit slice.
            void DispatchLmHeadSharded()
            {
                int rowsPerShard = QwenASRConfig.VOCAB_SIZE / 16;
                cs.SetInt("shard_rows", rowsPerShard);
                cs.SetInt("hidden_size", H);
                cs.SetBuffer(kLm1, "lm_input", normSingleBuf);
                cs.SetBuffer(kLm1, "logits_buf", logitsBuf);
                for (int k = 0; k < 16; k++)
                {
                    cs.SetInt("shard_base", k * rowsPerShard);
                    cs.SetBuffer(kLm1, "embed_weights", weights.Get($"dec/embed_tokens/part_{k}"));
                    cs.Dispatch(kLm1, (rowsPerShard + 511) / 512, 1, 1);
                }
            }

            // Shard-aware embedding lookup: one pass per shard; each token is written by exactly
            // the pass whose shard contains it (kernel early-outs otherwise).
            void DispatchEmbedTokens(int[] ids, int seqLen)
            {
                uint[] arr = new uint[seqLen];
                for (int i = 0; i < seqLen; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);

                int rowsPerShard = QwenASRConfig.VOCAB_SIZE / 16;
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", H);
                cs.SetInt("shard_rows", rowsPerShard);
                cs.SetBuffer(kEmbed, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbed, "embed_output", hiddenBuf);
                for (int k = 0; k < 16; k++)
                {
                    cs.SetInt("shard_base", k * rowsPerShard);
                    cs.SetBuffer(kEmbed, "embed_weights", weights.Get($"dec/embed_tokens/part_{k}"));
                    cs.Dispatch(kEmbed, Div256(seqLen * H), 1, 1);
                }
            }

            /// <summary>Prefill prompt (+audio embeds scattered at the contiguous pad run), then
            /// greedy-decode. result: generated ids (excluding prompt), ends at im_end/endoftext.</summary>
            public IEnumerator GreedyDecodeYielding(int[] promptIds, int audioPadStart, int nAudioTokens,
                                                    List<int> generated, int maxNew = 128)
            {
                cachedLen = 0;
                int S = promptIds.Length;
                EnsureDecoderScratch(S);
                if (S + maxNew > cacheCapacity)
                    throw new InvalidOperationException($"prompt {S} + maxNew {maxNew} exceeds KV capacity {cacheCapacity}.");

                DispatchEmbedTokens(promptIds, S);
                // scatter audio embeds over the <|audio_pad|> run (contiguous by construction)
                cs.SetInt("buffer_size", nAudioTokens * H);
                cs.SetInt("copy_src_offset", 0);
                cs.SetInt("copy_dst_offset", audioPadStart * H);
                cs.SetBuffer(kCopyTo, "buf_a", hiddenBuf);
                cs.SetBuffer(kCopyTo, "buf_b", projBuf);
                cs.Dispatch(kCopyTo, Div256(nAudioTokens * H), 1, 1);
                yield return null;

                DispatchDecoderLayers(S);
                DispatchFinalLogits(S);
                yield return null;

                int[] tok = new int[1];
                var rd = ReadArgmaxYielding(tok);
                while (rd.MoveNext()) yield return rd.Current;

                for (int step = 0; step < maxNew; step++)
                {
                    generated.Add(tok[0]);
                    if (tok[0] == QwenASRConfig.IM_END_TOKEN_ID || tok[0] == QwenASRConfig.ENDOFTEXT_TOKEN_ID)
                        yield break;

                    DispatchEmbedTokens(new[] { tok[0] }, 1);
                    DispatchDecoderLayers(1);
                    DispatchFinalLogits(1);
                    rd = ReadArgmaxYielding(tok);
                    while (rd.MoveNext()) yield return rd.Current;
                }
            }

            IEnumerator ReadArgmaxYielding(int[] result)
            {
                cs.SetInt("vocab_size", QwenASRConfig.VOCAB_SIZE);
                cs.SetBuffer(kArgMax, "logits_buf", logitsBuf);
                cs.SetBuffer(kArgMax, "argmax_result", argmaxBuf);
                cs.Dispatch(kArgMax, 1, 1, 1);
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(argmaxBuf);
                    while (!req.done) yield return null;
                    if (!req.hasError)
                    {
                        result[0] = (int)req.GetData<uint>()[0];
                        yield break;
                    }
                }
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                result[0] = (int)r[0];
            }

            public void ResetCache() => cachedLen = 0;

            public void Dispose()
            {
                weights?.Dispose();
                framesBuf?.Release(); dftCos?.Release(); dftSin?.Release(); powerBuf?.Release();
                melBuf?.Release(); gmaxBuf?.Release();
                conv1Buf?.Release(); conv2Buf?.Release(); conv3Buf?.Release();
                encX?.Release(); encNorm?.Release(); encQ?.Release(); encK?.Release(); encV?.Release();
                encScores?.Release(); encAtt?.Release(); encFfn?.Release(); projBuf?.Release(); projTmp?.Release();
                hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release();
                qBuf?.Release(); kBuf?.Release(); vBuf?.Release(); qNormBuf?.Release(); kNormBuf?.Release();
                attendedBuf?.Release(); attnOutBuf?.Release(); mlpInterBuf?.Release();
                lastHiddenBuf?.Release(); normSingleBuf?.Release();
                logitsBuf?.Release(); argmaxBuf?.Release(); tokenIdsBuf?.Release();
                ropeCos?.Release(); ropeSin?.Release();
                if (kCache != null) foreach (var b in kCache) b?.Release();
                if (vCache != null) foreach (var b in vCache) b?.Release();
            }
        }
    }
}
