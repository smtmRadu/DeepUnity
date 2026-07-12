using System;
using System.Collections;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // CausalHiFTGenerator — CosyVoice3's mel(80 @ 50Hz) -> 24kHz waveform vocoder.
        // Port spec: TTS/CosyVoice/SPEC.md §3. Same NSF+iSTFT skeleton as the validated
        // Chatterbox HiFT stage (upsample [8,5,3] k[16,11,7], resblocks k[3,7,11] d[1,3,5]
        // Snake, source resblocks k[7,7,11], iSTFT n_fft16 hop4); kernels shared via
        // CosyVoiceFlowCS. Causal deltas vs the Chatterbox donor:
        //   - conv_pre k5 pads RIGHT (4 mel frames lookahead)  -> Conv1D with pad_left=0
        //   - f0_predictor condnet: k4 RIGHT(3) then 4x k3 LEFT(2), ELU each; Linear+abs
        //   - ups = nearest-neighbor upsample THEN plain Conv1d k, LEFT-pad k-1
        //     (checkpoint layout [out,in,k] — NOT ConvTranspose)
        //   - source_downs: k30 s15 LEFT-pad14 | k6 s3 LEFT-pad2 | k1 pad0
        //   - resblocks fully causal: convs1 LEFT-pad k*d-d, convs2 LEFT-pad k-1
        //   - conv_post k7 LEFT-pad6; NO TrimFade (that was Chatterbox-specific)
        //   - head unchanged: mag=min(exp(x[:9]),1e2), phase=sin(x[9:]), iSTFT, clamp ±0.99
        // NSF source = SineGen2; generated with the CumsumPhase/SineMerge kernels (Chatterbox
        // semantics — exact SineGen2 deltas verified at A4; parity probes inject the reference
        // source dump, which also removes the phase/noise randomness).
        public class HiFTVocoder : IDisposable
        {
            const int MEL = CosyVoiceConfig.MEL_DIM;

            readonly ComputeShader cs;
            readonly CosyVoiceWeights weights;

            int kLinear, kConv, kSnake, kActivate, kRepeat, kGauss, kCumsum, kSineMerge,
                kSTFT, kMagPhase, kISTFT, kZero, kCopy, kCopySlice, kAdd, kScale;

            ComputeBuffer f0A, f0B, f0Buf, f0UpBuf, thetaBuf, noiseBuf, phaseVecBuf, srcBuf, sstftBuf;
            ComputeBuffer vA, vB, vC, vD, rbT1, rbT2, rbAcc, wavBuf;
            ComputeBuffer srcWinBuf, melWinBuf;   // A6-max windowed-streaming slices
            int curTg;

            /// <summary>A6-max: streaming chunks re-vocode only a mel window
            /// [emittedFrames - VOC_STREAM_OVERLAP_MEL, end) through the heavy main branch,
            /// instead of the full history (2.62x tax). F0 + the NSF source stay GLOBAL (the
            /// cumsum phase must be continuous across chunks) and are sliced into the window,
            /// so the only delta vs full re-vocode is the window's left-edge conv zero-pads —
            /// which decay within the overlap, before the first NEW emitted sample. false =
            /// legacy full re-vocode. Offline calls (speechOffset 0 + finalize) are bit-exact
            /// either way.</summary>
            public bool WindowedStreaming = true;
            float[] streamPhases;   // NSF per-harmonic phases, fixed per utterance (reference
                                    // carries a source cache across chunks; re-rolling them per
                                    // chunk would decorrelate the overlap band)
            float[] pendingTail;    // seam cross-fade: samples the previous chunk held back
            int pendingLen;         // 0 or VOC_STREAM_FADE
            /// <summary>Seams cross-faded this utterance (reset at speechOffset 0) — probe
            /// evidence that the fade engaged on every boundary incl. first and finalize.</summary>
            public int SeamsBlended { get; private set; }

            // ---- parity-probe injection hooks (null = generate on GPU) -----------------------
            /// <summary>Reference NSF source [S] — bypasses F0-upsample/CumsumPhase/SineMerge.</summary>
            public float[] InjectSource;
            /// <summary>Per-harmonic random phases [9] (h0 must be 0).</summary>
            public float[] InjectNsfPhases;
            /// <summary>N(0,1) noise [S*9] for the NSF noise branch.</summary>
            public float[] InjectNsfNoise;
            /// <summary>Predicted F0 [Tg] (mel rate, after abs) — parity readback.</summary>
            public ComputeBuffer DebugF0 => f0Buf;
            /// <summary>Per-stage tap (name, buffer, elemCount) for the bisecting parity probe;
            /// buffers are [T,C]. Null in normal use — taps force GPU syncs.</summary>
            public Action<string, ComputeBuffer, int> DebugTap;

            public float VocoderMs { get; private set; }

            public HiFTVocoder(CosyVoiceWeights weights)
            {
                this.weights = weights;
                cs = DeepUnityMeta.CosyVoiceFlowCS;
                kLinear = cs.FindKernel("LinearBias");
                kConv = cs.FindKernel("Conv1D");
                kSnake = cs.FindKernel("SnakeAct");
                kActivate = cs.FindKernel("Activate");
                kRepeat = cs.FindKernel("RepeatTime");
                kGauss = cs.FindKernel("GaussNoise");
                kCumsum = cs.FindKernel("CumsumPhaseHold");   // SineGen2 staircase phase (SPEC §3)
                kSineMerge = cs.FindKernel("SineMerge");
                kSTFT = cs.FindKernel("STFT16");
                kMagPhase = cs.FindKernel("MagPhase");
                kISTFT = cs.FindKernel("ISTFT16");
                kZero = cs.FindKernel("ZeroBuffer");
                kCopy = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAdd = cs.FindKernel("AddResidual");
                kScale = cs.FindKernel("ScaleBuf");
                phaseVecBuf = new ComputeBuffer(9, 4, ComputeBufferType.Structured);
            }

            static int Div256(int n) => (n + 255) / 256;

            static void Grow(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            void EnsureScratch(int Tg)
            {
                if (Tg <= curTg) return;
                curTg = Tg;
                int S = Tg * CosyVoiceConfig.SAMPLES_PER_MEL_FRAME;
                // vA/vB must hold the nearest-upsample INTERMEDIATES (output length at INPUT
                // channel count — unlike the ConvTranspose donor): 8Tg,512 | 40Tg,256 | 120Tg,128
                // -> worst 15360*Tg. The resblock/fusion track peaks at (120Tg+1)*64.
                int voc = 15360 * Tg;
                int rbMax = (120 * Tg + 1) * 64;
                Grow(ref f0A, Tg * 512); Grow(ref f0B, Tg * 512); Grow(ref f0Buf, Tg);
                Grow(ref f0UpBuf, S);
                Grow(ref thetaBuf, S * 9);
                Grow(ref noiseBuf, S * 9);
                Grow(ref srcBuf, S);
                Grow(ref sstftBuf, (S / CosyVoiceConfig.ISTFT_HOP + 1) * 18);
                Grow(ref vA, voc); Grow(ref vB, voc); Grow(ref vC, rbMax); Grow(ref vD, rbMax);
                Grow(ref rbT1, rbMax); Grow(ref rbT2, rbMax); Grow(ref rbAcc, rbMax);
                Grow(ref wavBuf, S);
                Grow(ref srcWinBuf, S);
                Grow(ref melWinBuf, Tg * MEL);
            }

            // ---------------- generic op helpers ([T, C] layout everywhere) ----------------------
            void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim, int act = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", 1);
                cs.SetBuffer(kLinear, "X", x);
                cs.SetBuffer(kLinear, "W", weights.Get(name + ".weight"));
                cs.SetBuffer(kLinear, "W_bias", weights.Get(name + ".bias"));
                cs.SetBuffer(kLinear, "Y", y);
                cs.Dispatch(kLinear, 1, (T + 7) / 8, (outDim + 31) / 32);
            }

            void Conv(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
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
                cs.SetBuffer(kConv, "W", weights.Get(name + ".weight"));
                cs.SetBuffer(kConv, "W_bias", weights.Get(name + ".bias"));
                cs.SetBuffer(kConv, "Y", y);
                cs.Dispatch(kConv, Div256(outLen * outCh), 1, 1);
            }

            void SnakeOp(string alphaName, ComputeBuffer buf, int T, int ch)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", ch);
                cs.SetBuffer(kSnake, "inout_buf", buf);
                cs.SetBuffer(kSnake, "snake_alpha", weights.Get(alphaName));
                cs.Dispatch(kSnake, Div256(T * ch), 1, 1);
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

            void RepeatOp(ComputeBuffer src, ComputeBuffer dst, int outT, int ch, int factor)
            {
                cs.SetInt("seq_len", outT); cs.SetInt("in_dim", ch); cs.SetInt("factor", factor);
                cs.SetBuffer(kRepeat, "X", src); cs.SetBuffer(kRepeat, "Y", dst);
                cs.Dispatch(kRepeat, Div256(outT * ch), 1, 1);
            }

            // Fully-causal ResBlock: 3x x += conv2_j(snake2(conv1_j(snake1(x)))), conv1 dilated
            // LEFT-pad k*d-d, conv2 LEFT-pad k-1. Scratch = rbT1/rbT2/rbAcc only, so x/outSum
            // may be any of the vA..vD buffers.
            void ResBlock(string p, ComputeBuffer x, ComputeBuffer outSum, int T, int ch, int kernel, bool accumulate)
            {
                CopyOp(rbAcc, x, T * ch);
                for (int j = 0; j < 3; j++)
                {
                    int dil = CosyVoiceConfig.RESBLOCK_DILATIONS[j];
                    CopyOp(rbT1, rbAcc, T * ch);
                    SnakeOp(p + $".activations1.{j}.alpha", rbT1, T, ch);
                    Conv(p + $".convs1.{j}", rbT1, rbT2, T, T, ch, ch, kernel, 1, dil, kernel * dil - dil);
                    SnakeOp(p + $".activations2.{j}.alpha", rbT2, T, ch);
                    Conv(p + $".convs2.{j}", rbT2, rbT1, T, T, ch, ch, kernel, 1, 1, kernel - 1);
                    AddOp(rbAcc, rbT1, T * ch);
                }
                if (accumulate) AddOp(outSum, rbAcc, T * ch);
                else CopyOp(outSum, rbAcc, T * ch);
            }

            /// <summary>melBuf [Tg, 80] ([T,C] layout) -> 24kHz waveform; onWav receives exactly
            /// Tg*480 mono float samples. Offline path (finalize semantics).</summary>
            public IEnumerator VocodeYielding(ComputeBuffer melBuf, int Tg, Action<float[]> onWav, int noiseSeed = 0)
                => VocodeChunkYielding(melBuf, Tg, true, 0, onWav, noiseSeed);

            /// <summary>Streaming vocode over the FULL mel history [M, 80] (CosyVoice3Model.token2wav
            /// recipe): finalize=false trims the causal lookaheads (f0 3 mel frames, conv_pre 4,
            /// istft tail 480 samples); onNewSamples receives only wav[speechOffset..]. The caller
            /// advances speechOffset by the returned count each chunk.</summary>
            public IEnumerator VocodeChunkYielding(ComputeBuffer melBuf, int M, bool finalize,
                                                   int speechOffset, Action<float[]> onNewSamples, int noiseSeed = 0)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                EnsureScratch(M);
                int Fg = finalize ? M : M - CosyVoiceConfig.PRE_LOOKAHEAD_LEN;        // f0/source frames
                int S = Fg * CosyVoiceConfig.SAMPLES_PER_MEL_FRAME;
                // A6-max window: the heavy main branch (conv_pre -> ups -> resblocks -> iSTFT)
                // runs only on mel rows [Wf, Fg). F0 + NSF source stay GLOBAL below (phase
                // continuity), so the window's only delta vs a full re-vocode is its left-edge
                // conv zero-pads — decayed within VOC_STREAM_OVERLAP_MEL, before the first NEW
                // emitted sample. Offline calls (speechOffset 0) keep Wf = 0 -> bit-exact.
                int Wf = (WindowedStreaming && speechOffset > 0)
                    ? Math.Max(0, speechOffset / CosyVoiceConfig.SAMPLES_PER_MEL_FRAME - CosyVoiceConfig.VOC_STREAM_OVERLAP_MEL)
                    : 0;
                int TgW = Fg - Wf;                                                    // windowed mel frames
                int SW = TgW * CosyVoiceConfig.SAMPLES_PER_MEL_FRAME;
                int Vg = finalize ? TgW : TgW - CosyVoiceConfig.CONV_PRE_LOOK_RIGHT;  // main-branch frames (window-local)

                // ---- F0 predictor: k4 RIGHT-lookahead(3) -> 4x k3 causal, ELU each; Linear + abs.
                // Reference runs this in float64; fp32 GPU gated by the parity probe (SPEC §3).
                // Non-finalize: out rows [0, M-3) read only REAL mel rows (== the context-split path).
                Conv("hift/f0_predictor.condnet.0", melBuf, f0A, Fg, M, MEL, 512, 4, 1, 1, 0, act: 7);
                for (int j = 1; j < 5; j++)
                {
                    Conv($"hift/f0_predictor.condnet.{2 * j}", f0A, f0B, Fg, Fg, 512, 512, 3, 1, 1, 2, act: 7);
                    (f0A, f0B) = (f0B, f0A);
                }
                Linear("hift/f0_predictor.classifier", f0A, f0Buf, Fg, 512, 1, act: 5);
                yield return null;

                // ---- NSF source (SineGen2), or the injected reference source for parity
                if (InjectSource != null)
                {
                    srcBuf.SetData(InjectSource, 0, 0, Math.Min(InjectSource.Length, S));
                }
                else
                {
                    RepeatOp(f0Buf, f0UpBuf, S, 1, CosyVoiceConfig.SAMPLES_PER_MEL_FRAME);

                    cs.SetInt("sample_len", S);
                    cs.SetInt("factor", CosyVoiceConfig.SAMPLES_PER_MEL_FRAME);
                    cs.SetBuffer(kCumsum, "f0_up", f0UpBuf);
                    cs.SetBuffer(kCumsum, "theta_out", thetaBuf);
                    cs.Dispatch(kCumsum, 1, 1, 1);

                    // SineGen2 rand_ini: fixed per-harmonic phase offsets (h0 = 0), 2*pi*U[0,1).
                    // Rolled ONCE per utterance (at speechOffset 0) and held across streaming
                    // chunks — per-chunk re-rolls would decorrelate the source at chunk seams
                    // (the reference carries a source cache for the same reason).
                    float[] ph = InjectNsfPhases;
                    if (ph == null)
                    {
                        if (speechOffset == 0 || streamPhases == null)
                        {
                            streamPhases = new float[9];
                            for (int h = 1; h < 9; h++) streamPhases[h] = UnityEngine.Random.Range(0f, 2f * Mathf.PI);
                        }
                        ph = streamPhases;
                    }
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
                    cs.SetFloat("sine_amp", CosyVoiceConfig.NSF_ALPHA);
                    cs.SetFloat("noise_std", CosyVoiceConfig.NSF_SIGMA);
                    cs.SetFloat("voiced_threshold", CosyVoiceConfig.NSF_VOICED_THRESHOLD);
                    cs.SetBuffer(kSineMerge, "f0_up", f0UpBuf);
                    cs.SetBuffer(kSineMerge, "theta_in", thetaBuf);
                    cs.SetBuffer(kSineMerge, "phase_vec", phaseVecBuf);
                    cs.SetBuffer(kSineMerge, "nsf_noise", noiseBuf);
                    cs.SetBuffer(kSineMerge, "nsf_w", weights.Get("hift/m_source.l_linear.weight"));
                    cs.SetBuffer(kSineMerge, "nsf_b", weights.Get("hift/m_source.l_linear.bias"));
                    cs.SetBuffer(kSineMerge, "Y", srcBuf);
                    cs.Dispatch(kSineMerge, Div256(S), 1, 1);
                }
                yield return null;

                // ---- windowed views for the main branch: mel rows [Wf, Fg), source samples
                // [Wf*480, S) — the source keeps its global phase, only the view is sliced
                CopySliceOp(melWinBuf, 0, melBuf, Wf * MEL, TgW * MEL);
                CopySliceOp(srcWinBuf, 0, srcBuf, Wf * CosyVoiceConfig.SAMPLES_PER_MEL_FRAME, SW);

                // ---- source spectrum: [n_frames, 18] (window-local; the STFT center-reflect at
                // the window start is a left-edge artifact the overlap absorbs)
                int nFrames = SW / CosyVoiceConfig.ISTFT_HOP + 1;
                cs.SetInt("n_frames", nFrames);
                cs.SetInt("sample_len", SW);
                cs.SetBuffer(kSTFT, "X", srcWinBuf);
                cs.SetBuffer(kSTFT, "Y", sstftBuf);
                cs.Dispatch(kSTFT, Div256(nFrames * 9), 1, 1);
                DebugTap?.Invoke("sstft", sstftBuf, nFrames * 18);
                yield return null;

                // ---- main branch: conv_pre k5 RIGHT-pad 4 (out len = Tg)
                int[] ups = CosyVoiceConfig.UPSAMPLE_RATES;
                int[] ker = CosyVoiceConfig.UPSAMPLE_KERNELS;
                int[] chs = { 512, 256, 128, 64 };
                int[] sdK = { 30, 6, 1 }; int[] sdS = { 15, 3, 1 }; int[] sdPL = { 14, 2, 0 };

                Conv("hift/conv_pre", melWinBuf, vA, Vg, TgW, MEL, 512, 5, 1, 1, 0);
                DebugTap?.Invoke("conv_pre", vA, Vg * 512);
                int curLen = Vg;
                for (int i = 0; i < 3; i++)
                {
                    ActivateOp(vA, curLen * chs[i], 4, CosyVoiceConfig.LRELU_SLOPE);

                    // CausalConv1dUpsample: nearest-neighbor x stride, then conv k LEFT-pad k-1
                    int outLen = curLen * ups[i];
                    RepeatOp(vA, vB, outLen, chs[i], ups[i]);
                    curLen = outLen;
                    Conv($"hift/ups.{i}", vB, vA, curLen, curLen, chs[i], chs[i + 1], ker[i], 1, 1, ker[i] - 1);
                    DebugTap?.Invoke($"up{i}", vA, curLen * chs[i + 1]);

                    if (i == 2)
                    {
                        // ReflectionPad1d((1,0)) on [T,C]: prepend row 1
                        CopySliceOp(vB, chs[3], vA, 0, curLen * chs[3]);
                        CopySliceOp(vB, 0, vA, chs[3], chs[3]);
                        curLen += 1;
                        (vA, vB) = (vB, vA);
                    }

                    // fusion: x += source_resblock_i(source_down_i(s_stft))
                    Conv($"hift/source_downs.{i}", sstftBuf, vB, curLen, nFrames, 18, chs[i + 1], sdK[i], sdS[i], 1, sdPL[i]);
                    DebugTap?.Invoke($"sdown{i}", vB, curLen * chs[i + 1]);
                    ResBlock($"hift/source_resblocks.{i}", vB, vC, curLen, chs[i + 1],
                             CosyVoiceConfig.SOURCE_RESBLOCK_KERNELS[i], accumulate: false);
                    DebugTap?.Invoke($"srb{i}", vC, curLen * chs[i + 1]);
                    AddOp(vA, vC, curLen * chs[i + 1]);
                    yield return null;

                    // x = mean of 3 causal resblocks (kernels 3,7,11)
                    ZeroOp(vD, curLen * chs[i + 1]);
                    for (int j = 0; j < 3; j++)
                    {
                        ResBlock($"hift/resblocks.{i * 3 + j}", vA, vD, curLen, chs[i + 1],
                                 CosyVoiceConfig.RESBLOCK_KERNELS[j], accumulate: true);
                        DebugTap?.Invoke($"rb{i * 3 + j}", rbAcc, curLen * chs[i + 1]);
                        yield return null;
                    }
                    CopyOp(vA, vD, curLen * chs[i + 1]);
                    ScaleOp(vA, curLen * chs[i + 1], 1f / 3f);
                }

                // conv_post: leaky(0.01) -> k7 LEFT-pad6 -> 18ch, mag/phase -> iSTFT -> clamp.
                // curLen = S/4 + 1 here (reflection pad), so (curLen-1)*4 = S samples exactly.
                ActivateOp(vA, curLen * 64, 4, 0.01f);
                Conv("hift/conv_post", vA, sstftBuf, curLen, curLen, 64, 18, 7, 1, 1, 6);
                DebugTap?.Invoke("conv_post", sstftBuf, curLen * 18);

                cs.SetInt("n_frames", curLen);
                cs.SetBuffer(kMagPhase, "inout_buf", sstftBuf);
                cs.Dispatch(kMagPhase, Div256(curLen * 9), 1, 1);

                int outSamples = (curLen - 1) * CosyVoiceConfig.ISTFT_HOP;
                cs.SetInt("sample_len", outSamples);
                cs.SetInt("n_frames", curLen);
                cs.SetFloat("audio_limit", CosyVoiceConfig.AUDIO_LIMIT);
                cs.SetBuffer(kISTFT, "X", sstftBuf);
                cs.SetBuffer(kISTFT, "Y", wavBuf);
                cs.Dispatch(kISTFT, Div256(outSamples), 1, 1);
                yield return null;

                if (!finalize)   // trim the istft lookahead tail (prod(ups)*hop samples)
                    outSamples -= CosyVoiceConfig.SAMPLES_PER_MEL_FRAME;
                // wavBuf is window-local: sample i is GLOBAL sample Wf*480 + i
                int winBase = Wf * CosyVoiceConfig.SAMPLES_PER_MEL_FRAME;
                int newCount = Math.Max(winBase + outSamples - speechOffset, 0);
                float[] wav = new float[newCount];
                if (newCount > 0)
                {
                    // async readback (Phase 4): the old blocking GetData was a hard fence on the
                    // whole vocoder tail — under the TTS overlap pump these yields are filled by
                    // LM decode. Values identical; spin-cap keeps editor probes deterministic.
                    // Cap 2000 (Phase 5): ~200 tokens of patience under the budgeted pump (~10
                    // spins/token); drains burn the spins in ms then hard-wait as intended.
                    bool got = false;
                    if (SystemInfo.supportsAsyncGPUReadback)
                    {
                        var req = UnityEngine.Rendering.AsyncGPUReadback.Request(
                            wavBuf, newCount * 4, (speechOffset - winBase) * 4);
                        int spins = 0;
                        while (!req.done)
                        {
                            if (++spins > 2000) { req.WaitForCompletion(); break; }
                            yield return null;
                        }
                        if (!req.hasError)
                        {
                            req.GetData<float>().CopyTo(wav);
                            got = true;
                        }
                    }
                    if (!got)
                        wavBuf.GetData(wav, 0, speechOffset - winBase, newCount);
                }

                // ---- seam cross-fade (windowed streaming only; offline stays bit-exact) --------
                // Each non-finalize chunk holds back the last VOC_STREAM_FADE samples; the next
                // chunk blends its recomputation over them with w(0)=0, so its first emitted
                // sample continues the previous vocode EXACTLY — no seam discontinuity survives,
                // whatever its source. TTS speechOffset bookkeeping is untouched (it advances by
                // emitted counts, and pendingTail always covers [speechOffset, +FADE)).
                float[] emit = wav;
                bool streamSeq = WindowedStreaming && (!finalize || speechOffset > 0);
                if (speechOffset == 0) { pendingLen = 0; SeamsBlended = 0; }   // new utterance
                if (streamSeq)
                {
                    if (pendingLen > 0 && newCount > 0)
                    {
                        int B = Math.Min(pendingLen, newCount);
                        for (int j = 0; j < B; j++)
                        {
                            float w = 0.5f - 0.5f * Mathf.Cos(Mathf.PI * j / pendingLen);
                            wav[j] = pendingTail[j] * (1f - w) + wav[j] * w;
                        }
                        pendingLen = 0;
                        SeamsBlended++;
                    }
                    if (!finalize && newCount > CosyVoiceConfig.VOC_STREAM_FADE)
                    {
                        pendingLen = CosyVoiceConfig.VOC_STREAM_FADE;
                        if (pendingTail == null || pendingTail.Length < pendingLen)
                            pendingTail = new float[pendingLen];
                        int emitCount = newCount - pendingLen;
                        Array.Copy(wav, emitCount, pendingTail, 0, pendingLen);
                        emit = new float[emitCount];
                        Array.Copy(wav, 0, emit, 0, emitCount);
                    }
                }
                sw.Stop();
                VocoderMs = (float)sw.Elapsed.TotalMilliseconds;
                onNewSamples?.Invoke(emit);
            }

            public void Dispose()
            {
                f0A?.Release(); f0B?.Release(); f0Buf?.Release(); f0UpBuf?.Release();
                thetaBuf?.Release(); noiseBuf?.Release(); phaseVecBuf?.Release();
                srcBuf?.Release(); sstftBuf?.Release();
                vA?.Release(); vB?.Release(); vC?.Release(); vD?.Release();
                rbT1?.Release(); rbT2?.Release(); rbAcc?.Release(); wavBuf?.Release();
                srcWinBuf?.Release(); melWinBuf?.Release();
            }
        }
    }
}
