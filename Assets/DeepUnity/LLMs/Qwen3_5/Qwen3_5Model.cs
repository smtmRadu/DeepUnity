using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // Forward orchestration for Qwen3.5 (text-only, FP16). Branches per layer between
        // a standard GQA + attn-output-gate full attention and a Gated DeltaNet linear layer.
        public class Qwen3_5Model : IDisposable
        {
            // FlashAttention-1 fused path for the full-attention layers (scores+mask+softmax+AV
            // in one dispatch, online softmax, no materialized score matrix; DeltaNet layers are
            // unaffected). Static so probes can A/B it against the legacy 4-dispatch path; the
            // kernel requires head_dim <= 256 — DispatchFullAttention falls back to legacy otherwise.
            public static bool UseFlashAttention = true;

            // Weight format this instance runs (set in the ctor; INT8 enables the INT8_WEIGHTS
            // shader keyword and binds the per-row scale buffers next to each weight buffer).
            public readonly LLMQuant Quant;

            ComputeShader cs;

            // Kernel ids (cached)
            int kEmbed, kRmsHidden, kRmsHead;
            int kQProjGated, kKProj, kVProj, kOProj;
            int kRopePartial, kWriteCache, kMaskCausal, kSoftmax, kAttnScores, kAttend, kAttnGate, kFlashAttn;
            int kInProjQKV, kInProjZ, kInProjA, kInProjB;
            int kConvUpdate, kConvPrefill, kSplitConv;
            int kL2NormHead, kDeltaNet, kRMSGated, kLinearOut;
            int kGateUp, kDown, kGateUp1, kDown1;
            int kLm, kLm1, kArgMax, kSample;
            int kZero, kCopy, kCopySlice, kAddRes;
            int kApplyPenalty, kMarkSeen, kZeroSeen;

            public Qwen3_5Weights weights;
            public Qwen3_5Cache cache;

            // RoPE caches (rot_dim/2 cos/sin per pos), packed FP16
            ComputeBuffer ropeCos, ropeSin;

            // Common scratch
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf, attnOutBuf, mlpInterBuf;
            ComputeBuffer logitsBuf, probsBuf, argmaxBuf, tokenIdsBuf;
            ComputeBuffer tokenSeenBuf; // [vocab] occurrence counts of generated tokens, for presence/repetition penalties
            ComputeBuffer lastHiddenBuf, normSingleBuf;

            // Full-attention scratch
            ComputeBuffer qBuf, kBuf, vBuf, gateBuf;
            ComputeBuffer qNormBuf, kNormBuf;
            ComputeBuffer attnScoresBuf, attendedBuf;

            // Linear-attention scratch
            ComputeBuffer linearQkvBuf;     // [seq, conv_dim]
            ComputeBuffer linearZBuf;       // [seq, value_dim]
            ComputeBuffer linearABuf;       // [seq, num_v_heads]
            ComputeBuffer linearBBuf;       // [seq, num_v_heads]
            ComputeBuffer linearQBuf, linearKBuf, linearVBuf;             // post-split
            ComputeBuffer linearQNormBuf, linearKNormBuf;                 // post L2 norm
            ComputeBuffer linearYBuf, linearYNormBuf;                     // delta-net out + gated norm

            int curSeqAlloc, curKvAlloc;

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int intermediate, vocab, ropeRotDim;
            readonly int convDim, convKernel, keyDim, valueDim, numVHeads, headKDim, headVDim;
            readonly float rmsEps, attnScale;
            readonly int cacheCapacity;
            readonly Qwen3_5LayerType[] layerTypes;

            // RoPE table is computed on a background thread and uploaded when ready, so it never blocks
            // the construction frame. Both must be ready before the first forward.
            volatile bool ropeReady;
            volatile bool _ropeComputed;
            uint[] _ropeCosData, _ropeSinData;

            public bool IsReady => weights.IsReady && ropeReady;

            public Qwen3_5Model(string paramsPath, int cacheCapacity, LLMQuant quant = LLMQuant.FP16)
            {
                Quant = quant;
                hiddenSize = Qwen3_5Config.HIDDEN_SIZE;
                headDim = Qwen3_5Config.HEAD_DIM;
                headsQ = Qwen3_5Config.HEADS_Q;
                headsKV = Qwen3_5Config.HEADS_KV;
                intermediate = Qwen3_5Config.MLP_INTERMEDIATE_SIZE;
                vocab = Qwen3_5Config.VOCAB_SIZE;
                numLayers = Qwen3_5Config.NUM_LAYERS;
                ropeRotDim = Qwen3_5Config.ROTATED_DIMS; // 64
                rmsEps = Qwen3_5Config.RMS_EPS;
                attnScale = MathF.Pow(headDim, -0.5f);
                this.cacheCapacity = cacheCapacity;
                layerTypes = Qwen3_5Config.layer_types;

                convDim    = Qwen3_5Config.LINEAR_CONV_DIM;
                convKernel = Qwen3_5Config.LINEAR_CONV_KERNEL_DIM;
                keyDim     = Qwen3_5Config.LINEAR_KEY_DIM;
                valueDim   = Qwen3_5Config.LINEAR_VALUE_DIM;
                numVHeads  = Qwen3_5Config.LINEAR_NUM_VALUE_HEADS;
                headKDim   = Qwen3_5Config.LINEAR_KEY_HEAD_DIM;
                headVDim   = Qwen3_5Config.LINEAR_VALUE_HEAD_DIM;

                var swPhase = System.Diagnostics.Stopwatch.StartNew();
                cs = DeepUnityMeta.Qwen3_5CS;
                // Keyword state lives on the shared shader asset — one quant mode per session;
                // don't run two differently-quantized Qwen instances simultaneously.
                if (quant == LLMQuant.INT8) cs.EnableKeyword("INT8_WEIGHTS"); else cs.DisableKeyword("INT8_WEIGHTS");
                if (quant == LLMQuant.INT4) cs.EnableKeyword("INT4_WEIGHTS"); else cs.DisableKeyword("INT4_WEIGHTS");
                CacheKernelIds();
                double tKernels = swPhase.Elapsed.TotalMilliseconds;

                weights = new Qwen3_5Weights(paramsPath, quant); // sets weights.allocMs internally

                swPhase.Restart();
                cache = new Qwen3_5Cache(
                    cacheCapacity, layerTypes,
                    headsKV, headDim,
                    convDim, convKernel,
                    numVHeads, headKDim, headVDim);
                double tCache = swPhase.Elapsed.TotalMilliseconds;

                swPhase.Restart();
                PrecomputeRoPEAsync();   // background compute + async upload; ~0 blocking here
                double tRope = swPhase.Elapsed.TotalMilliseconds;

                swPhase.Restart();
                probsBuf = new ComputeBuffer(vocab, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                tokenSeenBuf = new ComputeBuffer(vocab, 4, ComputeBufferType.Structured); // per-token occurrence counts for penalties
                double tScratch = swPhase.Elapsed.TotalMilliseconds;

                // Surface per-phase boot timings to the weights object; the single consolidated
                // "model booted up" log (in Qwen3_5ForCausalLM.InitializeChat) reads them.
                weights.bootKernelsMs = tKernels;
                weights.bootCacheMs   = tCache;
                weights.bootRopeMs    = tRope;
                weights.bootScratchMs = tScratch;
            }

            void CacheKernelIds()
            {
                kEmbed       = cs.FindKernel("EmbeddingLookup");
                kRmsHidden   = cs.FindKernel("RmsNormHidden");
                kRmsHead     = cs.FindKernel("RmsNormHead");
                kQProjGated  = cs.FindKernel("QProjGated");
                kKProj       = cs.FindKernel("KProj");
                kVProj       = cs.FindKernel("VProj");
                kOProj       = cs.FindKernel("OProj");
                kRopePartial = cs.FindKernel("ApplyRopePartial");
                kWriteCache  = cs.FindKernel("WriteCacheFull");
                kMaskCausal  = cs.FindKernel("ApplyMaskCausal");
                kSoftmax     = cs.FindKernel("SoftmaxRows");
                kAttnScores  = cs.FindKernel("ComputeAttentionScores");
                kAttend      = cs.FindKernel("AttendValues");
                kFlashAttn   = cs.FindKernel("FlashAttention");
                kAttnGate    = cs.FindKernel("ApplyAttnGate");
                kInProjQKV   = cs.FindKernel("LinearInProjQKV");
                kInProjZ     = cs.FindKernel("LinearInProjZ");
                kInProjA     = cs.FindKernel("LinearInProjA");
                kInProjB     = cs.FindKernel("LinearInProjB");
                kConvUpdate  = cs.FindKernel("CausalConv1DUpdate");
                kConvPrefill = cs.FindKernel("CausalConv1DPrefill");
                kSplitConv   = cs.FindKernel("SplitConvOutQKV");
                kL2NormHead  = cs.FindKernel("L2NormPerHead");
                kDeltaNet    = cs.FindKernel("DeltaNetRecurrent");
                kRMSGated    = cs.FindKernel("RMSNormGated");
                kLinearOut   = cs.FindKernel("LinearOutProj");
                kGateUp      = cs.FindKernel("GateUp");
                kDown        = cs.FindKernel("Down");
                kGateUp1     = cs.FindKernel("GateUp1Vec");
                kDown1       = cs.FindKernel("Down1Vec");
                kLm          = cs.FindKernel("LmHeadPredict");
                kLm1         = cs.FindKernel("LmHeadPredict1Vec");
                kArgMax      = cs.FindKernel("ArgMax");
                kSample      = cs.FindKernel("SampleToken");
                kZero        = cs.FindKernel("ZeroBuffer");
                kCopy        = cs.FindKernel("CopyBuffer");
                kCopySlice   = cs.FindKernel("CopySlice");
                kAddRes      = cs.FindKernel("AddResidual");
                kApplyPenalty = cs.FindKernel("ApplyRepetitionPresencePenalty");
                kMarkSeen     = cs.FindKernel("MarkSampledTokenSeen");
                kZeroSeen     = cs.FindKernel("ZeroTokenSeen");
            }

            // Allocates the RoPE buffers on the main thread (cheap), computes the cos/sin table on a
            // background thread (pure managed math), then uploads it via a main-thread coroutine when ready.
            // Keeps the ~hundreds-of-ms compute off the construction frame.
            void PrecomputeRoPEAsync()
            {
                int maxSeq = Mathf.Max(cacheCapacity, 8192);
                int rd2 = ropeRotDim / 2; // 32
                int rotDim = ropeRotDim;
                float theta = Qwen3_5Config.ROPE_THETA;
                int packedLen = (maxSeq * rd2) / 2;

                ropeCos = new ComputeBuffer(packedLen, 4, ComputeBufferType.Structured);
                ropeSin = new ComputeBuffer(packedLen, 4, ComputeBufferType.Structured);

                var sw = System.Diagnostics.Stopwatch.StartNew();
                _ = Task.Run(() =>
                {
                    // Inverse frequencies depend only on i (not pos) — computed once. (Base is head_dim,
                    // matching HF partial-rotation RoPE.) Each pos owns its own run of packed uints.
                    float[] invFreq = new float[rd2];
                    for (int i = 0; i < rd2; i++)
                        invFreq[i] = 1f / MathF.Pow(theta, 2f * i / rotDim);

                    uint[] c = new uint[packedLen];
                    uint[] s = new uint[packedLen];
                    for (int pos = 0; pos < maxSeq; pos++)
                    {
                        int baseU = pos * (rd2 / 2);
                        for (int j = 0; j < rd2 / 2; j++)
                        {
                            float a0 = pos * invFreq[2 * j];
                            float a1 = pos * invFreq[2 * j + 1];
                            c[baseU + j] = (uint)F32ToF16(MathF.Cos(a0)) | ((uint)F32ToF16(MathF.Cos(a1)) << 16);
                            s[baseU + j] = (uint)F32ToF16(MathF.Sin(a0)) | ((uint)F32ToF16(MathF.Sin(a1)) << 16);
                        }
                    }
                    _ropeCosData = c;
                    _ropeSinData = s;
                    _ropeComputed = true;
                });

                DeepUnityDispatcher.Run(UploadRopeWhenReady(sw));
            }

            System.Collections.IEnumerator UploadRopeWhenReady(System.Diagnostics.Stopwatch sw)
            {
                while (!_ropeComputed) yield return null;   // wait for the background compute (overlaps upload)
                ropeCos.SetData(_ropeCosData);
                ropeSin.SetData(_ropeSinData);
                _ropeCosData = null; _ropeSinData = null;
                weights.ropeAsyncMs = sw.Elapsed.TotalMilliseconds; // compute + upload wall time (async, non-blocking)
                ropeReady = true;
            }

            // Managed IEEE-754 float32 -> float16 (round to nearest). Replaces Unity's Mathf.FloatToHalf,
            // whose per-call native overhead made RoPE precompute the largest blocking item at boot.
            static ushort F32ToF16(float value)
            {
                int i = BitConverter.SingleToInt32Bits(value);
                int s = (i >> 16) & 0x00008000;
                int e = ((i >> 23) & 0x000000ff) - (127 - 15);
                int m = i & 0x007fffff;
                if (e <= 0)
                {
                    if (e < -10) return (ushort)s;          // underflow -> signed zero
                    m |= 0x00800000;
                    int t = 14 - e;
                    int a = (1 << (t - 1)) - 1;
                    int b = (m >> t) & 1;
                    m = (m + a + b) >> t;                    // round to nearest even
                    return (ushort)(s | m);
                }
                else if (e == 0xff - (127 - 15))
                {
                    if (m == 0) return (ushort)(s | 0x7c00); // inf
                    m >>= 13;
                    return (ushort)(s | 0x7c00 | m | (m == 0 ? 1 : 0)); // nan
                }
                else
                {
                    m = m + 0x00000fff + ((m >> 13) & 1);    // round to nearest even
                    if ((m & 0x00800000) != 0) { m = 0; e += 1; }
                    if (e > 30) return (ushort)(s | 0x7c00); // overflow -> inf
                    return (ushort)(s | (e << 10) | (m >> 13));
                }
            }

            void Realloc(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            void EnsureScratch(int seqLen, int totalKvLen)
            {
                if (seqLen <= curSeqAlloc && totalKvLen <= curKvAlloc) return;
                int sL = Math.Max(seqLen, curSeqAlloc);
                int kL = Math.Max(totalKvLen, curKvAlloc);

                Realloc(ref hiddenBuf,    sL * hiddenSize);
                Realloc(ref skipBuf,      sL * hiddenSize);
                Realloc(ref normOutBuf,   sL * hiddenSize);
                Realloc(ref attnOutBuf,   sL * hiddenSize);
                Realloc(ref mlpInterBuf,  sL * intermediate);
                Realloc(ref tokenIdsBuf,  sL);

                // Full-attention scratch
                Realloc(ref qBuf,         sL * headsQ * headDim);
                Realloc(ref gateBuf,      sL * headsQ * headDim);
                Realloc(ref kBuf,         sL * headsKV * headDim);
                Realloc(ref vBuf,         sL * headsKV * headDim);
                Realloc(ref qNormBuf,     sL * headsQ * headDim);
                Realloc(ref kNormBuf,     sL * headsKV * headDim);
                Realloc(ref attnScoresBuf, headsQ * sL * kL);
                Realloc(ref attendedBuf,  sL * headsQ * headDim);

                // Linear-attention scratch
                Realloc(ref linearQkvBuf,    sL * convDim);
                Realloc(ref linearZBuf,      sL * valueDim);
                Realloc(ref linearABuf,      sL * numVHeads);
                Realloc(ref linearBBuf,      sL * numVHeads);
                Realloc(ref linearQBuf,      sL * numVHeads * headKDim);
                Realloc(ref linearKBuf,      sL * numVHeads * headKDim);
                Realloc(ref linearVBuf,      sL * numVHeads * headVDim);
                Realloc(ref linearQNormBuf,  sL * numVHeads * headKDim);
                Realloc(ref linearKNormBuf,  sL * numVHeads * headKDim);
                Realloc(ref linearYBuf,      sL * numVHeads * headVDim);
                Realloc(ref linearYNormBuf,  sL * numVHeads * headVDim);

                curSeqAlloc = sL;
                curKvAlloc = kL;
            }

            void UploadTokens(Tensor ids, int seqLen)
            {
                uint[] arr = new uint[seqLen];
                for (int i = 0; i < seqLen; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);
            }

            static int Div256(int n) => (n + 255) / 256;

            // Quantized modes: bind the scale buffer next to its weight buffer (per-row for INT8,
            // per-32-group for INT4). No-op in FP16 — that variant never references the resources.
            void BindScales(int kernel, string name, ComputeBuffer scales)
            {
                if (Quant != LLMQuant.FP16) cs.SetBuffer(kernel, name, scales);
            }

            // ===================== FULL-ATTENTION DISPATCH =====================
            void DispatchFullAttention(int li, int seqLen, int totalKvLen, bool useCache, int cacheLen)
            {
                int kvLen = useCache ? totalKvLen : seqLen;

                int qOutDim = headsQ * headDim * 2;       // QProjGated p range
                int kvOutDim = headsKV * headDim;

                // Q + gate (fused)
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("head_dim", headDim);
                cs.SetBuffer(kQProjGated, "X", normOutBuf);
                cs.SetBuffer(kQProjGated, "W_Q", weights.W_Q[li]);
                BindScales(kQProjGated, "W_Q_scales", weights.W_QScales[li]);
                cs.SetBuffer(kQProjGated, "Q_out", qBuf);
                cs.SetBuffer(kQProjGated, "gate_out", gateBuf);
                cs.Dispatch(kQProjGated, 1, (seqLen + 7) / 8, (qOutDim + 31) / 32);

                cs.SetInt("num_heads_kv", headsKV);
                cs.SetBuffer(kKProj, "X", normOutBuf);
                cs.SetBuffer(kKProj, "W_K", weights.W_K[li]);
                BindScales(kKProj, "W_K_scales", weights.W_KScales[li]);
                cs.SetBuffer(kKProj, "K_out", kBuf);
                cs.Dispatch(kKProj, 1, (seqLen + 7) / 8, (kvOutDim + 31) / 32);

                cs.SetBuffer(kVProj, "X", normOutBuf);
                cs.SetBuffer(kVProj, "W_V", weights.W_V[li]);
                BindScales(kVProj, "W_V_scales", weights.W_VScales[li]);
                cs.SetBuffer(kVProj, "V_out", vBuf);
                cs.Dispatch(kVProj, 1, (seqLen + 7) / 8, (kvOutDim + 31) / 32);

                // q_norm / k_norm (per-head RMSNorm on head_dim)
                cs.SetInt("num_vectors", seqLen * headsQ);
                cs.SetInt("head_dim", headDim);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsHead, "norm_input", qBuf);
                cs.SetBuffer(kRmsHead, "norm_output", qNormBuf);
                cs.SetBuffer(kRmsHead, "norm_gamma", weights.qNormGamma[li]);
                cs.Dispatch(kRmsHead, Div256(seqLen * headsQ), 1, 1);

                cs.SetInt("num_vectors", seqLen * headsKV);
                cs.SetBuffer(kRmsHead, "norm_input", kBuf);
                cs.SetBuffer(kRmsHead, "norm_output", kNormBuf);
                cs.SetBuffer(kRmsHead, "norm_gamma", weights.kNormGamma[li]);
                cs.Dispatch(kRmsHead, Div256(seqLen * headsKV), 1, 1);

                // Partial RoPE on Q and K
                int rd2 = ropeRotDim / 2;
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("head_dim", headDim);
                cs.SetInt("rope_rot_dim", ropeRotDim);
                cs.SetInt("position_offset", cacheLen);
                cs.SetInt("rope_num_heads", headsQ);
                cs.SetBuffer(kRopePartial, "rope_buf", qNormBuf);
                cs.SetBuffer(kRopePartial, "rope_cos", ropeCos);
                cs.SetBuffer(kRopePartial, "rope_sin", ropeSin);
                cs.Dispatch(kRopePartial, (seqLen * headsQ * rd2 + 127) / 128, 1, 1);

                cs.SetInt("rope_num_heads", headsKV);
                cs.SetBuffer(kRopePartial, "rope_buf", kNormBuf);
                cs.SetBuffer(kRopePartial, "rope_cos", ropeCos);
                cs.SetBuffer(kRopePartial, "rope_sin", ropeSin);
                cs.Dispatch(kRopePartial, (seqLen * headsKV * rd2 + 127) / 128, 1, 1);

                ComputeBuffer kForAttn, vForAttn;
                if (useCache)
                {
                    cs.SetInt("seq_len", seqLen);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetInt("cache_len", cacheLen);
                    cs.SetBuffer(kWriteCache, "kv_new", kNormBuf);
                    cs.SetBuffer(kWriteCache, "kv_cache", cache.kCaches[li]);
                    cs.Dispatch(kWriteCache, Div256(seqLen * headsKV * headDim), 1, 1);

                    cs.SetBuffer(kWriteCache, "kv_new", vBuf);
                    cs.SetBuffer(kWriteCache, "kv_cache", cache.vCaches[li]);
                    cs.Dispatch(kWriteCache, Div256(seqLen * headsKV * headDim), 1, 1);

                    kForAttn = cache.kCaches[li];
                    vForAttn = cache.vCaches[li];
                }
                else
                {
                    kForAttn = kNormBuf;
                    vForAttn = vBuf;
                }

                if (UseFlashAttention && headDim <= 256)
                {
                    // Scores+mask+softmax+attend fused: FlashAttention (one threadgroup per query x head)
                    cs.SetInt("seq_len_q", seqLen);
                    cs.SetInt("seq_len_k", kvLen);
                    cs.SetInt("num_heads_q", headsQ);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetFloat("scale", attnScale);
                    cs.SetBuffer(kFlashAttn, "Q", qNormBuf);
                    cs.SetBuffer(kFlashAttn, "K", kForAttn);
                    cs.SetBuffer(kFlashAttn, "V", vForAttn);
                    cs.SetBuffer(kFlashAttn, "AttendedValues", attendedBuf);
                    cs.Dispatch(kFlashAttn, seqLen, headsQ, 1);
                }
                else
                {
                // Attention scores
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("sequence_length_k", kvLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("num_heads_kv", headsKV);
                cs.SetInt("head_dim", headDim);
                cs.SetFloat("scale", attnScale);
                cs.SetBuffer(kAttnScores, "Q", qNormBuf);
                cs.SetBuffer(kAttnScores, "K", kForAttn);
                cs.SetBuffer(kAttnScores, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kAttnScores, (seqLen + 3) / 4, (kvLen + 31) / 32, (headsQ + 3) / 4);

                // Causal mask
                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", kvLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetBuffer(kMaskCausal, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kMaskCausal, (kvLen + 15) / 16, (headsQ * seqLen + 15) / 16, 1);

                // Softmax
                cs.SetBuffer(kSoftmax, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kSoftmax, Div256(headsQ * seqLen), 1, 1);

                // Attend
                cs.SetInt("sequence_length_v", kvLen);
                cs.SetBuffer(kAttend, "AttentionWeights", attnScoresBuf);
                cs.SetBuffer(kAttend, "V", vForAttn);
                cs.SetBuffer(kAttend, "AttendedValues", attendedBuf);
                cs.Dispatch(kAttend, (headDim + 63) / 64, (seqLen + 3) / 4, (headsQ + 3) / 4);
                }

                // Output gate (sigmoid(gate) * attended) — Qwen3.5 specific
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("head_dim", headDim);
                cs.SetBuffer(kAttnGate, "AttendedValues", attendedBuf);
                cs.SetBuffer(kAttnGate, "gate_in", gateBuf);
                cs.Dispatch(kAttnGate, Div256(seqLen * headsQ * headDim), 1, 1);

                // O proj
                cs.SetInt("inner_embedding_dim", headsQ * headDim);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetBuffer(kOProj, "AttendedValues", attendedBuf);
                cs.SetBuffer(kOProj, "W_O", weights.W_O[li]);
                BindScales(kOProj, "W_O_scales", weights.W_OScales[li]);
                cs.SetBuffer(kOProj, "O", attnOutBuf);
                cs.Dispatch(kOProj, 1, (seqLen + 3) / 4, (hiddenSize + 31) / 32);
            }

            // ===================== LINEAR-ATTENTION (DeltaNet) DISPATCH =====================
            void DispatchLinearAttention(int li, int seqLen)
            {
                // 1. in_proj_qkv -> linearQkvBuf
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetInt("linear_conv_dim", convDim);
                cs.SetBuffer(kInProjQKV, "X", normOutBuf);
                cs.SetBuffer(kInProjQKV, "W_inProjQKV", weights.W_inProjQKV[li]);
                BindScales(kInProjQKV, "W_inProjQKV_scales", weights.W_inProjQKVScales[li]);
                cs.SetBuffer(kInProjQKV, "linear_qkv", linearQkvBuf);
                cs.Dispatch(kInProjQKV, 1, (seqLen + 7) / 8, (convDim + 31) / 32);

                // 2. in_proj_z -> linearZBuf
                cs.SetInt("linear_value_dim", valueDim);
                cs.SetBuffer(kInProjZ, "X", normOutBuf);
                cs.SetBuffer(kInProjZ, "W_inProjZ", weights.W_inProjZ[li]);
                BindScales(kInProjZ, "W_inProjZ_scales", weights.W_inProjZScales[li]);
                cs.SetBuffer(kInProjZ, "linear_z_w", linearZBuf);
                cs.Dispatch(kInProjZ, 1, (seqLen + 7) / 8, (valueDim + 31) / 32);

                // 3. in_proj_a -> linearABuf
                cs.SetInt("linear_num_v_heads", numVHeads);
                cs.SetBuffer(kInProjA, "X", normOutBuf);
                cs.SetBuffer(kInProjA, "W_inProjA", weights.W_inProjA[li]);
                cs.SetBuffer(kInProjA, "linear_a_w", linearABuf);
                cs.Dispatch(kInProjA, Div256(seqLen * numVHeads), 1, 1);

                // 4. in_proj_b -> linearBBuf
                cs.SetBuffer(kInProjB, "X", normOutBuf);
                cs.SetBuffer(kInProjB, "W_inProjB", weights.W_inProjB[li]);
                cs.SetBuffer(kInProjB, "linear_b_w", linearBBuf);
                cs.Dispatch(kInProjB, Div256(seqLen * numVHeads), 1, 1);

                // 5. causal conv1d (in-place on linearQkvBuf) + state update
                cs.SetInt("linear_conv_kernel", convKernel);
                if (seqLen == 1)
                {
                    cs.SetBuffer(kConvUpdate, "linear_qkv", linearQkvBuf);
                    cs.SetBuffer(kConvUpdate, "conv_state", cache.convStates[li]);
                    cs.SetBuffer(kConvUpdate, "conv_weight", weights.convWeight[li]);
                    cs.Dispatch(kConvUpdate, Div256(convDim), 1, 1);
                }
                else
                {
                    cs.SetInt("sequence_length_q", seqLen);
                    cs.SetBuffer(kConvPrefill, "linear_qkv", linearQkvBuf);
                    cs.SetBuffer(kConvPrefill, "conv_state", cache.convStates[li]);
                    cs.SetBuffer(kConvPrefill, "conv_weight", weights.convWeight[li]);
                    cs.Dispatch(kConvPrefill, Div256(convDim), 1, 1);
                }

                // 6. split conv output -> Q, K, V
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("linear_key_dim", keyDim);
                cs.SetInt("linear_head_k_dim", headKDim);
                cs.SetInt("linear_head_v_dim", headVDim);
                cs.SetBuffer(kSplitConv, "linear_qkv", linearQkvBuf);
                cs.SetBuffer(kSplitConv, "Q_out", linearQBuf);
                cs.SetBuffer(kSplitConv, "K_out", linearKBuf);
                cs.SetBuffer(kSplitConv, "V_out", linearVBuf);
                cs.Dispatch(kSplitConv, Div256(seqLen * convDim), 1, 1);

                // 7. L2 norm Q + fold the post-norm Q scaling (1/sqrt(head_k_dim))
                //    into the same kernel via the 'scale' uniform.
                float qScale = 1f / MathF.Sqrt(headKDim);
                cs.SetInt("num_vectors", seqLen * numVHeads);
                cs.SetInt("head_dim", headKDim);
                cs.SetFloat("scale", qScale);
                cs.SetBuffer(kL2NormHead, "norm_input", linearQBuf);
                cs.SetBuffer(kL2NormHead, "norm_output", linearQNormBuf);
                cs.Dispatch(kL2NormHead, Div256(seqLen * numVHeads), 1, 1);

                // 8. L2 norm K (no extra scaling)
                cs.SetFloat("scale", 1f);
                cs.SetBuffer(kL2NormHead, "norm_input", linearKBuf);
                cs.SetBuffer(kL2NormHead, "norm_output", linearKNormBuf);
                cs.Dispatch(kL2NormHead, Div256(seqLen * numVHeads), 1, 1);

                // 9. recurrent gated delta-rule scan
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("linear_num_v_heads", numVHeads);
                cs.SetInt("linear_head_k_dim", headKDim);
                cs.SetInt("linear_head_v_dim", headVDim);
                cs.SetBuffer(kDeltaNet, "linear_q", linearQNormBuf);
                cs.SetBuffer(kDeltaNet, "linear_k", linearKNormBuf);
                cs.SetBuffer(kDeltaNet, "linear_v", linearVBuf);
                cs.SetBuffer(kDeltaNet, "linear_a", linearABuf);
                cs.SetBuffer(kDeltaNet, "linear_b", linearBBuf);
                cs.SetBuffer(kDeltaNet, "linear_y", linearYBuf);
                cs.SetBuffer(kDeltaNet, "recurrent_state", cache.recurrentStates[li]);
                cs.SetBuffer(kDeltaNet, "dt_bias", weights.dtBias[li]);
                cs.SetBuffer(kDeltaNet, "A_log", weights.ALog[li]);
                cs.Dispatch(kDeltaNet, 1, numVHeads, 1);

                // 10. RMSNormGated (with z gate)
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetInt("linear_value_dim", valueDim);
                cs.SetBuffer(kRMSGated, "linear_y", linearYBuf);
                cs.SetBuffer(kRMSGated, "linear_z", linearZBuf);
                cs.SetBuffer(kRMSGated, "linear_y_norm", linearYNormBuf);
                cs.SetBuffer(kRMSGated, "norm_gamma", weights.linearNormGamma[li]);
                cs.Dispatch(kRMSGated, Div256(seqLen * numVHeads), 1, 1);

                // 11. out_proj -> attnOutBuf
                cs.SetInt("batch_size", 1);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetBuffer(kLinearOut, "linear_y_norm", linearYNormBuf);
                cs.SetBuffer(kLinearOut, "W_outProj", weights.W_outProj[li]);
                BindScales(kLinearOut, "W_outProj_scales", weights.W_outProjScales[li]);
                cs.SetBuffer(kLinearOut, "O", attnOutBuf);
                cs.Dispatch(kLinearOut, 1, (seqLen + 3) / 4, (hiddenSize + 31) / 32);
            }

            // ===================== Per-layer wrapper =====================
            void DispatchLayer(int li, int seqLen, int totalKvLen, bool useCache)
            {
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int hidTotal = seqLen * hiddenSize;

                // skip = hidden
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kCopy, "buf_a", skipBuf);
                cs.SetBuffer(kCopy, "buf_b", hiddenBuf);
                cs.Dispatch(kCopy, Div256(hidTotal), 1, 1);

                // input layernorm
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsHidden, "norm_gamma", weights.inputLnGamma[li]);
                cs.Dispatch(kRmsHidden, Div256(seqLen), 1, 1);

                // attention block -> attnOutBuf
                if (layerTypes[li] == Qwen3_5LayerType.FullAttention)
                    DispatchFullAttention(li, seqLen, totalKvLen, useCache, cacheLen);
                else
                    DispatchLinearAttention(li, seqLen);

                // hidden = attnOut + skip
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kCopy, "buf_a", hiddenBuf);
                cs.SetBuffer(kCopy, "buf_b", attnOutBuf);
                cs.Dispatch(kCopy, Div256(hidTotal), 1, 1);

                cs.SetBuffer(kAddRes, "buf_a", hiddenBuf);
                cs.SetBuffer(kAddRes, "buf_b", skipBuf);
                cs.Dispatch(kAddRes, Div256(hidTotal), 1, 1);

                // skip = hidden
                cs.SetBuffer(kCopy, "buf_a", skipBuf);
                cs.SetBuffer(kCopy, "buf_b", hiddenBuf);
                cs.Dispatch(kCopy, Div256(hidTotal), 1, 1);

                // post_attention_layernorm
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetBuffer(kRmsHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsHidden, "norm_gamma", weights.postAttnLnGamma[li]);
                cs.Dispatch(kRmsHidden, Div256(seqLen), 1, 1);

                // hidden = normOut (so MLP can work in-place via 'input')
                cs.SetBuffer(kCopy, "buf_a", hiddenBuf);
                cs.SetBuffer(kCopy, "buf_b", normOutBuf);
                cs.Dispatch(kCopy, Div256(hidTotal), 1, 1);

                // MLP (silu)
                bool vec1 = seqLen == 1;
                int kGU = vec1 ? kGateUp1 : kGateUp;
                int kDN = vec1 ? kDown1 : kDown;

                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("intermediate_size", intermediate);
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("activation_type", 0); // silu
                cs.SetBuffer(kGU, "input", hiddenBuf);
                cs.SetBuffer(kGU, "mlp_gate_w", weights.mlpGate[li]);
                cs.SetBuffer(kGU, "mlp_up_w",   weights.mlpUp[li]);
                BindScales(kGU, "mlp_gate_scales", weights.mlpGateScales[li]);
                BindScales(kGU, "mlp_up_scales",   weights.mlpUpScales[li]);
                cs.SetBuffer(kGU, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kGU, (intermediate + 255) / 256, 1, 1);
                else      cs.Dispatch(kGU, (intermediate + 63) / 64, (seqLen + 7) / 8, 1);

                cs.SetBuffer(kDN, "input", hiddenBuf);
                cs.SetBuffer(kDN, "mlp_down_w", weights.mlpDown[li]);
                BindScales(kDN, "mlp_down_scales", weights.mlpDownScales[li]);
                cs.SetBuffer(kDN, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kDN, (hiddenSize + 255) / 256, 1, 1);
                else      cs.Dispatch(kDN, (hiddenSize + 63) / 64, (seqLen + 7) / 8, 1);

                // hidden += skip
                cs.SetBuffer(kAddRes, "buf_a", hiddenBuf);
                cs.SetBuffer(kAddRes, "buf_b", skipBuf);
                cs.Dispatch(kAddRes, Div256(hidTotal), 1, 1);
            }

            public void Forward(Tensor input_ids, bool useCache, bool lastPosOnly)
            {
                int seqLen = input_ids.Size(-1);
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int totalKvLen = cacheLen + seqLen;

                EnsureScratch(seqLen, totalKvLen);
                UploadTokens(input_ids, seqLen);

                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetBuffer(kEmbed, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbed, "embed_weights", weights.embedLmHead);
                BindScales(kEmbed, "embed_scales", weights.embedScales);
                cs.SetBuffer(kEmbed, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbed, Div256(seqLen * hiddenSize), 1, 1);

                for (int i = 0; i < numLayers; i++)
                    DispatchLayer(i, seqLen, totalKvLen, useCache);

                if (useCache) cache.CachedTokenCount += seqLen;

                if (lastPosOnly) DispatchFinalLast(seqLen);
                else DispatchFinalAll(seqLen);
            }

            public IEnumerator ForwardYielding(Tensor input_ids, bool useCache, bool lastPosOnly)
            {
                int seqLen = input_ids.Size(-1);
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int totalKvLen = cacheLen + seqLen;

                EnsureScratch(seqLen, totalKvLen);
                UploadTokens(input_ids, seqLen);

                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetBuffer(kEmbed, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbed, "embed_weights", weights.embedLmHead);
                BindScales(kEmbed, "embed_scales", weights.embedScales);
                cs.SetBuffer(kEmbed, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbed, Div256(seqLen * hiddenSize), 1, 1);
                yield return null;

                for (int i = 0; i < numLayers; i++)
                {
                    DispatchLayer(i, seqLen, totalKvLen, useCache);
                    yield return null;
                }

                if (useCache) cache.CachedTokenCount += seqLen;

                if (lastPosOnly) DispatchFinalLast(seqLen);
                else DispatchFinalAll(seqLen);
                yield return null;
            }

            void DispatchFinalLast(int seqLen)
            {
                cs.SetInt("buffer_size", hiddenSize);
                cs.SetInt("copy_src_offset", (seqLen - 1) * hiddenSize);
                cs.SetBuffer(kCopySlice, "buf_a", lastHiddenBuf);
                cs.SetBuffer(kCopySlice, "buf_b", hiddenBuf);
                cs.Dispatch(kCopySlice, Div256(hiddenSize), 1, 1);

                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsHidden, "norm_input", lastHiddenBuf);
                cs.SetBuffer(kRmsHidden, "norm_output", normSingleBuf);
                cs.SetBuffer(kRmsHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsHidden, 1, 1, 1);

                Realloc(ref logitsBuf, vocab);
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("vocab_size", vocab);
                cs.SetBuffer(kLm1, "lm_weights", weights.embedLmHead);
                BindScales(kLm1, "embed_scales", weights.embedScales);
                cs.SetBuffer(kLm1, "lm_input", normSingleBuf);
                cs.SetBuffer(kLm1, "lm_output", logitsBuf);
                cs.Dispatch(kLm1, (vocab + 511) / 512, 1, 1);
            }

            void DispatchFinalAll(int seqLen)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsHidden, Div256(seqLen), 1, 1);

                Realloc(ref logitsBuf, seqLen * vocab);
                bool v1 = seqLen == 1;
                int k = v1 ? kLm1 : kLm;
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("vocab_size", vocab);
                cs.SetBuffer(k, "lm_weights", weights.embedLmHead);
                BindScales(k, "embed_scales", weights.embedScales);
                cs.SetBuffer(k, "lm_input", normOutBuf);
                cs.SetBuffer(k, "lm_output", logitsBuf);
                if (v1) cs.Dispatch(k, (vocab + 511) / 512, 1, 1);
                else    cs.Dispatch(k, (vocab + 31) / 32, (seqLen + 7) / 8, 1);
            }

            // Penalize already-generated tokens in-place on the logits (GPU). repetition_penalty=1 and
            // presence_penalty=0 are both no-ops, so this is free when penalties are disabled.
            void ApplyPenalties(float presencePenalty, float repetitionPenalty)
            {
                cs.SetInt("vocab_size", vocab);
                cs.SetFloat("presence_penalty", presencePenalty);
                cs.SetFloat("repetition_penalty", repetitionPenalty);
                cs.SetBuffer(kApplyPenalty, "logits_buf", logitsBuf);
                cs.SetBuffer(kApplyPenalty, "token_seen", tokenSeenBuf);
                cs.Dispatch(kApplyPenalty, (vocab + 255) / 256, 1, 1);
            }

            // Record the token just written to argmax_result so it is penalized on subsequent steps.
            void MarkSampledSeen()
            {
                cs.SetBuffer(kMarkSeen, "argmax_result", argmaxBuf);
                cs.SetBuffer(kMarkSeen, "token_seen", tokenSeenBuf);
                cs.Dispatch(kMarkSeen, 1, 1, 1);
            }

            // Queues the penalty + sampler (+ mark-seen) kernels; the chosen token id lands in
            // argmaxBuf on the GPU. Reading it back is the caller's job (sync or async).
            void DispatchSampleKernels(float temperature, int topK, float topP, float minP,
                                       float presencePenalty, float repetitionPenalty)
            {
                ApplyPenalties(presencePenalty, repetitionPenalty);
                cs.SetInt("vocab_size", vocab);
                if (temperature == 0f)
                {
                    cs.SetBuffer(kArgMax, "logits_buf", logitsBuf);
                    cs.SetBuffer(kArgMax, "argmax_result", argmaxBuf);
                    cs.Dispatch(kArgMax, 1, 1, 1);
                }
                else
                {
                    cs.SetFloat("temperature", temperature);
                    cs.SetInt("top_k_val", topK);
                    cs.SetFloat("top_p_val", topP);
                    cs.SetFloat("min_p_val", minP);
                    cs.SetInt("rng_seed", UnityEngine.Random.Range(int.MinValue, int.MaxValue));
                    cs.SetBuffer(kSample, "logits_buf", logitsBuf);
                    cs.SetBuffer(kSample, "probs_buf", probsBuf);
                    cs.SetBuffer(kSample, "argmax_result", argmaxBuf);
                    cs.Dispatch(kSample, 1, 1, 1);
                }
                MarkSampledSeen();
            }

            public int SampleGreedy(float presencePenalty = 0f, float repetitionPenalty = 1f)
                => Sample(0f, 0, 1f, 0f, presencePenalty, repetitionPenalty);

            public int SampleStochastic(float temperature, int topK, float topP, float minP,
                                        float presencePenalty = 0f, float repetitionPenalty = 1f)
                => Sample(temperature, topK, topP, minP, presencePenalty, repetitionPenalty);

            // Synchronous sample: blocks the main thread until every queued GPU dispatch (the whole
            // forward) has finished. Fine for offline use; interactive paths should prefer
            // SampleYielding, which waits on an AsyncGPUReadback instead of stalling the frame.
            public int Sample(float temperature, int topK, float topP, float minP,
                              float presencePenalty = 0f, float repetitionPenalty = 1f)
            {
                DispatchSampleKernels(temperature, topK, topP, minP, presencePenalty, repetitionPenalty);
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                return (int)r[0];
            }

            // Async sample: same kernels, but the 4-byte token id comes back via AsyncGPUReadback,
            // so the main thread never blocks on the GPU queue (the sync GetData stalls for the
            // entire in-flight forward — ~hundreds of ms right after a prefill). Writes result[0].
            public IEnumerator SampleYielding(float temperature, int topK, float topP, float minP,
                                              float presencePenalty, float repetitionPenalty, int[] result)
            {
                DispatchSampleKernels(temperature, topK, topP, minP, presencePenalty, repetitionPenalty);
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
                uint[] r = new uint[1]; argmaxBuf.GetData(r); // fallback: sync readback
                result[0] = (int)r[0];
            }

            public Tensor ReadLogits(int seqLen)
                => seqLen == 1 ? Tensor.Constant(logitsBuf, vocab) : Tensor.Constant(logitsBuf, seqLen, vocab);

            public void ResetCache()
            {
                cache.Reset();
                // Zero the SSM states on the GPU (the KV caches need no zeroing — CachedTokenCount
                // masks them). Done here rather than in Qwen3_5Cache so it can use the ZeroBuffer
                // kernel instead of main-thread SetData with managed zero arrays.
                if (Qwen3_5Config.USE_KV_CACHE)
                {
                    for (int i = 0; i < numLayers; i++)
                    {
                        if (cache.convStates[i] != null) ZeroFloatBuffer(cache.convStates[i]);
                        if (cache.recurrentStates[i] != null) ZeroFloatBuffer(cache.recurrentStates[i]);
                    }
                }
                // Clear generated-token history so penalties start fresh for the new sequence.
                cs.SetInt("vocab_size", vocab);
                cs.SetBuffer(kZeroSeen, "token_seen", tokenSeenBuf);
                cs.Dispatch(kZeroSeen, (vocab + 255) / 256, 1, 1);
            }

            void ZeroFloatBuffer(ComputeBuffer buf)
            {
                cs.SetInt("buffer_size", buf.count);
                cs.SetBuffer(kZero, "buf_a", buf);
                cs.Dispatch(kZero, Div256(buf.count), 1, 1);
            }

            bool _warmedUp;

            // Every buffer property name in Qwen3_5CS.compute; used by WarmupKernelsIndividually to
            // bind a distinct dummy to each (SetBuffer on a name a kernel doesn't use is a no-op;
            // distinct buffers because D3D11 forbids the same UAV in two slots of one dispatch).
            static readonly string[] WARMUP_BUFFER_NAMES =
            {
                "embed_weights", "lm_weights", "norm_gamma",
                "embed_scales", "W_Q_scales", "W_K_scales", "W_V_scales", "W_O_scales",
                "W_inProjQKV_scales", "W_inProjZ_scales", "W_outProj_scales",
                "mlp_gate_scales", "mlp_up_scales", "mlp_down_scales",
                "W_Q", "W_K", "W_V", "W_O",
                "W_inProjQKV", "W_inProjZ", "W_inProjA", "W_inProjB",
                "conv_weight", "dt_bias", "A_log", "W_outProj",
                "mlp_gate_w", "mlp_up_w", "mlp_down_w",
                "rope_cos", "rope_sin",
                "token_ids", "embed_output", "norm_output", "norm_input",
                "X", "Q_out", "K_out", "V_out", "gate_out", "rope_buf",
                "kv_cache", "kv_new", "Q", "K", "V",
                "AttentionWeights", "AttendedValues", "O", "gate_in",
                "input", "intermediate", "mlp_input", "mlp_output",
                "conv_state", "recurrent_state",
                "linear_qkv", "linear_q", "linear_k", "linear_v",
                "linear_q_norm", "linear_k_norm",
                "linear_a", "linear_b", "linear_z",
                "linear_a_w", "linear_b_w", "linear_z_w",
                "linear_y", "linear_y_norm",
                "lm_input", "lm_output",
                "logits_buf", "probs_buf", "argmax_result", "token_seen",
                "buf_a", "buf_b",
            };

            // Every integer uniform that gates a kernel's thread guards. Zeroing them makes every
            // warmup dispatch degenerate (all threads early-out), so binding dummies is safe.
            static readonly string[] WARMUP_SIZE_UNIFORMS =
            {
                "batch_size", "sequence_length_q", "sequence_length_k", "sequence_length_v",
                "embedding_dim", "inner_embedding_dim", "num_heads_q", "num_heads_kv", "head_dim",
                "seq_len", "seq_len_q", "seq_len_k", "hidden_size", "intermediate_size", "num_vectors",
                "rope_rot_dim", "position_offset", "rope_num_heads", "cache_len",
                "linear_conv_dim", "linear_value_dim", "linear_key_dim", "linear_num_v_heads",
                "linear_conv_kernel", "linear_head_k_dim", "linear_head_v_dim",
                "activation_type", "vocab_size", "top_k_val", "rng_seed",
                "buffer_size", "copy_src_offset",
            };

            static readonly string[] ALL_KERNEL_NAMES =
            {
                "EmbeddingLookup", "RmsNormHidden", "RmsNormHead",
                "QProjGated", "KProj", "VProj", "OProj",
                "ApplyRopePartial", "WriteCacheFull", "ApplyMaskCausal", "SoftmaxRows",
                "ComputeAttentionScores", "FlashAttention", "AttendValues", "ApplyAttnGate",
                "LinearInProjQKV", "LinearInProjZ", "LinearInProjA", "LinearInProjB",
                "CausalConv1DUpdate", "CausalConv1DPrefill", "SplitConvOutQKV",
                "L2NormPerHead", "DeltaNetRecurrent", "RMSNormGated", "LinearOutProj",
                "GateUp", "Down", "GateUp1Vec", "Down1Vec",
                "LmHeadPredict", "LmHeadPredict1Vec", "ArgMax", "SampleToken",
                "ZeroBuffer", "CopyBuffer", "CopySlice", "AddResidual",
                "ApplyRepetitionPresencePenalty", "MarkSampledTokenSeen", "ZeroTokenSeen",
            };

            static bool _kernelsPrewarmed;

            // The driver compiles each kernel's ISA on its FIRST dispatch — a one-time per-session
            // cost of up to ~800 ms for the big kernels (DeltaNet). This pass dispatches every kernel
            // once with zero-size uniforms — ONE kernel compile per frame. It is STATIC and needs no
            // model or weights, so a game can run it at scene start (e.g. while the player walks
            // around), long before any LLM is constructed; Warmup() also runs it (idempotent).
            public static System.Collections.IEnumerator PrewarmKernels()
            {
                if (_kernelsPrewarmed) yield break;
                _kernelsPrewarmed = true;

                ComputeShader shader = DeepUnityMeta.Qwen3_5CS;

                var dummies = new ComputeBuffer[WARMUP_BUFFER_NAMES.Length];
                for (int i = 0; i < dummies.Length; i++)
                    dummies[i] = new ComputeBuffer(256, 4, ComputeBufferType.Structured);

                foreach (string u in WARMUP_SIZE_UNIFORMS) shader.SetInt(u, 0);

                foreach (string name in ALL_KERNEL_NAMES)
                {
                    int k = shader.FindKernel(name);
                    for (int i = 0; i < WARMUP_BUFFER_NAMES.Length; i++)
                        shader.SetBuffer(k, WARMUP_BUFFER_NAMES[i], dummies[i]);
                    shader.Dispatch(k, 1, 1, 1);
                    yield return null;
                }

                foreach (var d in dummies) d.Release();
            }

            // Compiles every compute kernel behind the loading screen so the first real reply is fast:
            // first one degenerate dispatch per kernel per frame (the actual driver-compile cost,
            // overlapping the weight stream), then throwaway forwards + both sampler paths to exercise
            // the real code paths. Two token counts cover both kernel variants — multi-token (prefill)
            // and single-token (generation, the *1Vec kernels). Idempotent (runs once per model).
            public System.Collections.IEnumerator Warmup()
            {
                if (_warmedUp) yield break;
                var sw = System.Diagnostics.Stopwatch.StartNew();

                var pk = PrewarmKernels();              // no-op if already prewarmed at scene start
                while (pk.MoveNext()) yield return pk.Current;

                while (!IsReady) yield return null;

                int[] tok = new int[1];
                foreach (int n in new[] { 4, 1 })
                {
                    // Match the real inference path (useCache) so the cache-write/update kernels warm too.
                    var e = ForwardYielding(Tensor.Constant(new float[n]), useCache: Qwen3_5Config.USE_KV_CACHE, lastPosOnly: true);
                    while (e.MoveNext()) yield return e.Current;
                    // Async readback: a sync Sample here blocks ~200 ms waiting for the queued forward.
                    var s = SampleYielding(1f, 20, 1f, 0f, 0f, 1f, tok);
                    while (s.MoveNext()) yield return s.Current;
                }
                var g = SampleYielding(0f, 0, 1f, 0f, 0f, 1f, tok); // greedy (temperature==0) path
                while (g.MoveNext()) yield return g.Current;
                yield return null;

                ResetCache(); // undo the warmup's token_seen marks / cache writes
                weights.warmupMs = sw.Elapsed.TotalMilliseconds;
                _warmedUp = true;
            }

            public void Dispose()
            {
                weights?.Dispose(); cache?.Dispose();
                ropeCos?.Release(); ropeSin?.Release();
                hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release(); attnOutBuf?.Release();
                mlpInterBuf?.Release();
                logitsBuf?.Release(); probsBuf?.Release(); argmaxBuf?.Release(); tokenIdsBuf?.Release();
                tokenSeenBuf?.Release();
                lastHiddenBuf?.Release(); normSingleBuf?.Release();
                qBuf?.Release(); kBuf?.Release(); vBuf?.Release(); gateBuf?.Release();
                qNormBuf?.Release(); kNormBuf?.Release();
                attnScoresBuf?.Release(); attendedBuf?.Release();
                linearQkvBuf?.Release(); linearZBuf?.Release(); linearABuf?.Release(); linearBBuf?.Release();
                linearQBuf?.Release(); linearKBuf?.Release(); linearVBuf?.Release();
                linearQNormBuf?.Release(); linearKNormBuf?.Release();
                linearYBuf?.Release(); linearYNormBuf?.Release();
            }
        }
    }
}
