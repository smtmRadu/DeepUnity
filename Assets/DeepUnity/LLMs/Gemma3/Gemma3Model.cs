using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class Gemma3Model : IDisposable
        {
            // FlashAttention-1 fused attention (scores+mask+softmax+AV in one dispatch, online
            // softmax, no materialized score matrix; sliding-window layers never touch KV outside
            // the window). Static so probes can A/B it against the legacy 4-dispatch path; the
            // kernel requires head_dim <= 256 — DispatchLayer falls back to legacy otherwise.
            public static bool UseFlashAttention = true;

            // Weight format this instance runs (set in the ctor; INT8/INT4 enable the matching
            // *_WEIGHTS shader keyword and bind scale buffers next to each matmul weight — per
            // output row for INT8, per 32-weight group for INT4).
            public readonly LLMQuant Quant;

            // KV-cache precision (independent of the weight Quant above). FP16 enables the KV_FP16
            // shader keyword (KVCache.hlsl) and halves the KV buffers / per-step read bandwidth.
            // Named KV (not KVQuant) to avoid clashing with the KVQuant type name. Mirrors Gemma3Cache.KV.
            public readonly KVQuant KV;

            ComputeShader cs;

            // kernel ids
            int kEmbedLookup, kRmsNormHidden, kRmsNormHead, kSplitQKV;
            int kApplyRope, kWriteCacheFull, kApplyMask, kSoftmaxRows, kFlashAttn;
            int kArgMax, kSampleToken, kZeroBuffer, kCopyBuffer, kCopySlice, kAddResidual;
            int kQKVProj, kAttnScores, kAttendValues, kOProj;
            int kGateUp, kDown, kGateUp1Vec, kDown1Vec;
            int kLmHead, kLmHead1Vec;

            public Gemma3Weights weights;
            public Gemma3Cache cache;

            // RoPE (packed FP16 in uint buffers)
            ComputeBuffer ropeCosFull, ropeSinFull;
            ComputeBuffer ropeCosLocal, ropeSinLocal;

            // FP32 scratch buffers (same as Gemma3GPU)
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf;
            ComputeBuffer qkvBuf, qBuf, kBuf, vBuf, qNormBuf, kNormBuf;
            ComputeBuffer attnScoresBuf, attendedBuf, attnOutBuf;
            ComputeBuffer mlpInterBuf;
            ComputeBuffer logitsBuf, probsBuf;
            ComputeBuffer argmaxBuf;
            ComputeBuffer tokenIdsBuf;
            ComputeBuffer lastHiddenBuf;
            ComputeBuffer normSingleBuf;

            int curSeqAlloc, curKvAlloc;

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;
            readonly int slidingWindow;
            readonly float rmsEps, embedScale, attnScaling;

            // RoPE tables are computed on a background thread and uploaded by a main-thread
            // coroutine when ready, so the ~tens-of-MB trig + fp16 conversion never blocks the
            // construction frame. Both weights and RoPE must be ready before the first forward.
            volatile bool ropeReady;
            volatile bool _ropeComputed;
            uint[] _ropeCF, _ropeSF, _ropeCL, _ropeSL;

            public bool IsReady => weights.IsReady && ropeReady;

            public Gemma3Model(string paramsPath, int cacheCapacity, LLMQuant quant = LLMQuant.FP16,
                               KVQuant kvQuant = KVQuant.FP16)
            {
                Quant = quant;
                KV = kvQuant;
                numLayers = Gemma3Modeling.Gemma3Config.NUM_LAYERS;
                hiddenSize = Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
                headDim = Gemma3Modeling.Gemma3Config.HEAD_DIM;
                headsQ = Gemma3Modeling.Gemma3Config.HEADS_Q;
                headsKV = Gemma3Modeling.Gemma3Config.HEADS_KV;
                intermediateSize = Gemma3Modeling.Gemma3Config.MLP_INTERMEDIATE_SIZE;
                vocabSize = Gemma3Modeling.Gemma3Config.VOCAB_SIZE;
                slidingWindow = Gemma3Modeling.Gemma3Config.SLIDING_WINDOW;
                rmsEps = Gemma3Modeling.Gemma3Config.RMS_EPS;
                embedScale = MathF.Sqrt(hiddenSize);
                attnScaling = MathF.Pow(Gemma3Modeling.Gemma3Config.QUERY_PRE_ATTENTION_SCALAR, -0.5f);

                float exp = Gemma3Modeling.Gemma3Config.ATTN_EXPANSION_FACTOR;
                innerEmbDim = (int)(hiddenSize * exp);
                qkvProjDim = innerEmbDim + 2 * (innerEmbDim * headsKV / headsQ);

                cs = DeepUnityMeta.Gemma3CS;
                // Keyword state lives on the shared shader asset — one quant mode per session;
                // don't run two differently-quantized Gemma instances simultaneously.
                if (quant == LLMQuant.INT8) cs.EnableKeyword("INT8_WEIGHTS"); else cs.DisableKeyword("INT8_WEIGHTS");
                if (quant == LLMQuant.INT4) cs.EnableKeyword("INT4_WEIGHTS"); else cs.DisableKeyword("INT4_WEIGHTS");
                KVQuantUtil.SetKeyword(cs, kvQuant);   // KV_FP16 (or none for FP32) — KV precision is independent of the weight quant
                CacheKernelIds();

                weights = new Gemma3Weights(paramsPath, quant);
                cache = new Gemma3Cache(numLayers, cacheCapacity, headsKV, headDim, kvQuant);

                PrecomputeRoPEAsync();

                // Fixed-size FP32 buffers
                probsBuf = new ComputeBuffer(vocabSize, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
            }

            void CacheKernelIds()
            {
                kEmbedLookup = cs.FindKernel("EmbeddingLookup");
                kRmsNormHidden = cs.FindKernel("RmsNormHidden");
                kRmsNormHead = cs.FindKernel("RmsNormHead");
                kSplitQKV = cs.FindKernel("SplitQKV");
                kApplyRope = cs.FindKernel("ApplyRopeSplitHalf");
                kWriteCacheFull = cs.FindKernel("WriteCacheFull");
                kApplyMask = cs.FindKernel("ApplyMask");
                kSoftmaxRows = cs.FindKernel("SoftmaxRows");
                kFlashAttn = cs.FindKernel("FlashAttention");
                kArgMax = cs.FindKernel("ArgMax");
                kSampleToken = cs.FindKernel("SampleToken");
                kZeroBuffer = cs.FindKernel("ZeroBuffer");
                kCopyBuffer = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAddResidual = cs.FindKernel("AddResidual");
                kQKVProj = cs.FindKernel("QKVProj");
                kAttnScores = cs.FindKernel("ComputeAttentionScores");
                kAttendValues = cs.FindKernel("AttendValues");
                kOProj = cs.FindKernel("OProj");
                kGateUp = cs.FindKernel("GateUp");
                kDown = cs.FindKernel("Down");
                kGateUp1Vec = cs.FindKernel("GateUp1Vec");
                kDown1Vec = cs.FindKernel("Down1Vec");
                kLmHead = cs.FindKernel("LmHeadPredict");
                kLmHead1Vec = cs.FindKernel("LmHeadPredict1Vec");
            }

            // Pack FP16 RoPE caches into uint buffers
            static ComputeBuffer PackedHalfBuf(int halfCount)
            {
                return new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            }

            // Allocates the RoPE buffers (cheap), computes the four cos/sin tables on a background
            // thread (pure managed math — Unity's Mathf.FloatToHalf has per-call native overhead,
            // so a managed converter is used), then uploads via a main-thread coroutine when ready.
            void PrecomputeRoPEAsync()
            {
                int maxSeq = Gemma3Modeling.Gemma3Config.MAX_POSITION_EMBEDDINGS;
                int hd2 = headDim / 2;
                int hDim = headDim;
                int thetaFull = Gemma3Modeling.Gemma3Config.ROPE_THETA;
                int thetaLocal = Gemma3Modeling.Gemma3Config.ROPE_LOCAL_BASE_FREQUENCY;
                int packedLen = (maxSeq * hd2) / 2;

                ropeCosFull = PackedHalfBuf(maxSeq * hd2);
                ropeSinFull = PackedHalfBuf(maxSeq * hd2);
                ropeCosLocal = PackedHalfBuf(maxSeq * hd2);
                ropeSinLocal = PackedHalfBuf(maxSeq * hd2);

                _ = Task.Run(() =>
                {
                    uint[] cF = new uint[packedLen], sF = new uint[packedLen];
                    uint[] cL = new uint[packedLen], sL = new uint[packedLen];
                    Parallel.For(0, maxSeq, pos =>
                    {
                        int baseU = pos * (hd2 / 2);
                        for (int j = 0; j < hd2 / 2; j++)
                        {
                            int i0 = 2 * j, i1 = 2 * j + 1;
                            float fF0 = 1f / MathF.Pow(thetaFull, 2f * i0 / hDim);
                            float fF1 = 1f / MathF.Pow(thetaFull, 2f * i1 / hDim);
                            float fL0 = 1f / MathF.Pow(thetaLocal, 2f * i0 / hDim);
                            float fL1 = 1f / MathF.Pow(thetaLocal, 2f * i1 / hDim);
                            cF[baseU + j] = (uint)F32ToF16(MathF.Cos(pos * fF0)) | ((uint)F32ToF16(MathF.Cos(pos * fF1)) << 16);
                            sF[baseU + j] = (uint)F32ToF16(MathF.Sin(pos * fF0)) | ((uint)F32ToF16(MathF.Sin(pos * fF1)) << 16);
                            cL[baseU + j] = (uint)F32ToF16(MathF.Cos(pos * fL0)) | ((uint)F32ToF16(MathF.Cos(pos * fL1)) << 16);
                            sL[baseU + j] = (uint)F32ToF16(MathF.Sin(pos * fL0)) | ((uint)F32ToF16(MathF.Sin(pos * fL1)) << 16);
                        }
                    });
                    _ropeCF = cF; _ropeSF = sF; _ropeCL = cL; _ropeSL = sL;
                    _ropeComputed = true;
                });

                DeepUnityDispatcher.Run(UploadRopeWhenReady());
            }

            IEnumerator UploadRopeWhenReady()
            {
                while (!_ropeComputed) yield return null;
                ropeCosFull.SetData(_ropeCF); ropeSinFull.SetData(_ropeSF);
                ropeCosLocal.SetData(_ropeCL); ropeSinLocal.SetData(_ropeSL);
                _ropeCF = _ropeSF = _ropeCL = _ropeSL = null;
                ropeReady = true;
            }

            // Managed IEEE-754 float32 -> float16 (round to nearest), same as Qwen3_5Model.
            static ushort F32ToF16(float value)
            {
                int i = BitConverter.SingleToInt32Bits(value);
                int s = (i >> 16) & 0x00008000;
                int e = ((i >> 23) & 0x000000ff) - (127 - 15);
                int m = i & 0x007fffff;
                if (e <= 0)
                {
                    if (e < -10) return (ushort)s;
                    m |= 0x00800000;
                    int t = 14 - e;
                    int a = (1 << (t - 1)) - 1;
                    int b = (m >> t) & 1;
                    m = (m + a + b) >> t;
                    return (ushort)(s | m);
                }
                else if (e == 0xff - (127 - 15))
                {
                    if (m == 0) return (ushort)(s | 0x7c00);
                    m >>= 13;
                    return (ushort)(s | 0x7c00 | m | (m == 0 ? 1 : 0));
                }
                else
                {
                    m = m + 0x00000fff + ((m >> 13) & 1);
                    if ((m & 0x00800000) != 0) { m = 0; e += 1; }
                    if (e > 30) return (ushort)(s | 0x7c00);
                    return (ushort)(s | (e << 10) | (m >> 13));
                }
            }

            // FP32 buffer management (same as Gemma3GPU)
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

                Realloc(ref hiddenBuf, sL * hiddenSize);
                Realloc(ref skipBuf, sL * hiddenSize);
                Realloc(ref normOutBuf, sL * hiddenSize);
                Realloc(ref qkvBuf, sL * qkvProjDim);
                Realloc(ref qBuf, sL * headsQ * headDim);
                Realloc(ref kBuf, sL * headsKV * headDim);
                Realloc(ref vBuf, sL * headsKV * headDim);
                Realloc(ref qNormBuf, sL * headsQ * headDim);
                Realloc(ref kNormBuf, sL * headsKV * headDim);
                Realloc(ref attnScoresBuf, headsQ * sL * kL);
                Realloc(ref attendedBuf, sL * headsQ * headDim);
                Realloc(ref attnOutBuf, sL * hiddenSize);
                Realloc(ref mlpInterBuf, sL * intermediateSize);
                Realloc(ref tokenIdsBuf, sL);

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

            // Quantized modes: bind the scale buffer next to its weight buffer (per output row for
            // INT8, per 32-weight group for INT4). No-op in FP16 — that shader variant strips the
            // scale resources, so the name doesn't exist there.
            void BindScales(int kernel, string name, ComputeBuffer scales)
            {
                if (Quant != LLMQuant.FP16 && scales != null) cs.SetBuffer(kernel, name, scales);  // embed/lm_head scales are null (always fp16)
            }

            // INT8 KV only: the attention read kernels dequantize K/V via the per-(token,head) scale/zp
            // buffers, so they MUST be bound (under the KV_INT8 keyword the shader references them; an
            // unbound StructuredBuffer is a D3D11 error). No-op for FP32/FP16 (those variants strip the
            // names, and cache.kScaleZp/vScaleZp are null). `li` is the current layer.
            void BindKvScaleZp(int kernel, int li)
            {
                if (KV != KVQuant.INT8) return;
                cs.SetBuffer(kernel, "k_scale_zp", cache.kScaleZp[li]);
                cs.SetBuffer(kernel, "v_scale_zp", cache.vScaleZp[li]);
            }

            // ---- layer dispatch (identical flow to Gemma3GPU, but uses packed FP16 weight buffer names) ----
            void DispatchLayer(int li, int seqLen, int totalKvLen, bool useCache)
            {
                bool isSW = Gemma3Modeling.Gemma3Config.layer_types[li] == Gemma3Modeling.GemmaLayerType.SlidingWindowAttention;
                int swSize = isSW ? slidingWindow : 0;
                var cosC = isSW ? ropeCosLocal : ropeCosFull;
                var sinC = isSW ? ropeSinLocal : ropeSinFull;
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int kvLen = useCache ? totalKvLen : seqLen;
                int hd2 = headDim / 2;
                int hidTotal = seqLen * hiddenSize;

                // 1. copy hidden → skip
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", hiddenBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                // 2. input layernorm
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.inputLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 3. QKV proj (FP16 weights)
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetInt("qkv_proj_dim", qkvProjDim);
                cs.SetBuffer(kQKVProj, "X", normOutBuf);
                cs.SetBuffer(kQKVProj, "W_QKV", weights.W_QKV[li]);
                BindScales(kQKVProj, "W_QKV_scales", weights.W_QKVScales[li]);
                cs.SetBuffer(kQKVProj, "QKV", qkvBuf);
                cs.Dispatch(kQKVProj, 1, (seqLen + 7) / 8, (qkvProjDim + 31) / 32);

                // 4. split QKV
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("qkv_proj_dim", qkvProjDim);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("num_heads_kv", headsKV);
                cs.SetInt("head_dim", headDim);
                cs.SetBuffer(kSplitQKV, "qkv_packed", qkvBuf);
                cs.SetBuffer(kSplitQKV, "split_q", qBuf);
                cs.SetBuffer(kSplitQKV, "split_k", kBuf);
                cs.SetBuffer(kSplitQKV, "split_v", vBuf);
                cs.Dispatch(kSplitQKV, Div256(seqLen * qkvProjDim), 1, 1);

                // 5. Q norm (FP16 gamma)
                int numVecsQ = seqLen * headsQ;
                cs.SetInt("num_vectors", numVecsQ);
                cs.SetInt("head_dim", headDim);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHead, "norm_input", qBuf);
                cs.SetBuffer(kRmsNormHead, "norm_output", qNormBuf);
                cs.SetBuffer(kRmsNormHead, "norm_gamma", weights.qNormGamma[li]);
                cs.Dispatch(kRmsNormHead, Div256(numVecsQ), 1, 1);

                // 6. K norm
                int numVecsK = seqLen * headsKV;
                cs.SetInt("num_vectors", numVecsK);
                cs.SetBuffer(kRmsNormHead, "norm_input", kBuf);
                cs.SetBuffer(kRmsNormHead, "norm_output", kNormBuf);
                cs.SetBuffer(kRmsNormHead, "norm_gamma", weights.kNormGamma[li]);
                cs.Dispatch(kRmsNormHead, Div256(numVecsK), 1, 1);

                // 7. RoPE Q (FP16 cos/sin)
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("head_dim", headDim);
                cs.SetInt("rope_num_heads", headsQ);
                cs.SetInt("position_offset", cacheLen);
                cs.SetBuffer(kApplyRope, "rope_buf", qNormBuf);
                cs.SetBuffer(kApplyRope, "rope_cos", cosC);
                cs.SetBuffer(kApplyRope, "rope_sin", sinC);
                cs.Dispatch(kApplyRope, (seqLen * headsQ * hd2 + 127) / 128, 1, 1);

                // 8. RoPE K
                cs.SetInt("rope_num_heads", headsKV);
                cs.SetBuffer(kApplyRope, "rope_buf", kNormBuf);
                cs.Dispatch(kApplyRope, (seqLen * headsKV * hd2 + 127) / 128, 1, 1);

                ComputeBuffer kForAttn, vForAttn;
                if (useCache)
                {
                    cs.SetInt("seq_len", seqLen);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetInt("cache_len", cacheLen);
                    // INT8 KV quantizes on write: ONE group per (token, kv-head) (the shader reduces
                    // min/max over head_dim there), and the per-(token,head) scale/zp is written through
                    // kv_scale_zp_w. FP32/FP16 write one uint per thread (WriteUnits = uint count).
                    bool kvInt8 = KV == KVQuant.INT8;
                    cs.SetBuffer(kWriteCacheFull, "kv_new", kNormBuf);
                    cs.SetBuffer(kWriteCacheFull, "kv_cache", cache.kCaches[li]);
                    if (kvInt8)
                    {
                        cs.SetBuffer(kWriteCacheFull, "kv_scale_zp_w", cache.kScaleZp[li]);
                        cs.Dispatch(kWriteCacheFull, seqLen * headsKV, 1, 1);
                    }
                    else cs.Dispatch(kWriteCacheFull, Div256(KVQuantUtil.WriteUnits(seqLen * headsKV * headDim, KV)), 1, 1);
                    cs.SetBuffer(kWriteCacheFull, "kv_new", vBuf);
                    cs.SetBuffer(kWriteCacheFull, "kv_cache", cache.vCaches[li]);
                    if (kvInt8)
                    {
                        cs.SetBuffer(kWriteCacheFull, "kv_scale_zp_w", cache.vScaleZp[li]);
                        cs.Dispatch(kWriteCacheFull, seqLen * headsKV, 1, 1);
                    }
                    else cs.Dispatch(kWriteCacheFull, Div256(KVQuantUtil.WriteUnits(seqLen * headsKV * headDim, KV)), 1, 1);
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
                    // 11-14 fused: FlashAttention (one threadgroup per query x head)
                    cs.SetInt("seq_len_q", seqLen);
                    cs.SetInt("seq_len_k", kvLen);
                    cs.SetInt("num_heads_q", headsQ);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetInt("sliding_window_size", swSize);
                    cs.SetInt("bidirectional", 0); // causal LM
                    cs.SetFloat("scale", attnScaling);
                    cs.SetBuffer(kFlashAttn, "Q", qNormBuf);
                    cs.SetBuffer(kFlashAttn, "K", kForAttn);
                    cs.SetBuffer(kFlashAttn, "V", vForAttn);
                    BindKvScaleZp(kFlashAttn, li);   // INT8 KV: K and V both dequantized here
                    cs.SetBuffer(kFlashAttn, "AttendedValues", attendedBuf);
                    cs.Dispatch(kFlashAttn, seqLen, headsQ, 1);
                }
                else
                {
                // 11. attention scores
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("sequence_length_k", kvLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("num_heads_kv", headsKV);
                cs.SetInt("head_dim", headDim);
                cs.SetFloat("scale", attnScaling);
                cs.SetBuffer(kAttnScores, "Q", qNormBuf);
                cs.SetBuffer(kAttnScores, "K", kForAttn);
                BindKvScaleZp(kAttnScores, li);   // INT8 KV: K dequantized here (binds both; only k used)
                cs.SetBuffer(kAttnScores, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kAttnScores, (seqLen + 3) / 4, (kvLen + 31) / 32, (headsQ + 3) / 4);

                // 12. mask
                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", kvLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("sliding_window_size", swSize);
                cs.SetInt("bidirectional", 0); // causal LM
                cs.SetBuffer(kApplyMask, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kApplyMask, (kvLen + 15) / 16, (headsQ * seqLen + 15) / 16, 1);

                // 13. softmax
                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", kvLen);
                cs.SetBuffer(kSoftmaxRows, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kSoftmaxRows, Div256(headsQ * seqLen), 1, 1);

                // 14. attend values
                cs.SetInt("sequence_length_v", kvLen);
                cs.SetBuffer(kAttendValues, "AttentionWeights", attnScoresBuf);
                cs.SetBuffer(kAttendValues, "V", vForAttn);
                BindKvScaleZp(kAttendValues, li);   // INT8 KV: V dequantized here (binds both; only v used)
                cs.SetBuffer(kAttendValues, "AttendedValues", attendedBuf);
                cs.Dispatch(kAttendValues, (headDim + 63) / 64, (seqLen + 3) / 4, (headsQ + 3) / 4);
                }

                // 15. O proj (FP16 weights)
                cs.SetInt("inner_embedding_dim", innerEmbDim);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetBuffer(kOProj, "AttendedValues", attendedBuf);
                cs.SetBuffer(kOProj, "W_O", weights.W_O[li]);
                BindScales(kOProj, "W_O_scales", weights.W_OScales[li]);
                cs.SetBuffer(kOProj, "O", attnOutBuf);
                cs.Dispatch(kOProj, 1, (seqLen + 3) / 4, (hiddenSize + 31) / 32);

                // 16. post-attn layernorm
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", attnOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postAttnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 17. residual
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kAddResidual, "buf_a", normOutBuf);
                cs.SetBuffer(kAddResidual, "buf_b", skipBuf);
                cs.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                // 18. copy normOut → skip
                cs.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", normOutBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                // 19. pre-FFN layernorm → hiddenBuf
                cs.SetInt("seq_len", seqLen);
                cs.SetBuffer(kRmsNormHidden, "norm_input", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.preFfnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 20-21. MLP (FP16 weights)
                bool vec1 = seqLen == 1;
                int kGU = vec1 ? kGateUp1Vec : kGateUp;
                int kDN = vec1 ? kDown1Vec : kDown;

                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("intermediate_size", intermediateSize);
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("activation_type", 1);
                cs.SetBuffer(kGU, "input", hiddenBuf);
                cs.SetBuffer(kGU, "mlp_weights", weights.mlpWeights[li]);
                BindScales(kGU, "mlp_scales", weights.mlpScales[li]);
                cs.SetBuffer(kGU, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kGU, (intermediateSize + 255) / 256, 1, 1);
                else cs.Dispatch(kGU, (intermediateSize + 63) / 64, (seqLen + 3) / 4, 1);

                cs.SetBuffer(kDN, "input", hiddenBuf);
                cs.SetBuffer(kDN, "mlp_weights", weights.mlpWeights[li]);
                BindScales(kDN, "mlp_scales", weights.mlpScales[li]);
                cs.SetBuffer(kDN, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kDN, (intermediateSize + 319) / 320, 1, 1);
                else cs.Dispatch(kDN, (hiddenSize + 31) / 32, (seqLen + 3) / 4, 1);

                // 22. post-FFN layernorm
                cs.SetInt("seq_len", seqLen);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postFfnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 23. residual
                cs.SetBuffer(kAddResidual, "buf_a", normOutBuf);
                cs.SetBuffer(kAddResidual, "buf_b", skipBuf);
                cs.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                // 24. normOut → hidden
                cs.SetBuffer(kCopyBuffer, "buf_a", hiddenBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", normOutBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);
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
                cs.SetFloat("embed_scale", embedScale);
                cs.SetBuffer(kEmbedLookup, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbedLookup, "embed_weights", weights.embedLmHead);
                BindScales(kEmbedLookup, "embed_scales", weights.embedScales);
                cs.SetBuffer(kEmbedLookup, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbedLookup, Div256(seqLen * hiddenSize), 1, 1);

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
                cs.SetFloat("embed_scale", embedScale);
                cs.SetBuffer(kEmbedLookup, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbedLookup, "embed_weights", weights.embedLmHead);
                BindScales(kEmbedLookup, "embed_scales", weights.embedScales);
                cs.SetBuffer(kEmbedLookup, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbedLookup, Div256(seqLen * hiddenSize), 1, 1);
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
                cs.SetBuffer(kRmsNormHidden, "norm_input", lastHiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normSingleBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsNormHidden, 1, 1, 1);

                Realloc(ref logitsBuf, vocabSize);
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("vocab_size", vocabSize);
                cs.SetBuffer(kLmHead1Vec, "lm_weights", weights.embedLmHead);
                BindScales(kLmHead1Vec, "embed_scales", weights.embedScales);
                cs.SetBuffer(kLmHead1Vec, "lm_input", normSingleBuf);
                cs.SetBuffer(kLmHead1Vec, "lm_output", logitsBuf);
                cs.Dispatch(kLmHead1Vec, (vocabSize + 511) / 512, 1, 1);
            }

            void DispatchFinalAll(int seqLen)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                Realloc(ref logitsBuf, seqLen * vocabSize);
                bool v1 = seqLen == 1;
                int k = v1 ? kLmHead1Vec : kLmHead;
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("vocab_size", vocabSize);
                cs.SetBuffer(k, "lm_weights", weights.embedLmHead);
                BindScales(k, "embed_scales", weights.embedScales);
                cs.SetBuffer(k, "lm_input", normOutBuf);
                cs.SetBuffer(k, "lm_output", logitsBuf);
                if (v1) cs.Dispatch(k, (vocabSize + 511) / 512, 1, 1);
                else cs.Dispatch(k, (vocabSize + 31) / 32, (seqLen + 7) / 8, 1);
            }

            // Queues the sampler kernel; the chosen token id lands in argmaxBuf on the GPU.
            // Reading it back is the caller's job (sync or async).
            void DispatchSampleKernels(float temperature, int topK, float topP, float minP)
            {
                cs.SetInt("vocab_size", vocabSize);
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
                    cs.SetBuffer(kSampleToken, "logits_buf", logitsBuf);
                    cs.SetBuffer(kSampleToken, "probs_buf", probsBuf);
                    cs.SetBuffer(kSampleToken, "argmax_result", argmaxBuf);
                    cs.Dispatch(kSampleToken, 1, 1, 1);
                }
            }

            public int SampleGreedy() => Sample(0f, 0, 1f, 0f);

            public int SampleStochastic(float temperature, int topK, float topP, float minP)
                => Sample(temperature, topK, topP, minP);

            // Synchronous sample: blocks the main thread until every queued GPU dispatch has
            // finished. Interactive paths should prefer SampleYielding.
            public int Sample(float temperature, int topK, float topP, float minP)
            {
                DispatchSampleKernels(temperature, topK, topP, minP);
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                return (int)r[0];
            }

            // Async sample: the 4-byte token id comes back via AsyncGPUReadback, so the main thread
            // never blocks on the in-flight GPU queue. Writes result[0].
            public IEnumerator SampleYielding(float temperature, int topK, float topP, float minP, int[] result)
            {
                DispatchSampleKernels(temperature, topK, topP, minP);
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
            {
                // Logits are FP32, read back directly
                return seqLen == 1
                    ? Tensor.Constant(logitsBuf, vocabSize)
                    : Tensor.Constant(logitsBuf, seqLen, vocabSize);
            }

            public void ResetCache() => cache.Reset();

            // Every buffer property name in Gemma3CS.compute; used by PrewarmKernels (SetBuffer on a
            // name a kernel doesn't use is a no-op; distinct dummies because D3D11 forbids the same
            // UAV in two slots of one dispatch).
            static readonly string[] WARMUP_BUFFER_NAMES =
            {
                "embed_weights", "norm_gamma", "W_QKV", "W_O", "mlp_weights", "lm_weights",
                "embed_scales", "W_QKV_scales", "W_O_scales", "mlp_scales",
                "rope_cos", "rope_sin",
                "token_ids", "embed_output", "norm_output", "norm_input",
                "qkv_packed", "split_q", "split_k", "split_v", "rope_buf",
                "kv_cache", "kv_new",
                "k_scale_zp", "v_scale_zp", "kv_scale_zp_w",   // INT8 KV variant only (no-op names otherwise)
                "X", "QKV", "Q", "K", "V",
                "AttentionWeights", "AttendedValues", "O",
                "input", "intermediate",
                "lm_input", "lm_output",
                "logits_buf", "probs_buf", "argmax_result",
                "buf_a", "buf_b",
            };

            // Every integer uniform that gates a kernel's thread guards; zeroed so every warmup
            // dispatch is degenerate (all threads early-out) and binding dummies is safe.
            static readonly string[] WARMUP_SIZE_UNIFORMS =
            {
                "seq_len", "hidden_size", "head_dim", "num_heads_q", "num_heads_kv", "vocab_size",
                "position_offset", "cache_len", "cache_capacity", "buffer_size", "qkv_proj_dim",
                "num_vectors", "rope_num_heads", "seq_len_q", "seq_len_k", "sliding_window_size",
                "bidirectional", "copy_src_offset", "embedding_dim", "inner_embedding_dim",
                "intermediate_size", "sequence_length_q", "sequence_length_k", "sequence_length_v",
                "batch_size", "top_k_val", "rng_seed", "activation_type",
            };

            static readonly string[] ALL_KERNEL_NAMES =
            {
                "EmbeddingLookup", "RmsNormHidden", "RmsNormHead", "SplitQKV",
                "ApplyRopeSplitHalf", "WriteCacheFull", "WriteCacheSliding", "ApplyMask", "SoftmaxRows",
                "QKVProj", "ComputeAttentionScores", "FlashAttention", "AttendValues", "OProj",
                "GateUp", "Down", "GateUp1Vec", "Down1Vec",
                "LmHeadPredict", "LmHeadPredict1Vec", "ArgMax", "SampleToken",
                "ZeroBuffer", "CopyBuffer", "CopySlice", "AddResidual",
            };

            static bool _kernelsPrewarmed;

            // The driver compiles each kernel's ISA on its FIRST dispatch — a one-time per-session
            // cost. This dispatches every kernel once with zero-size uniforms, ONE kernel compile
            // per frame. Static, needs no model or weights — run it at scene start (see
            // Gemma3ForCausalLM.Prewarm); Warmup() also runs it (idempotent).
            public static IEnumerator PrewarmKernels()
            {
                if (_kernelsPrewarmed) yield break;
                _kernelsPrewarmed = true;

                ComputeShader shader = DeepUnityMeta.Gemma3CS;

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

            bool _warmedUp;

            // Compiles every compute kernel behind the loading screen so the first real reply is
            // fast: one degenerate dispatch per kernel per frame (the driver-compile cost,
            // overlapping the weight stream), then throwaway forwards + both sampler paths over the
            // real code paths. Two token counts cover both kernel variants — multi-token (prefill)
            // and single-token (generation, the *1Vec kernels). Idempotent (runs once per model).
            public IEnumerator Warmup()
            {
                if (_warmedUp) yield break;

                var pk = PrewarmKernels();              // no weights needed — overlaps the upload
                while (pk.MoveNext()) yield return pk.Current;

                while (!IsReady) yield return null;

                int[] tok = new int[1];
                foreach (int n in new[] { 4, 1 })
                {
                    var e = ForwardYielding(Tensor.Constant(new float[n]), useCache: true, lastPosOnly: true);
                    while (e.MoveNext()) yield return e.Current;
                    // Async readback: a sync Sample here would block on the queued forward.
                    var s = SampleYielding(1f, 64, 0.95f, 0f, tok);
                    while (s.MoveNext()) yield return s.Current;
                }
                var g = SampleYielding(0f, 0, 1f, 0f, tok); // greedy (temperature==0) path
                while (g.MoveNext()) yield return g.Current;
                yield return null;

                ResetCache(); // undo the warmup's cache writes
                _warmedUp = true;
            }

            public void Dispose()
            {
                weights?.Dispose(); cache?.Dispose();
                ropeCosFull?.Release(); ropeSinFull?.Release();
                ropeCosLocal?.Release(); ropeSinLocal?.Release();
                hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release();
                qkvBuf?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release();
                qNormBuf?.Release(); kNormBuf?.Release();
                attnScoresBuf?.Release(); attendedBuf?.Release(); attnOutBuf?.Release();
                mlpInterBuf?.Release(); logitsBuf?.Release(); probsBuf?.Release();
                argmaxBuf?.Release(); tokenIdsBuf?.Release();
                lastHiddenBuf?.Release(); normSingleBuf?.Release();
            }
        }
    }
}
