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
            ComputeShader cs;

            // kernel ids
            int kEmbedLookup, kRmsNormHidden, kRmsNormHead, kSplitQKV;
            int kApplyRope, kWriteCacheFull, kApplyMask, kSoftmaxRows;
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

            public bool IsReady => weights.IsReady;

            public Gemma3Model(string paramsPath, int cacheCapacity)
            {
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
                CacheKernelIds();

                weights = new Gemma3Weights(paramsPath);
                cache = new Gemma3Cache(numLayers, cacheCapacity, headsKV, headDim);

                PrecomputeRoPE();

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

            static void UploadHalfs(ComputeBuffer buf, ushort[] data)
            {
                uint[] packed = new uint[data.Length / 2];
                for (int i = 0; i < packed.Length; i++)
                    packed[i] = (uint)data[2 * i] | ((uint)data[2 * i + 1] << 16);
                buf.SetData(packed);
            }

            void PrecomputeRoPE()
            {
                int maxSeq = Gemma3Modeling.Gemma3Config.MAX_POSITION_EMBEDDINGS;
                int hd2 = headDim / 2;
                int thetaFull = Gemma3Modeling.Gemma3Config.ROPE_THETA;
                int thetaLocal = Gemma3Modeling.Gemma3Config.ROPE_LOCAL_BASE_FREQUENCY;

                ushort[] cF = new ushort[maxSeq * hd2];
                ushort[] sF = new ushort[maxSeq * hd2];
                ushort[] cL = new ushort[maxSeq * hd2];
                ushort[] sL = new ushort[maxSeq * hd2];

                Parallel.For(0, maxSeq, pos =>
                {
                    for (int i = 0; i < hd2; i++)
                    {
                        int idx = pos * hd2 + i;
                        float fF = 1f / MathF.Pow(thetaFull, 2f * i / headDim);
                        float fL = 1f / MathF.Pow(thetaLocal, 2f * i / headDim);
                        cF[idx] = Mathf.FloatToHalf(MathF.Cos(pos * fF));
                        sF[idx] = Mathf.FloatToHalf(MathF.Sin(pos * fF));
                        cL[idx] = Mathf.FloatToHalf(MathF.Cos(pos * fL));
                        sL[idx] = Mathf.FloatToHalf(MathF.Sin(pos * fL));
                    }
                });

                ropeCosFull = PackedHalfBuf(maxSeq * hd2);
                ropeSinFull = PackedHalfBuf(maxSeq * hd2);
                ropeCosLocal = PackedHalfBuf(maxSeq * hd2);
                ropeSinLocal = PackedHalfBuf(maxSeq * hd2);
                UploadHalfs(ropeCosFull, cF); UploadHalfs(ropeSinFull, sF);
                UploadHalfs(ropeCosLocal, cL); UploadHalfs(ropeSinLocal, sL);
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
                    cs.SetBuffer(kWriteCacheFull, "kv_new", kNormBuf);
                    cs.SetBuffer(kWriteCacheFull, "kv_cache", cache.kCaches[li]);
                    cs.Dispatch(kWriteCacheFull, Div256(seqLen * headsKV * headDim), 1, 1);
                    cs.SetBuffer(kWriteCacheFull, "kv_new", vBuf);
                    cs.SetBuffer(kWriteCacheFull, "kv_cache", cache.vCaches[li]);
                    cs.Dispatch(kWriteCacheFull, Div256(seqLen * headsKV * headDim), 1, 1);
                    kForAttn = cache.kCaches[li];
                    vForAttn = cache.vCaches[li];
                }
                else
                {
                    kForAttn = kNormBuf;
                    vForAttn = vBuf;
                }

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
                cs.SetBuffer(kAttendValues, "AttendedValues", attendedBuf);
                cs.Dispatch(kAttendValues, (headDim + 63) / 64, (seqLen + 3) / 4, (headsQ + 3) / 4);

                // 15. O proj (FP16 weights)
                cs.SetInt("inner_embedding_dim", innerEmbDim);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetBuffer(kOProj, "AttendedValues", attendedBuf);
                cs.SetBuffer(kOProj, "W_O", weights.W_O[li]);
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
                cs.SetBuffer(kGU, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kGU, (intermediateSize + 255) / 256, 1, 1);
                else cs.Dispatch(kGU, (intermediateSize + 63) / 64, (seqLen + 3) / 4, 1);

                cs.SetBuffer(kDN, "input", hiddenBuf);
                cs.SetBuffer(kDN, "mlp_weights", weights.mlpWeights[li]);
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
                cs.SetBuffer(k, "lm_input", normOutBuf);
                cs.SetBuffer(k, "lm_output", logitsBuf);
                if (v1) cs.Dispatch(k, (vocabSize + 511) / 512, 1, 1);
                else cs.Dispatch(k, (vocabSize + 31) / 32, (seqLen + 7) / 8, 1);
            }

            public int SampleGreedy()
            {
                cs.SetInt("vocab_size", vocabSize);
                cs.SetBuffer(kArgMax, "logits_buf", logitsBuf);
                cs.SetBuffer(kArgMax, "argmax_result", argmaxBuf);
                cs.Dispatch(kArgMax, 1, 1, 1);
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                return (int)r[0];
            }

            public int SampleStochastic(float temperature, int topK, float topP, float minP)
            {
                cs.SetInt("vocab_size", vocabSize);
                cs.SetFloat("temperature", temperature);
                cs.SetInt("top_k_val", topK);
                cs.SetFloat("top_p_val", topP);
                cs.SetFloat("min_p_val", minP);
                cs.SetInt("rng_seed", UnityEngine.Random.Range(int.MinValue, int.MaxValue));
                cs.SetBuffer(kSampleToken, "logits_buf", logitsBuf);
                cs.SetBuffer(kSampleToken, "probs_buf", probsBuf);
                cs.SetBuffer(kSampleToken, "argmax_result", argmaxBuf);
                cs.Dispatch(kSampleToken, 1, 1, 1);
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                return (int)r[0];
            }

            public int Sample(float temperature, int topK, float topP, float minP)
            {
                return temperature == 0f ? SampleGreedy() : SampleStochastic(temperature, topK, topP, minP);
            }

            public Tensor ReadLogits(int seqLen)
            {
                // Logits are FP32, read back directly
                return seqLen == 1
                    ? Tensor.Constant(logitsBuf, vocabSize)
                    : Tensor.Constant(logitsBuf, seqLen, vocabSize);
            }

            public void ResetCache() => cache.Reset();

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
