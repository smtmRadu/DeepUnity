using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3GPUFP32Modeling
    {
        public class Gemma3GPUFP32Model : IDisposable
        {
            // ---- shaders ----
            ComputeShader nextCS, gqaCS, gluCS, lmHeadCS;

            // ---- kernel ids (Gemma3NextCS) ----
            int kEmbedLookup, kRmsNormHidden, kRmsNormHead, kSplitQKV;
            int kApplyRope, kWriteCacheFull, kApplyMask, kSoftmaxRows;
            int kArgMax, kSampleToken, kZeroBuffer, kCopyBuffer, kCopySlice, kAddResidual;

            // ---- kernel ids (reused shaders) ----
            int kQKVProj, kAttnScores, kAttendValues, kOProj;
            int kGateUp, kDown, kGateUp1Vec, kDown1Vec;
            int kLmHead, kLmHead1Vec;

            // ---- sub-components ----
            public Gemma3GPUFP32Weights weights;
            public Gemma3GPUFP32Cache cache;

            // ---- RoPE cos/sin caches ----
            ComputeBuffer ropeCosFull, ropeSinFull;
            ComputeBuffer ropeCosLocal, ropeSinLocal;

            // ---- scratch buffers (resized as needed) ----
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf;
            ComputeBuffer qkvBuf, qBuf, kBuf, vBuf, qNormBuf, kNormBuf;
            ComputeBuffer attnScoresBuf, attendedBuf, attnOutBuf;
            ComputeBuffer mlpInterBuf;
            ComputeBuffer logitsBuf, probsBuf;
            ComputeBuffer argmaxBuf;       // uint[1]
            ComputeBuffer tokenIdsBuf;     // uint[seqLen]
            ComputeBuffer lastHiddenBuf;   // float[hiddenSize]  (for last-position extract)
            ComputeBuffer normSingleBuf;   // float[hiddenSize]

            int curSeqAlloc, curKvAlloc;

            // ---- config ----
            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;
            readonly int slidingWindow;
            readonly float rmsEps, embedScale, attnScaling;

            public bool IsReady => weights.IsReady;

            public Gemma3GPUFP32Model(string paramsPath, int cacheCapacity)
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

                // ---- shaders ----
                nextCS = DeepUnityMeta.Gemma3FP32CS;
                gqaCS = DeepUnityMeta.GQAInferenceCS;
                gluCS = DeepUnityMeta.GLUInferenceCS;
                lmHeadCS = DeepUnityMeta.LmHeadInferenceCS;

                CacheKernelIds();

                // ---- weights + cache ----
                weights = new Gemma3GPUFP32Weights(paramsPath);
                cache = new Gemma3GPUFP32Cache(numLayers, cacheCapacity, headsKV, headDim);

                // ---- RoPE precompute ----
                PrecomputeRoPE();

                // ---- fixed-size buffers ----
                probsBuf = new ComputeBuffer(vocabSize, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
            }

            void CacheKernelIds()
            {
                kEmbedLookup = nextCS.FindKernel("EmbeddingLookup");
                kRmsNormHidden = nextCS.FindKernel("RmsNormHidden");
                kRmsNormHead = nextCS.FindKernel("RmsNormHead");
                kSplitQKV = nextCS.FindKernel("SplitQKV");
                kApplyRope = nextCS.FindKernel("ApplyRopeSplitHalf");
                kWriteCacheFull = nextCS.FindKernel("WriteCacheFull");
                kApplyMask = nextCS.FindKernel("ApplyMask");
                kSoftmaxRows = nextCS.FindKernel("SoftmaxRows");
                kArgMax = nextCS.FindKernel("ArgMax");
                kSampleToken = nextCS.FindKernel("SampleToken");
                kZeroBuffer = nextCS.FindKernel("ZeroBuffer");
                kCopyBuffer = nextCS.FindKernel("CopyBuffer");
                kCopySlice = nextCS.FindKernel("CopySlice");
                kAddResidual = nextCS.FindKernel("AddResidual");

                kQKVProj = gqaCS.FindKernel("QKVProj");
                kAttnScores = gqaCS.FindKernel("ComputeAttentionScores");
                kAttendValues = gqaCS.FindKernel("AttendValues");
                kOProj = gqaCS.FindKernel("OProj");

                kGateUp = gluCS.FindKernel("GateUp");
                kDown = gluCS.FindKernel("Down");
                kGateUp1Vec = gluCS.FindKernel("GateUp1Vec");
                kDown1Vec = gluCS.FindKernel("Down1Vec");

                kLmHead = lmHeadCS.FindKernel("Predict");
                kLmHead1Vec = lmHeadCS.FindKernel("Predict1Vec");
            }

            void PrecomputeRoPE()
            {
                int maxSeq = Gemma3Modeling.Gemma3Config.MAX_POSITION_EMBEDDINGS;
                int hd2 = headDim / 2;
                int thetaFull = Gemma3Modeling.Gemma3Config.ROPE_THETA;
                int thetaLocal = Gemma3Modeling.Gemma3Config.ROPE_LOCAL_BASE_FREQUENCY;

                float[] cF = new float[maxSeq * hd2];
                float[] sF = new float[maxSeq * hd2];
                float[] cL = new float[maxSeq * hd2];
                float[] sL = new float[maxSeq * hd2];

                Parallel.For(0, maxSeq, pos =>
                {
                    for (int i = 0; i < hd2; i++)
                    {
                        int idx = pos * hd2 + i;
                        float fF = 1f / MathF.Pow(thetaFull, 2f * i / headDim);
                        float fL = 1f / MathF.Pow(thetaLocal, 2f * i / headDim);
                        float aF = pos * fF;
                        float aL = pos * fL;
                        cF[idx] = MathF.Cos(aF); sF[idx] = MathF.Sin(aF);
                        cL[idx] = MathF.Cos(aL); sL[idx] = MathF.Sin(aL);
                    }
                });

                ropeCosFull = new ComputeBuffer(maxSeq * hd2, 4, ComputeBufferType.Structured);
                ropeSinFull = new ComputeBuffer(maxSeq * hd2, 4, ComputeBufferType.Structured);
                ropeCosLocal = new ComputeBuffer(maxSeq * hd2, 4, ComputeBufferType.Structured);
                ropeSinLocal = new ComputeBuffer(maxSeq * hd2, 4, ComputeBufferType.Structured);
                ropeCosFull.SetData(cF); ropeSinFull.SetData(sF);
                ropeCosLocal.SetData(cL); ropeSinLocal.SetData(sL);
            }

            // ---- buffer management ----
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

            // ---- upload token ids ----
            void UploadTokens(Tensor ids, int seqLen)
            {
                uint[] arr = new uint[seqLen];
                for (int i = 0; i < seqLen; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);
            }

            // ---- dispatch helpers ----
            static int Div256(int n) => (n + 255) / 256;

            // ---- full layer dispatch ----
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
                nextCS.SetInt("buffer_size", hidTotal);
                nextCS.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                nextCS.SetBuffer(kCopyBuffer, "buf_b", hiddenBuf);
                nextCS.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                // 2. input layernorm
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("hidden_size", hiddenSize);
                nextCS.SetFloat("norm_eps", rmsEps);
                nextCS.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_gamma", weights.inputLnGamma[li]);
                nextCS.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 3. QKV proj (reuse GQAInferenceCS)
                gqaCS.SetInt("batch_size", 1);
                gqaCS.SetInt("sequence_length_q", seqLen);
                gqaCS.SetInt("embedding_dim", hiddenSize);
                gqaCS.SetInt("qkv_proj_dim", qkvProjDim);
                gqaCS.SetBuffer(kQKVProj, "X", normOutBuf);
                gqaCS.SetBuffer(kQKVProj, "W_QKV", weights.W_QKV[li]);
                gqaCS.SetBuffer(kQKVProj, "QKV", qkvBuf);
                gqaCS.Dispatch(kQKVProj, 1, (seqLen + 7) / 8, (qkvProjDim + 31) / 32);

                // 4. split QKV
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("qkv_proj_dim", qkvProjDim);
                nextCS.SetInt("num_heads_q", headsQ);
                nextCS.SetInt("num_heads_kv", headsKV);
                nextCS.SetInt("head_dim", headDim);
                nextCS.SetBuffer(kSplitQKV, "qkv_packed", qkvBuf);
                nextCS.SetBuffer(kSplitQKV, "split_q", qBuf);
                nextCS.SetBuffer(kSplitQKV, "split_k", kBuf);
                nextCS.SetBuffer(kSplitQKV, "split_v", vBuf);
                nextCS.Dispatch(kSplitQKV, Div256(seqLen * qkvProjDim), 1, 1);

                // 5. Q norm
                int numVecsQ = seqLen * headsQ;
                nextCS.SetInt("num_vectors", numVecsQ);
                nextCS.SetInt("head_dim", headDim);
                nextCS.SetFloat("norm_eps", rmsEps);
                nextCS.SetBuffer(kRmsNormHead, "norm_input", qBuf);
                nextCS.SetBuffer(kRmsNormHead, "norm_output", qNormBuf);
                nextCS.SetBuffer(kRmsNormHead, "norm_gamma", weights.qNormGamma[li]);
                nextCS.Dispatch(kRmsNormHead, Div256(numVecsQ), 1, 1);

                // 6. K norm
                int numVecsK = seqLen * headsKV;
                nextCS.SetInt("num_vectors", numVecsK);
                nextCS.SetBuffer(kRmsNormHead, "norm_input", kBuf);
                nextCS.SetBuffer(kRmsNormHead, "norm_output", kNormBuf);
                nextCS.SetBuffer(kRmsNormHead, "norm_gamma", weights.kNormGamma[li]);
                nextCS.Dispatch(kRmsNormHead, Div256(numVecsK), 1, 1);

                // 7. RoPE Q
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("head_dim", headDim);
                nextCS.SetInt("rope_num_heads", headsQ);
                nextCS.SetInt("position_offset", cacheLen);
                nextCS.SetBuffer(kApplyRope, "rope_buf", qNormBuf);
                nextCS.SetBuffer(kApplyRope, "rope_cos", cosC);
                nextCS.SetBuffer(kApplyRope, "rope_sin", sinC);
                nextCS.Dispatch(kApplyRope, (seqLen * headsQ * hd2 + 127) / 128, 1, 1);

                // 8. RoPE K
                nextCS.SetInt("rope_num_heads", headsKV);
                nextCS.SetBuffer(kApplyRope, "rope_buf", kNormBuf);
                nextCS.Dispatch(kApplyRope, (seqLen * headsKV * hd2 + 127) / 128, 1, 1);

                // K/V for attention
                ComputeBuffer kForAttn, vForAttn;

                if (useCache)
                {
                    // 9. write K cache
                    nextCS.SetInt("seq_len", seqLen);
                    nextCS.SetInt("num_heads_kv", headsKV);
                    nextCS.SetInt("head_dim", headDim);
                    nextCS.SetInt("cache_len", cacheLen);
                    nextCS.SetBuffer(kWriteCacheFull, "kv_new", kNormBuf);
                    nextCS.SetBuffer(kWriteCacheFull, "kv_cache", cache.kCaches[li]);
                    nextCS.Dispatch(kWriteCacheFull, Div256(seqLen * headsKV * headDim), 1, 1);

                    // 10. write V cache
                    nextCS.SetBuffer(kWriteCacheFull, "kv_new", vBuf);
                    nextCS.SetBuffer(kWriteCacheFull, "kv_cache", cache.vCaches[li]);
                    nextCS.Dispatch(kWriteCacheFull, Div256(seqLen * headsKV * headDim), 1, 1);

                    kForAttn = cache.kCaches[li];
                    vForAttn = cache.vCaches[li];
                }
                else
                {
                    kForAttn = kNormBuf;
                    vForAttn = vBuf;
                }

                // 11. attention scores (reuse GQAInferenceCS)
                gqaCS.SetInt("batch_size", 1);
                gqaCS.SetInt("sequence_length_q", seqLen);
                gqaCS.SetInt("sequence_length_k", kvLen);
                gqaCS.SetInt("num_heads_q", headsQ);
                gqaCS.SetInt("num_heads_kv", headsKV);
                gqaCS.SetInt("head_dim", headDim);
                gqaCS.SetFloat("scale", attnScaling);
                gqaCS.SetBuffer(kAttnScores, "Q", qNormBuf);
                gqaCS.SetBuffer(kAttnScores, "K", kForAttn);
                gqaCS.SetBuffer(kAttnScores, "AttentionWeights", attnScoresBuf);
                gqaCS.Dispatch(kAttnScores, (seqLen + 3) / 4, (kvLen + 31) / 32, (headsQ + 3) / 4);

                // 12. mask
                nextCS.SetInt("seq_len_q", seqLen);
                nextCS.SetInt("seq_len_k", kvLen);
                nextCS.SetInt("num_heads_q", headsQ);
                nextCS.SetInt("sliding_window_size", swSize);
                nextCS.SetBuffer(kApplyMask, "attn_scores", attnScoresBuf);
                nextCS.Dispatch(kApplyMask, (kvLen + 15) / 16, (headsQ * seqLen + 15) / 16, 1);

                // 13. softmax
                nextCS.SetBuffer(kSoftmaxRows, "attn_scores", attnScoresBuf);
                nextCS.Dispatch(kSoftmaxRows, Div256(headsQ * seqLen), 1, 1);

                // 14. attend values (reuse GQAInferenceCS)
                gqaCS.SetInt("sequence_length_v", kvLen);
                gqaCS.SetBuffer(kAttendValues, "AttentionWeights", attnScoresBuf);
                gqaCS.SetBuffer(kAttendValues, "V", vForAttn);
                gqaCS.SetBuffer(kAttendValues, "AttendedValues", attendedBuf);
                gqaCS.Dispatch(kAttendValues, (headDim + 63) / 64, (seqLen + 3) / 4, (headsQ + 3) / 4);

                // 15. O proj (reuse GQAInferenceCS)
                gqaCS.SetInt("inner_embedding_dim", innerEmbDim);
                gqaCS.SetInt("embedding_dim", hiddenSize);
                gqaCS.SetBuffer(kOProj, "AttendedValues", attendedBuf);
                gqaCS.SetBuffer(kOProj, "W_O", weights.W_O[li]);
                gqaCS.SetBuffer(kOProj, "O", attnOutBuf);
                gqaCS.Dispatch(kOProj, 1, (seqLen + 3) / 4, (hiddenSize + 31) / 32);

                // 16. post-attention layernorm
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("hidden_size", hiddenSize);
                nextCS.SetFloat("norm_eps", rmsEps);
                nextCS.SetBuffer(kRmsNormHidden, "norm_input", attnOutBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postAttnLnGamma[li]);
                nextCS.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 17. residual: normOut += skip
                nextCS.SetInt("buffer_size", hidTotal);
                nextCS.SetBuffer(kAddResidual, "buf_a", normOutBuf);
                nextCS.SetBuffer(kAddResidual, "buf_b", skipBuf);
                nextCS.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                // 18. copy normOut → skip (for MLP residual)
                nextCS.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                nextCS.SetBuffer(kCopyBuffer, "buf_b", normOutBuf);
                nextCS.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                // 19. pre-FFN layernorm → hiddenBuf
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetBuffer(kRmsNormHidden, "norm_input", normOutBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_output", hiddenBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_gamma", weights.preFfnLnGamma[li]);
                nextCS.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 20-21. MLP (reuse GLUInferenceCS)
                bool vec1 = seqLen == 1;
                int kGU = vec1 ? kGateUp1Vec : kGateUp;
                int kDN = vec1 ? kDown1Vec : kDown;

                gluCS.SetInt("hidden_size", hiddenSize);
                gluCS.SetInt("intermediate_size", intermediateSize);
                gluCS.SetInt("batch_size", 1);
                gluCS.SetInt("seq_len", seqLen);
                gluCS.SetInt("activation_type", 1); // gelu_tanh
                gluCS.SetBuffer(kGU, "input", hiddenBuf);
                gluCS.SetBuffer(kGU, "weights", weights.mlpWeights[li]);
                gluCS.SetBuffer(kGU, "intermediate", mlpInterBuf);

                if (vec1)
                    gluCS.Dispatch(kGU, (intermediateSize + 255) / 256, 1, 1);
                else
                    gluCS.Dispatch(kGU, (intermediateSize + 63) / 64, (seqLen + 3) / 4, 1);

                gluCS.SetBuffer(kDN, "input", hiddenBuf);
                gluCS.SetBuffer(kDN, "weights", weights.mlpWeights[li]);
                gluCS.SetBuffer(kDN, "intermediate", mlpInterBuf);

                if (vec1)
                    gluCS.Dispatch(kDN, (intermediateSize + 319) / 320, 1, 1);
                else
                    gluCS.Dispatch(kDN, (hiddenSize + 31) / 32, (seqLen + 3) / 4, 1);

                // 22. post-FFN layernorm
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postFfnLnGamma[li]);
                nextCS.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 23. residual: normOut += skip
                nextCS.SetBuffer(kAddResidual, "buf_a", normOutBuf);
                nextCS.SetBuffer(kAddResidual, "buf_b", skipBuf);
                nextCS.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                // 24. normOut → hidden (next layer input)
                nextCS.SetBuffer(kCopyBuffer, "buf_a", hiddenBuf);
                nextCS.SetBuffer(kCopyBuffer, "buf_b", normOutBuf);
                nextCS.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);
            }

            // ---- forward (blocking) ----
            public void Forward(Tensor input_ids, bool useCache, bool lastPosOnly)
            {
                int seqLen = input_ids.Size(-1);
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int totalKvLen = cacheLen + seqLen;

                EnsureScratch(seqLen, totalKvLen);
                UploadTokens(input_ids, seqLen);

                // embedding
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("hidden_size", hiddenSize);
                nextCS.SetFloat("embed_scale", embedScale);
                nextCS.SetBuffer(kEmbedLookup, "token_ids", tokenIdsBuf);
                nextCS.SetBuffer(kEmbedLookup, "embed_weights", weights.embedLmHead);
                nextCS.SetBuffer(kEmbedLookup, "embed_output", hiddenBuf);
                nextCS.Dispatch(kEmbedLookup, Div256(seqLen * hiddenSize), 1, 1);

                // layers
                for (int i = 0; i < numLayers; i++)
                    DispatchLayer(i, seqLen, totalKvLen, useCache);

                if (useCache)
                    cache.CachedTokenCount += seqLen;

                // final norm + lm_head
                if (lastPosOnly)
                    DispatchFinalNormAndLmHeadLast(seqLen);
                else
                    DispatchFinalNormAndLmHeadAll(seqLen);
            }

            // ---- coroutine forward (yields between layers) ----
            public IEnumerator ForwardYielding(Tensor input_ids, bool useCache, bool lastPosOnly)
            {
                int seqLen = input_ids.Size(-1);
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int totalKvLen = cacheLen + seqLen;

                EnsureScratch(seqLen, totalKvLen);
                UploadTokens(input_ids, seqLen);

                // embedding
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("hidden_size", hiddenSize);
                nextCS.SetFloat("embed_scale", embedScale);
                nextCS.SetBuffer(kEmbedLookup, "token_ids", tokenIdsBuf);
                nextCS.SetBuffer(kEmbedLookup, "embed_weights", weights.embedLmHead);
                nextCS.SetBuffer(kEmbedLookup, "embed_output", hiddenBuf);
                nextCS.Dispatch(kEmbedLookup, Div256(seqLen * hiddenSize), 1, 1);
                yield return null;

                for (int i = 0; i < numLayers; i++)
                {
                    DispatchLayer(i, seqLen, totalKvLen, useCache);
                    yield return null;
                }

                if (useCache)
                    cache.CachedTokenCount += seqLen;

                if (lastPosOnly)
                    DispatchFinalNormAndLmHeadLast(seqLen);
                else
                    DispatchFinalNormAndLmHeadAll(seqLen);
                yield return null;
            }

            void DispatchFinalNormAndLmHeadLast(int seqLen)
            {
                // extract last hidden vector
                nextCS.SetInt("buffer_size", hiddenSize);
                nextCS.SetInt("copy_src_offset", (seqLen - 1) * hiddenSize);
                nextCS.SetBuffer(kCopySlice, "buf_a", lastHiddenBuf);
                nextCS.SetBuffer(kCopySlice, "buf_b", hiddenBuf);
                nextCS.Dispatch(kCopySlice, Div256(hiddenSize), 1, 1);

                // final norm on 1 position
                nextCS.SetInt("seq_len", 1);
                nextCS.SetInt("hidden_size", hiddenSize);
                nextCS.SetFloat("norm_eps", rmsEps);
                nextCS.SetBuffer(kRmsNormHidden, "norm_input", lastHiddenBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_output", normSingleBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                nextCS.Dispatch(kRmsNormHidden, 1, 1, 1);

                // lm_head for 1 position
                Realloc(ref logitsBuf, vocabSize);
                int k = kLmHead1Vec;
                lmHeadCS.SetInt("batch_size", 1);
                lmHeadCS.SetInt("seq_len", 1);
                lmHeadCS.SetInt("hidden_size", hiddenSize);
                lmHeadCS.SetInt("vocab_size", vocabSize);
                lmHeadCS.SetBuffer(k, "weights", weights.embedLmHead);
                lmHeadCS.SetBuffer(k, "input", normSingleBuf);
                lmHeadCS.SetBuffer(k, "output", logitsBuf);
                lmHeadCS.Dispatch(k, (vocabSize + 511) / 512, 1, 1);
            }

            void DispatchFinalNormAndLmHeadAll(int seqLen)
            {
                // final norm on all positions
                nextCS.SetInt("seq_len", seqLen);
                nextCS.SetInt("hidden_size", hiddenSize);
                nextCS.SetFloat("norm_eps", rmsEps);
                nextCS.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                nextCS.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                nextCS.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // lm_head for all positions
                Realloc(ref logitsBuf, seqLen * vocabSize);
                bool v1 = seqLen == 1;
                int k = v1 ? kLmHead1Vec : kLmHead;
                lmHeadCS.SetInt("batch_size", 1);
                lmHeadCS.SetInt("seq_len", seqLen);
                lmHeadCS.SetInt("hidden_size", hiddenSize);
                lmHeadCS.SetInt("vocab_size", vocabSize);
                lmHeadCS.SetBuffer(k, "weights", weights.embedLmHead);
                lmHeadCS.SetBuffer(k, "input", normOutBuf);
                lmHeadCS.SetBuffer(k, "output", logitsBuf);

                if (v1)
                    lmHeadCS.Dispatch(k, (vocabSize + 511) / 512, 1, 1);
                else
                    lmHeadCS.Dispatch(k, (vocabSize + 31) / 32, (seqLen + 7) / 8, 1);
            }

            // ---- sampling ----
            public int SampleGreedy()
            {
                nextCS.SetInt("vocab_size", vocabSize);
                nextCS.SetBuffer(kArgMax, "logits_buf", logitsBuf);
                nextCS.SetBuffer(kArgMax, "argmax_result", argmaxBuf);
                nextCS.Dispatch(kArgMax, 1, 1, 1);

                uint[] result = new uint[1];
                argmaxBuf.GetData(result);
                return (int)result[0];
            }

            public int SampleStochastic(float temperature, int topK, float topP, float minP)
            {
                nextCS.SetInt("vocab_size", vocabSize);
                nextCS.SetFloat("temperature", temperature);
                nextCS.SetInt("top_k_val", topK);
                nextCS.SetFloat("top_p_val", topP);
                nextCS.SetFloat("min_p_val", minP);
                nextCS.SetInt("rng_seed", UnityEngine.Random.Range(int.MinValue, int.MaxValue));
                nextCS.SetBuffer(kSampleToken, "logits_buf", logitsBuf);
                nextCS.SetBuffer(kSampleToken, "probs_buf", probsBuf);
                nextCS.SetBuffer(kSampleToken, "argmax_result", argmaxBuf);
                nextCS.Dispatch(kSampleToken, 1, 1, 1);

                uint[] result = new uint[1];
                argmaxBuf.GetData(result);
                return (int)result[0];
            }

            public int Sample(float temperature, int topK, float topP, float minP)
            {
                return temperature == 0f ? SampleGreedy() : SampleStochastic(temperature, topK, topP, minP);
            }

            // ---- read back logits for Predict API ----
            public Tensor ReadLogits(int seqLen)
            {
                return seqLen == 1
                    ? Tensor.Constant(logitsBuf, vocabSize)
                    : Tensor.Constant(logitsBuf, seqLen, vocabSize);
            }

            // ---- lifecycle ----
            public void ResetCache() => cache.Reset();

            public void Dispose()
            {
                weights?.Dispose();
                cache?.Dispose();

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
