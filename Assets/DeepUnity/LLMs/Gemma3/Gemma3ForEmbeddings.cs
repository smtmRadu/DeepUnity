// Gemma3ForEmbeddings — sentence-embedding inference using the embedding-gemma-300m
// architecture (24-layer Gemma3 trunk + mean-pool + 2-layer dense head + L2 norm).
// Reuses every kernel in `Gemma3CS.compute` (transformer trunk) and `FFNInferenceCS.compute`
// (post-trunk dense head). Bidirectional attention is enabled via the `bidirectional`
// uniform we added to `ApplyMask` in Gemma3CS.
//
// Public API:
//     var emb = new Gemma3ForEmbeddings();
//     yield return emb.EncodeQuery("hello world", v => Debug.Log(v.Norm()));
//
// Expected disk layout under `params_embedding/` (everything FP16 on disk, like params_it):
//     embed_tokens/part_{0..15}.bin    — packed FP16  [vocab*hidden]
//     norm.bin                          — packed FP16  [hidden]
//     dense_1.bin                       — packed FP16  [hf_intermediate * hidden]   (up: H -> I)
//     dense_2.bin                       — packed FP16  [hidden * hf_intermediate]   (down: I -> H)
//     layer_{i}/                        — same names as Gemma3 (params_it) + extra {pre,post}_feedforward_layernorm.bin
// FFNInferenceCS reads the dense-head weights as packed FP16 (StructuredBuffer<uint>),
// so end-to-end inference stays in FP16 for weights and FP32 for activations.
//
using System;
using System.Collections;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        // ----------------------------- CONFIG --------------------------------
        public static class EmbeddingGemmaConfig
        {
            public const int
                PAD_IDX = 0,
                EOS_IDX = 1,
                BOS_IDX = 2,
                VOCAB_SIZE = 262144,
                HIDDEN_SIZE = 768,
                MLP_INTERMEDIATE_SIZE = 1152,
                HEAD_FFN_INTERMEDIATE_SIZE = 3072,   // dense head (post-trunk)
                NUM_LAYERS = 24,
                MAX_POSITION_EMBEDDINGS = 2048,
                ROPE_LOCAL_BASE_FREQUENCY = 10_000,
                ROPE_THETA = 1_000_000,
                HEAD_DIM = 256,
                HEADS_Q = 3,
                HEADS_KV = 1,
                SLIDING_WINDOW = 512;
            public const float
                RMS_EPS = 1e-6f,
                QUERY_PRE_ATTENTION_SCALAR = 256f,
                ATTN_EXPANSION_FACTOR = 1f;     // inner_emb == hidden; q_proj 768->768 etc.

            public static readonly GemmaLayerType[] layer_types = new GemmaLayerType[]
            {
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.FullAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.FullAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.FullAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.SlidingWindowAttention,
                GemmaLayerType.SlidingWindowAttention, GemmaLayerType.FullAttention,
            };
        }

        // ----------------------------- WEIGHTS -------------------------------
        // Mirrors Gemma3Weights but sized from EmbeddingGemmaConfig and adds the
        // post-trunk dense head + the {pre,post}_feedforward_layernorm gammas.
        public class EmbeddingGemmaWeights : IDisposable
        {
            public ComputeBuffer embedTokens;          // FP16 [vocab*hidden]
            public ComputeBuffer[] W_QKV, W_O, mlpWeights;
            public ComputeBuffer[] qNormGamma, kNormGamma;
            public ComputeBuffer[] inputLnGamma, postAttnLnGamma, preFfnLnGamma, postFfnLnGamma;
            public ComputeBuffer finalNormGamma;
            public ComputeBuffer denseHeadWeights;     // packed FP16 — [up || down]

            public bool IsReady { get; private set; }

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize, denseInter;

            public EmbeddingGemmaWeights(string paramsPath)
            {
                numLayers = EmbeddingGemmaConfig.NUM_LAYERS;
                hiddenSize = EmbeddingGemmaConfig.HIDDEN_SIZE;
                headDim = EmbeddingGemmaConfig.HEAD_DIM;
                headsQ = EmbeddingGemmaConfig.HEADS_Q;
                headsKV = EmbeddingGemmaConfig.HEADS_KV;
                intermediateSize = EmbeddingGemmaConfig.MLP_INTERMEDIATE_SIZE;
                vocabSize = EmbeddingGemmaConfig.VOCAB_SIZE;
                denseInter = EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE;

                float exp = EmbeddingGemmaConfig.ATTN_EXPANSION_FACTOR;
                innerEmbDim = (int)(hiddenSize * exp);
                qkvProjDim = innerEmbDim + 2 * (innerEmbDim * headsKV / headsQ);

                embedTokens = HalfBuf(vocabSize * hiddenSize);

                W_QKV = new ComputeBuffer[numLayers];
                W_O = new ComputeBuffer[numLayers];
                mlpWeights = new ComputeBuffer[numLayers];
                qNormGamma = new ComputeBuffer[numLayers];
                kNormGamma = new ComputeBuffer[numLayers];
                inputLnGamma = new ComputeBuffer[numLayers];
                postAttnLnGamma = new ComputeBuffer[numLayers];
                preFfnLnGamma = new ComputeBuffer[numLayers];
                postFfnLnGamma = new ComputeBuffer[numLayers];

                for (int i = 0; i < numLayers; i++)
                {
                    W_QKV[i] = HalfBuf(hiddenSize * qkvProjDim);
                    W_O[i] = HalfBuf(innerEmbDim * hiddenSize);
                    mlpWeights[i] = HalfBuf(hiddenSize * intermediateSize * 3);
                    qNormGamma[i] = HalfBuf(headDim);
                    kNormGamma[i] = HalfBuf(headDim);
                    inputLnGamma[i] = HalfBuf(hiddenSize);
                    postAttnLnGamma[i] = HalfBuf(hiddenSize);
                    preFfnLnGamma[i] = HalfBuf(hiddenSize);
                    postFfnLnGamma[i] = HalfBuf(hiddenSize);
                }
                finalNormGamma = HalfBuf(hiddenSize);

                // Dense head: packed FP16, [up [denseInter, hidden] || down [hidden, denseInter]]
                // Total halves = 2 * hiddenSize * denseInter; packed into uint => /2 entries of 4 bytes.
                denseHeadWeights = new ComputeBuffer((2 * hiddenSize * denseInter) / 2, 4, ComputeBufferType.Structured);

                _ = LoadAllAsync(paramsPath);
            }

            // ---- helpers ----
            static ComputeBuffer HalfBuf(int halfCount) =>
                new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);

            static uint[] ReadFP16Packed(string path, int numHalves)
            {
                byte[] bytes = System.IO.File.ReadAllBytes(path);
                uint[] packed = new uint[numHalves / 2];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
                return packed;
            }


            async Task LoadAllAsync(string paramsPath)
            {
                Task embedTask = LoadEmbeddingAsync(paramsPath);
                Task headTask = LoadDenseHeadAsync(paramsPath);
                Task[] layerTasks = new Task[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    int idx = i;
                    layerTasks[i] = LoadLayerAsync(paramsPath, idx);
                }
                Task<uint[]> normTask = Task.Run(() => ReadFP16Packed(paramsPath + "/norm.bin", hiddenSize));

                await Task.WhenAll(embedTask, headTask, normTask, Task.WhenAll(layerTasks));
                finalNormGamma.SetData(normTask.Result);
                IsReady = true;
                ConsoleMessage.Info("EmbeddingGemma weights loaded.");
            }

            async Task LoadEmbeddingAsync(string paramsPath)
            {
                // Same chunked split as the original: 16 equal chunks of the flat embedding.
                const int NUM_CHUNKS = 16;
                int totalHalves = vocabSize * hiddenSize;
                int chunkHalves = totalHalves / NUM_CHUNKS;

                Task<uint[]>[] tasks = new Task<uint[]>[NUM_CHUNKS];
                for (int i = 0; i < NUM_CHUNKS; i++)
                {
                    int idx = i;
                    string path = $"{paramsPath}/embed_tokens/part_{idx}.bin";
                    tasks[idx] = Task.Run(() => ReadFP16Packed(path, chunkHalves));
                }
                uint[][] results = await Task.WhenAll(tasks);

                uint[] combined = await Task.Run(() =>
                {
                    int totalPacked = totalHalves / 2;
                    uint[] r = new uint[totalPacked];
                    int offset = 0;
                    for (int i = 0; i < NUM_CHUNKS; i++)
                    {
                        Array.Copy(results[i], 0, r, offset, results[i].Length);
                        offset += results[i].Length;
                    }
                    return r;
                });
                embedTokens.SetData(combined);
            }

            async Task LoadDenseHeadAsync(string paramsPath)
            {
                int n = hiddenSize * denseInter;
                Task<uint[]> tUp   = Task.Run(() => ReadFP16Packed($"{paramsPath}/dense_1.bin", n));
                Task<uint[]> tDown = Task.Run(() => ReadFP16Packed($"{paramsPath}/dense_2.bin", n));
                await Task.WhenAll(tUp, tDown);
                uint[] flat = await Task.Run(() => tUp.Result.Concat(tDown.Result).ToArray());
                denseHeadWeights.SetData(flat);
            }

            async Task LoadLayerAsync(string paramsPath, int layerIdx)
            {
                string lp = $"{paramsPath}/layer_{layerIdx}";
                int qSize = hiddenSize * innerEmbDim;
                int kvSize = hiddenSize * innerEmbDim * headsKV / headsQ;
                int oSize = innerEmbDim * hiddenSize;
                int mlpPart = hiddenSize * intermediateSize;

                var tQ    = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_proj.bin", qSize));
                var tK    = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_proj.bin", kvSize));
                var tV    = Task.Run(() => ReadFP16Packed(lp + "/self_attn_v_proj.bin", kvSize));
                var tO    = Task.Run(() => ReadFP16Packed(lp + "/self_attn_o_proj.bin", oSize));
                var tQN   = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_norm.bin", headDim));
                var tKN   = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_norm.bin", headDim));
                var tGate = Task.Run(() => ReadFP16Packed(lp + "/mlp_gate_proj.bin", mlpPart));
                var tUp   = Task.Run(() => ReadFP16Packed(lp + "/mlp_up_proj.bin", mlpPart));
                var tDown = Task.Run(() => ReadFP16Packed(lp + "/mlp_down_proj.bin", mlpPart));
                var tILn  = Task.Run(() => ReadFP16Packed(lp + "/input_layernorm.bin", hiddenSize));
                var tPALn = Task.Run(() => ReadFP16Packed(lp + "/post_attention_layernorm.bin", hiddenSize));
                var tPFLn = Task.Run(() => ReadFP16Packed(lp + "/pre_feedforward_layernorm.bin", hiddenSize));
                var tPPLn = Task.Run(() => ReadFP16Packed(lp + "/post_feedforward_layernorm.bin", hiddenSize));

                await Task.WhenAll(tQ, tK, tV, tO, tQN, tKN, tGate, tUp, tDown, tILn, tPALn, tPFLn, tPPLn);

                int qPacked = qSize / 2, kvPacked = kvSize / 2, mlpPacked = mlpPart / 2;
                int qkvFlatLen = (hiddenSize * qkvProjDim) / 2;
                int mlpFlatLen = (mlpPart * 3) / 2;

                var (flatQKV, flatMLP) = await Task.Run(() =>
                {
                    uint[] qkv = new uint[qkvFlatLen];
                    Array.Copy(tQ.Result, 0, qkv, 0, qPacked);
                    Array.Copy(tK.Result, 0, qkv, qPacked, kvPacked);
                    Array.Copy(tV.Result, 0, qkv, qPacked + kvPacked, kvPacked);

                    uint[] mlp = new uint[mlpFlatLen];
                    Array.Copy(tGate.Result, 0, mlp, 0, mlpPacked);
                    Array.Copy(tUp.Result,   0, mlp, mlpPacked, mlpPacked);
                    Array.Copy(tDown.Result, 0, mlp, 2 * mlpPacked, mlpPacked);
                    return (qkv, mlp);
                });

                W_QKV[layerIdx].SetData(flatQKV);
                W_O[layerIdx].SetData(tO.Result);
                mlpWeights[layerIdx].SetData(flatMLP);
                qNormGamma[layerIdx].SetData(tQN.Result);
                kNormGamma[layerIdx].SetData(tKN.Result);
                inputLnGamma[layerIdx].SetData(tILn.Result);
                postAttnLnGamma[layerIdx].SetData(tPALn.Result);
                preFfnLnGamma[layerIdx].SetData(tPFLn.Result);
                postFfnLnGamma[layerIdx].SetData(tPPLn.Result);
            }

            public void Dispose()
            {
                embedTokens?.Release();
                finalNormGamma?.Release();
                denseHeadWeights?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    W_QKV[i]?.Release(); W_O[i]?.Release(); mlpWeights[i]?.Release();
                    qNormGamma[i]?.Release(); kNormGamma[i]?.Release();
                    inputLnGamma[i]?.Release(); postAttnLnGamma[i]?.Release();
                    preFfnLnGamma[i]?.Release(); postFfnLnGamma[i]?.Release();
                }
            }
        }

        // ----------------------------- MODEL ---------------------------------
        // Mirrors Gemma3Model.DispatchLayer but with `bidirectional=1` on the mask
        // and skips the LM head (the trunk output goes to the dense head).
        public class EmbeddingGemmaModel : IDisposable
        {
            ComputeShader cs;
            ComputeShader denseCs;

            int kEmbedLookup, kRmsNormHidden, kRmsNormHead, kSplitQKV;
            int kApplyRope, kApplyMask, kSoftmaxRows;
            int kZeroBuffer, kCopyBuffer, kAddResidual;
            int kQKVProj, kAttnScores, kAttendValues, kOProj;
            int kGateUp, kDown, kGateUp1Vec, kDown1Vec;
            int kDenseUp1Vec, kDenseDown1Vec;

            public EmbeddingGemmaWeights weights;

            ComputeBuffer ropeCosFull, ropeSinFull, ropeCosLocal, ropeSinLocal;
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf;
            ComputeBuffer qkvBuf, qBuf, kBuf, vBuf, qNormBuf, kNormBuf;
            ComputeBuffer attnScoresBuf, attendedBuf, attnOutBuf;
            ComputeBuffer mlpInterBuf;
            ComputeBuffer tokenIdsBuf;
            ComputeBuffer pooledBuf, denseInterBuf;        // post-trunk dense head

            int curSeqAlloc;

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;
            readonly int slidingWindow, denseInter;
            readonly float rmsEps, embedScale, attnScaling;

            public bool IsReady => weights.IsReady;

            public EmbeddingGemmaModel(string paramsPath)
            {
                numLayers = EmbeddingGemmaConfig.NUM_LAYERS;
                hiddenSize = EmbeddingGemmaConfig.HIDDEN_SIZE;
                headDim = EmbeddingGemmaConfig.HEAD_DIM;
                headsQ = EmbeddingGemmaConfig.HEADS_Q;
                headsKV = EmbeddingGemmaConfig.HEADS_KV;
                intermediateSize = EmbeddingGemmaConfig.MLP_INTERMEDIATE_SIZE;
                vocabSize = EmbeddingGemmaConfig.VOCAB_SIZE;
                slidingWindow = EmbeddingGemmaConfig.SLIDING_WINDOW;
                denseInter = EmbeddingGemmaConfig.HEAD_FFN_INTERMEDIATE_SIZE;
                rmsEps = EmbeddingGemmaConfig.RMS_EPS;
                embedScale = MathF.Sqrt(hiddenSize);
                attnScaling = MathF.Pow(EmbeddingGemmaConfig.QUERY_PRE_ATTENTION_SCALAR, -0.5f);

                float exp = EmbeddingGemmaConfig.ATTN_EXPANSION_FACTOR;
                innerEmbDim = (int)(hiddenSize * exp);
                qkvProjDim = innerEmbDim + 2 * (innerEmbDim * headsKV / headsQ);

                cs = DeepUnityMeta.Gemma3CS;
                denseCs = DeepUnityMeta.FFNInferenceCS;
                CacheKernelIds();

                weights = new EmbeddingGemmaWeights(paramsPath);

                PrecomputeRoPE();

                pooledBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                denseInterBuf = new ComputeBuffer(denseInter, 4, ComputeBufferType.Structured);
            }

            void CacheKernelIds()
            {
                kEmbedLookup = cs.FindKernel("EmbeddingLookup");
                kRmsNormHidden = cs.FindKernel("RmsNormHidden");
                kRmsNormHead = cs.FindKernel("RmsNormHead");
                kSplitQKV = cs.FindKernel("SplitQKV");
                kApplyRope = cs.FindKernel("ApplyRopeSplitHalf");
                kApplyMask = cs.FindKernel("ApplyMask");
                kSoftmaxRows = cs.FindKernel("SoftmaxRows");
                kZeroBuffer = cs.FindKernel("ZeroBuffer");
                kCopyBuffer = cs.FindKernel("CopyBuffer");
                kAddResidual = cs.FindKernel("AddResidual");
                kQKVProj = cs.FindKernel("QKVProj");
                kAttnScores = cs.FindKernel("ComputeAttentionScores");
                kAttendValues = cs.FindKernel("AttendValues");
                kOProj = cs.FindKernel("OProj");
                kGateUp = cs.FindKernel("GateUp");
                kDown = cs.FindKernel("Down");
                kGateUp1Vec = cs.FindKernel("GateUp1Vec");
                kDown1Vec = cs.FindKernel("Down1Vec");

                kDenseUp1Vec = denseCs.FindKernel("Up1Vec");
                kDenseDown1Vec = denseCs.FindKernel("Down1Vec");
            }

            // RoPE precompute (same as Gemma3Model)
            static ComputeBuffer PackedHalfBuf(int halfCount) => new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            static void UploadHalfs(ComputeBuffer buf, ushort[] data)
            {
                uint[] packed = new uint[data.Length / 2];
                for (int i = 0; i < packed.Length; i++) packed[i] = (uint)data[2 * i] | ((uint)data[2 * i + 1] << 16);
                buf.SetData(packed);
            }

            void PrecomputeRoPE()
            {
                int maxSeq = EmbeddingGemmaConfig.MAX_POSITION_EMBEDDINGS;
                int hd2 = headDim / 2;
                int thetaFull = EmbeddingGemmaConfig.ROPE_THETA;
                int thetaLocal = EmbeddingGemmaConfig.ROPE_LOCAL_BASE_FREQUENCY;

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
                ropeCosFull  = PackedHalfBuf(maxSeq * hd2);
                ropeSinFull  = PackedHalfBuf(maxSeq * hd2);
                ropeCosLocal = PackedHalfBuf(maxSeq * hd2);
                ropeSinLocal = PackedHalfBuf(maxSeq * hd2);
                UploadHalfs(ropeCosFull,  cF); UploadHalfs(ropeSinFull,  sF);
                UploadHalfs(ropeCosLocal, cL); UploadHalfs(ropeSinLocal, sL);
            }

            void Realloc(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            void EnsureScratch(int seqLen)
            {
                if (seqLen <= curSeqAlloc) return;
                int sL = Math.Max(seqLen, curSeqAlloc);

                Realloc(ref hiddenBuf,    sL * hiddenSize);
                Realloc(ref skipBuf,      sL * hiddenSize);
                Realloc(ref normOutBuf,   sL * hiddenSize);
                Realloc(ref qkvBuf,       sL * qkvProjDim);
                Realloc(ref qBuf,         sL * headsQ * headDim);
                Realloc(ref kBuf,         sL * headsKV * headDim);
                Realloc(ref vBuf,         sL * headsKV * headDim);
                Realloc(ref qNormBuf,     sL * headsQ * headDim);
                Realloc(ref kNormBuf,     sL * headsKV * headDim);
                Realloc(ref attnScoresBuf, headsQ * sL * sL);
                Realloc(ref attendedBuf,  sL * headsQ * headDim);
                Realloc(ref attnOutBuf,   sL * hiddenSize);
                Realloc(ref mlpInterBuf,  sL * intermediateSize);
                Realloc(ref tokenIdsBuf,  sL);
                curSeqAlloc = sL;
            }

            void UploadTokens(Tensor ids, int seqLen)
            {
                uint[] arr = new uint[seqLen];
                for (int i = 0; i < seqLen; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);
            }

            static int Div256(int n) => (n + 255) / 256;

            // Identical layer pipeline as Gemma3Model.DispatchLayer except:
            //   - bidirectional = 1 (no causal mask; SW masks symmetrically)
            //   - no KV cache (single forward pass for one sequence)
            void DispatchLayer(int li, int seqLen)
            {
                bool isSW = EmbeddingGemmaConfig.layer_types[li] == GemmaLayerType.SlidingWindowAttention;
                int swSize = isSW ? slidingWindow : 0;
                var cosC = isSW ? ropeCosLocal : ropeCosFull;
                var sinC = isSW ? ropeSinLocal : ropeSinFull;
                int hd2 = headDim / 2;
                int hidTotal = seqLen * hiddenSize;

                // 1. copy hidden -> skip
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

                // 3. QKV proj
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

                // 5. Q norm / 6. K norm
                int numVecsQ = seqLen * headsQ;
                cs.SetInt("num_vectors", numVecsQ);
                cs.SetInt("head_dim", headDim);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHead, "norm_input", qBuf);
                cs.SetBuffer(kRmsNormHead, "norm_output", qNormBuf);
                cs.SetBuffer(kRmsNormHead, "norm_gamma", weights.qNormGamma[li]);
                cs.Dispatch(kRmsNormHead, Div256(numVecsQ), 1, 1);

                int numVecsK = seqLen * headsKV;
                cs.SetInt("num_vectors", numVecsK);
                cs.SetBuffer(kRmsNormHead, "norm_input", kBuf);
                cs.SetBuffer(kRmsNormHead, "norm_output", kNormBuf);
                cs.SetBuffer(kRmsNormHead, "norm_gamma", weights.kNormGamma[li]);
                cs.Dispatch(kRmsNormHead, Div256(numVecsK), 1, 1);

                // 7. RoPE Q / 8. RoPE K
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("head_dim", headDim);
                cs.SetInt("rope_num_heads", headsQ);
                cs.SetInt("position_offset", 0);
                cs.SetBuffer(kApplyRope, "rope_buf", qNormBuf);
                cs.SetBuffer(kApplyRope, "rope_cos", cosC);
                cs.SetBuffer(kApplyRope, "rope_sin", sinC);
                cs.Dispatch(kApplyRope, (seqLen * headsQ * hd2 + 127) / 128, 1, 1);

                cs.SetInt("rope_num_heads", headsKV);
                cs.SetBuffer(kApplyRope, "rope_buf", kNormBuf);
                cs.Dispatch(kApplyRope, (seqLen * headsKV * hd2 + 127) / 128, 1, 1);

                // 11. attention scores (no cache; K/V live in *NormBuf / vBuf)
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("sequence_length_k", seqLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("num_heads_kv", headsKV);
                cs.SetInt("head_dim", headDim);
                cs.SetFloat("scale", attnScaling);
                cs.SetBuffer(kAttnScores, "Q", qNormBuf);
                cs.SetBuffer(kAttnScores, "K", kNormBuf);
                cs.SetBuffer(kAttnScores, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kAttnScores, (seqLen + 3) / 4, (seqLen + 31) / 32, (headsQ + 3) / 4);

                // 12. mask — bidirectional, optional symmetric SW
                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", seqLen);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("sliding_window_size", swSize);
                cs.SetInt("bidirectional", 1);     // <-- the embedding-model difference
                cs.SetBuffer(kApplyMask, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kApplyMask, (seqLen + 15) / 16, (headsQ * seqLen + 15) / 16, 1);

                // 13. softmax
                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", seqLen);
                cs.SetBuffer(kSoftmaxRows, "AttentionWeights", attnScoresBuf);
                cs.Dispatch(kSoftmaxRows, Div256(headsQ * seqLen), 1, 1);

                // 14. attend values
                cs.SetInt("sequence_length_v", seqLen);
                cs.SetBuffer(kAttendValues, "AttentionWeights", attnScoresBuf);
                cs.SetBuffer(kAttendValues, "V", vBuf);
                cs.SetBuffer(kAttendValues, "AttendedValues", attendedBuf);
                cs.Dispatch(kAttendValues, (headDim + 63) / 64, (seqLen + 3) / 4, (headsQ + 3) / 4);

                // 15. O proj
                cs.SetInt("inner_embedding_dim", innerEmbDim);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetBuffer(kOProj, "AttendedValues", attendedBuf);
                cs.SetBuffer(kOProj, "W_O", weights.W_O[li]);
                cs.SetBuffer(kOProj, "O", attnOutBuf);
                cs.Dispatch(kOProj, 1, (seqLen + 3) / 4, (hiddenSize + 31) / 32);

                // 16-17. post-attn LN + residual
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", attnOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postAttnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kAddResidual, "buf_a", normOutBuf);
                cs.SetBuffer(kAddResidual, "buf_b", skipBuf);
                cs.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                // 18-19. copy + pre-FFN LN
                cs.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", normOutBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                cs.SetInt("seq_len", seqLen);
                cs.SetBuffer(kRmsNormHidden, "norm_input", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.preFfnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 20-21. MLP
                bool vec1 = seqLen == 1;
                int kGU = vec1 ? kGateUp1Vec : kGateUp;
                int kDN = vec1 ? kDown1Vec   : kDown;

                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("intermediate_size", intermediateSize);
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("activation_type", 1);   // gelu (Gemma3 hidden_act)
                cs.SetBuffer(kGU, "input", hiddenBuf);
                cs.SetBuffer(kGU, "mlp_weights", weights.mlpWeights[li]);
                cs.SetBuffer(kGU, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kGU, (intermediateSize + 255) / 256, 1, 1);
                else      cs.Dispatch(kGU, (intermediateSize + 63) / 64, (seqLen + 3) / 4, 1);

                cs.SetBuffer(kDN, "input", hiddenBuf);
                cs.SetBuffer(kDN, "mlp_weights", weights.mlpWeights[li]);
                cs.SetBuffer(kDN, "intermediate", mlpInterBuf);
                if (vec1) cs.Dispatch(kDN, (intermediateSize + 319) / 320, 1, 1);
                else      cs.Dispatch(kDN, (hiddenSize + 31) / 32, (seqLen + 3) / 4, 1);

                // 22-23-24. post-FFN LN + residual + copy back to hidden
                cs.SetInt("seq_len", seqLen);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postFfnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                cs.SetBuffer(kAddResidual, "buf_a", normOutBuf);
                cs.SetBuffer(kAddResidual, "buf_b", skipBuf);
                cs.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                cs.SetBuffer(kCopyBuffer, "buf_a", hiddenBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", normOutBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);
            }

            /// <summary>
            /// Runs the full embedding forward and returns a unit-norm sentence embedding.
            /// </summary>
            public Tensor Predict(Tensor input_ids)
            {
                int seqLen = input_ids.Size(-1);
                if (seqLen > EmbeddingGemmaConfig.MAX_POSITION_EMBEDDINGS)
                    throw new ArgumentException($"input_ids ({seqLen}) exceeds MAX_POSITION_EMBEDDINGS={EmbeddingGemmaConfig.MAX_POSITION_EMBEDDINGS}");

                EnsureScratch(seqLen);
                UploadTokens(input_ids, seqLen);

                // Embedding lookup with sqrt(hidden) scale (handled in kernel via embed_scale)
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("embed_scale", embedScale);
                cs.SetBuffer(kEmbedLookup, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbedLookup, "embed_weights", weights.embedTokens);
                cs.SetBuffer(kEmbedLookup, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbedLookup, Div256(seqLen * hiddenSize), 1, 1);

                for (int i = 0; i < numLayers; i++) DispatchLayer(i, seqLen);

                // Final RMSNorm -> normOutBuf
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // Mean-pool over seq -> pooledBuf [hidden]. Read back to CPU (hidden=768 fp32 ≈ 3KB),
                // average, upload to GPU. Single CPU-GPU bounce per call; trivial cost vs the trunk.
                float[] flat = new float[seqLen * hiddenSize];
                normOutBuf.GetData(flat, 0, 0, seqLen * hiddenSize);
                float[] pooled = new float[hiddenSize];
                for (int t = 0; t < seqLen; t++)
                    for (int h = 0; h < hiddenSize; h++)
                        pooled[h] += flat[t * hiddenSize + h];
                float invN = 1f / seqLen;
                for (int h = 0; h < hiddenSize; h++) pooled[h] *= invN;
                pooledBuf.SetData(pooled);

                // Dense head: Up1Vec (linear, no activation) -> denseInterBuf, then Down1Vec -> pooledBuf
                denseCs.SetInt("activation_type", -1);
                denseCs.SetInt("hidden_size", hiddenSize);
                denseCs.SetInt("intermediate_size", denseInter);
                denseCs.SetInt("batch_size", 1);
                denseCs.SetInt("seq_len", 1);
                denseCs.SetBuffer(kDenseUp1Vec, "input", pooledBuf);
                denseCs.SetBuffer(kDenseUp1Vec, "intermediate", denseInterBuf);
                denseCs.SetBuffer(kDenseUp1Vec, "weights", weights.denseHeadWeights);
                denseCs.Dispatch(kDenseUp1Vec, (denseInter + 255) / 256, 1, 1);

                denseCs.SetBuffer(kDenseDown1Vec, "input", pooledBuf);
                denseCs.SetBuffer(kDenseDown1Vec, "intermediate", denseInterBuf);
                denseCs.SetBuffer(kDenseDown1Vec, "weights", weights.denseHeadWeights);
                denseCs.Dispatch(kDenseDown1Vec, (hiddenSize + 319) / 320, 1, 1);

                // L2 normalize on CPU (single 768-vec)
                float[] outArr = new float[hiddenSize];
                pooledBuf.GetData(outArr);
                double sq = 0;
                for (int i = 0; i < hiddenSize; i++) sq += outArr[i] * outArr[i];
                float invNorm = 1f / MathF.Max((float)Math.Sqrt(sq), 1e-12f);
                for (int i = 0; i < hiddenSize; i++) outArr[i] *= invNorm;
                return Tensor.Constant(outArr);
            }

            public void Dispose()
            {
                weights?.Dispose();
                ropeCosFull?.Release(); ropeSinFull?.Release();
                ropeCosLocal?.Release(); ropeSinLocal?.Release();
                hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release();
                qkvBuf?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release();
                qNormBuf?.Release(); kNormBuf?.Release();
                attnScoresBuf?.Release(); attendedBuf?.Release(); attnOutBuf?.Release();
                mlpInterBuf?.Release(); tokenIdsBuf?.Release();
                pooledBuf?.Release(); denseInterBuf?.Release();
            }
        }
    }

    // ----------------------------- PUBLIC WRAPPER ----------------------------
    public class Gemma3ForEmbeddings
    {
        public Gemma3Modeling.EmbeddingGemmaModel model;
        public Gemma3TokenizerFast tokenizer;
        public bool IsReady => model.IsReady && tokenizer.IsReady;

        public Gemma3ForEmbeddings(
            string params_path = "Assets/DeepUnity/LLMs/Gemma3/params_embedding",
            string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json")
        {
            tokenizer = new Gemma3TokenizerFast(tokenizer_path, load_async: true);
#if UNITY_EDITOR
            UnityEditor.EditorApplication.playModeStateChanged += OnPlayModeChanged;
#endif
            Stopwatch sw = Stopwatch.StartNew();
            model = new Gemma3Modeling.EmbeddingGemmaModel(params_path);
            ConsoleMessage.Info($"Gemma3ForEmbeddings created ({sw.Elapsed.TotalSeconds:0.00} s)");
        }

        ~Gemma3ForEmbeddings() { model?.Dispose(); }

#if UNITY_EDITOR
        private void OnPlayModeChanged(UnityEditor.PlayModeStateChange state)
        {
            if (state == UnityEditor.PlayModeStateChange.ExitingPlayMode) model?.Dispose();
        }
#endif

        /// <summary>
        /// Encodes a prompt to a unit-norm sentence-embedding tensor of shape [HIDDEN_SIZE].
        /// `prompt` is appended with `&lt;eos&gt;` (the encoder also adds `&lt;bos&gt;` at the
        /// start) to match the sentence-transformers reference behavior.
        /// </summary>
        public IEnumerator EncodeQuery(string prompt, Action<Tensor> onEmbeddingReceived)
        {
            while (!IsReady) yield return new WaitForSeconds(0.01f);

            Stopwatch sw = Stopwatch.StartNew();
            prompt += "<eos>";
            (Tensor input_ids, _) = tokenizer.Encode(prompt,
                add_special_tokens: true,
                truncation: true,
                max_length: Gemma3Modeling.EmbeddingGemmaConfig.MAX_POSITION_EMBEDDINGS);
            yield return null;

            Tensor emb = model.Predict(input_ids);
            ConsoleMessage.Info($"Gemma3 embedding encoded ({sw.Elapsed.TotalSeconds:0.000} s, |x|={emb.Norm()[0]:0.000})");
            onEmbeddingReceived?.Invoke(emb);
        }
    }
}
