using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // FP16 weights packed two halves per uint, stored on the GPU as ComputeBuffer<uint>.
        // Per-layer arrays are sized to NUM_LAYERS; entries are null for the wrong layer type.
        public class Qwen3_5Weights : IDisposable
        {
            // Tied embedding (also serves as lm_head) + final RMSNorm
            public ComputeBuffer embedLmHead;
            public ComputeBuffer finalNormGamma;

            // Per-layer common
            public ComputeBuffer[] inputLnGamma;
            public ComputeBuffer[] postAttnLnGamma;
            public ComputeBuffer[] mlpGate, mlpUp, mlpDown;

            // Full-attention layers (null on linear layers)
            public ComputeBuffer[] W_Q;        // [hidden, heads_q * head_dim * 2]   (Q + gate)
            public ComputeBuffer[] W_K;        // [hidden, heads_kv * head_dim]
            public ComputeBuffer[] W_V;        // [hidden, heads_kv * head_dim]
            public ComputeBuffer[] W_O;        // [heads_q * head_dim, hidden]
            public ComputeBuffer[] qNormGamma; // [head_dim]
            public ComputeBuffer[] kNormGamma; // [head_dim]

            // Linear (Gated DeltaNet) layers (null on full layers)
            public ComputeBuffer[] W_inProjQKV;     // [hidden, key_dim*2 + value_dim]
            public ComputeBuffer[] W_inProjZ;       // [hidden, value_dim]
            public ComputeBuffer[] W_inProjA;       // [hidden, num_v_heads]
            public ComputeBuffer[] W_inProjB;       // [hidden, num_v_heads]
            public ComputeBuffer[] convWeight;      // [conv_dim, kernel_size]   depthwise
            public ComputeBuffer[] dtBias;          // [num_v_heads]
            public ComputeBuffer[] ALog;            // [num_v_heads]
            public ComputeBuffer[] linearNormGamma; // [head_v_dim]
            public ComputeBuffer[] W_outProj;       // [value_dim, hidden]

            public bool IsReady { get; private set; }

            readonly int numLayers, hidden, headDim, headsQ, headsKV, intermediate, vocab;
            readonly int keyDim, valueDim, qkvLinDim, convDim, kernelSize, numVHeads, headVDim;
            readonly Qwen3_5LayerType[] layerTypes;

            const int EMBED_NUM_CHUNKS = 16;

            public Qwen3_5Weights(string paramsPath)
            {
                hidden = Qwen3_5Config.HIDDEN_SIZE;
                headDim = Qwen3_5Config.HEAD_DIM;
                headsQ = Qwen3_5Config.HEADS_Q;
                headsKV = Qwen3_5Config.HEADS_KV;
                intermediate = Qwen3_5Config.MLP_INTERMEDIATE_SIZE;
                vocab = Qwen3_5Config.VOCAB_SIZE;
                numLayers = Qwen3_5Config.NUM_LAYERS;
                layerTypes = Qwen3_5Config.layer_types;

                keyDim    = Qwen3_5Config.LINEAR_KEY_DIM;
                valueDim  = Qwen3_5Config.LINEAR_VALUE_DIM;
                qkvLinDim = keyDim * 2 + valueDim;
                convDim   = Qwen3_5Config.LINEAR_CONV_DIM;
                kernelSize = Qwen3_5Config.LINEAR_CONV_KERNEL_DIM;
                numVHeads = Qwen3_5Config.LINEAR_NUM_VALUE_HEADS;
                headVDim  = Qwen3_5Config.LINEAR_VALUE_HEAD_DIM;

                // Allocate
                embedLmHead    = HalfBuf(vocab * hidden);
                finalNormGamma = HalfBuf(hidden);

                inputLnGamma    = new ComputeBuffer[numLayers];
                postAttnLnGamma = new ComputeBuffer[numLayers];
                mlpGate = new ComputeBuffer[numLayers];
                mlpUp   = new ComputeBuffer[numLayers];
                mlpDown = new ComputeBuffer[numLayers];

                W_Q = new ComputeBuffer[numLayers];
                W_K = new ComputeBuffer[numLayers];
                W_V = new ComputeBuffer[numLayers];
                W_O = new ComputeBuffer[numLayers];
                qNormGamma = new ComputeBuffer[numLayers];
                kNormGamma = new ComputeBuffer[numLayers];

                W_inProjQKV = new ComputeBuffer[numLayers];
                W_inProjZ = new ComputeBuffer[numLayers];
                W_inProjA = new ComputeBuffer[numLayers];
                W_inProjB = new ComputeBuffer[numLayers];
                convWeight = new ComputeBuffer[numLayers];
                dtBias = new ComputeBuffer[numLayers];
                ALog = new ComputeBuffer[numLayers];
                linearNormGamma = new ComputeBuffer[numLayers];
                W_outProj = new ComputeBuffer[numLayers];

                int qProjOut = headsQ * headDim * 2; // 4096 (Q + gate)
                int kvProjOut = headsKV * headDim;   // 512

                for (int i = 0; i < numLayers; i++)
                {
                    inputLnGamma[i]    = HalfBuf(hidden);
                    postAttnLnGamma[i] = HalfBuf(hidden);
                    mlpGate[i] = HalfBuf(hidden * intermediate);
                    mlpUp[i]   = HalfBuf(hidden * intermediate);
                    mlpDown[i] = HalfBuf(intermediate * hidden);

                    if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                    {
                        W_Q[i] = HalfBuf(hidden * qProjOut);
                        W_K[i] = HalfBuf(hidden * kvProjOut);
                        W_V[i] = HalfBuf(hidden * kvProjOut);
                        W_O[i] = HalfBuf((headsQ * headDim) * hidden);
                        qNormGamma[i] = HalfBuf(headDim);
                        kNormGamma[i] = HalfBuf(headDim);
                    }
                    else
                    {
                        W_inProjQKV[i] = HalfBuf(hidden * qkvLinDim);
                        W_inProjZ[i]   = HalfBuf(hidden * valueDim);
                        W_inProjA[i]   = HalfBuf(hidden * numVHeads);
                        W_inProjB[i]   = HalfBuf(hidden * numVHeads);
                        convWeight[i]  = HalfBuf(convDim * kernelSize);
                        dtBias[i]      = HalfBuf(numVHeads);
                        ALog[i]        = HalfBuf(numVHeads);
                        linearNormGamma[i] = HalfBuf(headVDim);
                        W_outProj[i]   = HalfBuf(valueDim * hidden);
                    }
                }

                _ = LoadAllAsync(paramsPath);
            }

            // FP16 packed: 2 halves per 4-byte uint.
            static ComputeBuffer HalfBuf(int halfCount)
            {
                if ((halfCount & 1) != 0)
                    throw new ArgumentException($"HalfBuf needs even count, got {halfCount}");
                return new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            }

            static uint[] ReadFP16Packed(string path, int numHalves)
            {
                byte[] bytes = System.IO.File.ReadAllBytes(path);
                if (bytes.Length != numHalves * 2)
                    throw new System.IO.IOException($"Bad size {bytes.Length}, expected {numHalves * 2} for {path}");
                uint[] packed = new uint[numHalves / 2];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
                return packed;
            }

            async Task LoadAllAsync(string paramsPath)
            {
                Task tEmbed = LoadEmbeddingAsync(paramsPath);
                Task<uint[]> tNorm = Task.Run(() => ReadFP16Packed(paramsPath + "/norm.bin", hidden));

                Task[] layerTasks = new Task[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    int idx = i;
                    layerTasks[i] = LoadLayerAsync(paramsPath, idx);
                }

                await Task.WhenAll(tEmbed, tNorm, Task.WhenAll(layerTasks));
                finalNormGamma.SetData(tNorm.Result);
                IsReady = true;
                ConsoleMessage.Info("Qwen3.5 weights loaded.");
            }

            async Task LoadEmbeddingAsync(string paramsPath)
            {
                int totalHalves = vocab * hidden;
                int perChunk = totalHalves / EMBED_NUM_CHUNKS; // exactly divisible for 0.8B

                Task<uint[]>[] tasks = new Task<uint[]>[EMBED_NUM_CHUNKS];
                for (int i = 0; i < EMBED_NUM_CHUNKS; i++)
                {
                    int idx = i;
                    tasks[i] = Task.Run(() => ReadFP16Packed($"{paramsPath}/embed_tokens/part_{idx}.bin", perChunk));
                }
                uint[][] results = await Task.WhenAll(tasks);

                uint[] combined = await Task.Run(() =>
                {
                    int totalPacked = totalHalves / 2;
                    uint[] r = new uint[totalPacked];
                    int offset = 0;
                    for (int i = 0; i < EMBED_NUM_CHUNKS; i++)
                    {
                        Array.Copy(results[i], 0, r, offset, results[i].Length);
                        offset += results[i].Length;
                    }
                    return r;
                });

                embedLmHead.SetData(combined);
            }

            async Task LoadLayerAsync(string paramsPath, int i)
            {
                string lp = $"{paramsPath}/layer_{i}";

                int qProjOut = headsQ * headDim * 2;
                int kvProjOut = headsKV * headDim;
                int oIn = headsQ * headDim;

                var tInLn   = Task.Run(() => ReadFP16Packed(lp + "/input_layernorm.bin", hidden));
                var tPALn   = Task.Run(() => ReadFP16Packed(lp + "/post_attention_layernorm.bin", hidden));
                var tGate   = Task.Run(() => ReadFP16Packed(lp + "/mlp_gate_proj.bin", hidden * intermediate));
                var tUp     = Task.Run(() => ReadFP16Packed(lp + "/mlp_up_proj.bin",   hidden * intermediate));
                var tDown   = Task.Run(() => ReadFP16Packed(lp + "/mlp_down_proj.bin", intermediate * hidden));

                if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                {
                    var tQ  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_proj.bin", hidden * qProjOut));
                    var tK  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_proj.bin", hidden * kvProjOut));
                    var tV  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_v_proj.bin", hidden * kvProjOut));
                    var tO  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_o_proj.bin", oIn * hidden));
                    var tQN = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_norm.bin", headDim));
                    var tKN = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_norm.bin", headDim));

                    await Task.WhenAll(tInLn, tPALn, tGate, tUp, tDown, tQ, tK, tV, tO, tQN, tKN);

                    inputLnGamma[i].SetData(tInLn.Result);
                    postAttnLnGamma[i].SetData(tPALn.Result);
                    mlpGate[i].SetData(tGate.Result);
                    mlpUp[i].SetData(tUp.Result);
                    mlpDown[i].SetData(tDown.Result);
                    W_Q[i].SetData(tQ.Result);
                    W_K[i].SetData(tK.Result);
                    W_V[i].SetData(tV.Result);
                    W_O[i].SetData(tO.Result);
                    qNormGamma[i].SetData(tQN.Result);
                    kNormGamma[i].SetData(tKN.Result);
                }
                else
                {
                    var tQKV  = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_qkv.bin", hidden * qkvLinDim));
                    var tZ    = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_z.bin",   hidden * valueDim));
                    var tA    = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_a.bin",   hidden * numVHeads));
                    var tB    = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_b.bin",   hidden * numVHeads));
                    var tCv   = Task.Run(() => ReadFP16Packed(lp + "/linear_conv1d.bin",      convDim * kernelSize));
                    var tDt   = Task.Run(() => ReadFP16Packed(lp + "/linear_dt_bias.bin",     numVHeads));
                    var tAlog = Task.Run(() => ReadFP16Packed(lp + "/linear_A_log.bin",       numVHeads));
                    var tNm   = Task.Run(() => ReadFP16Packed(lp + "/linear_norm.bin",        headVDim));
                    var tOut  = Task.Run(() => ReadFP16Packed(lp + "/linear_out_proj.bin",    valueDim * hidden));

                    await Task.WhenAll(tInLn, tPALn, tGate, tUp, tDown, tQKV, tZ, tA, tB, tCv, tDt, tAlog, tNm, tOut);

                    inputLnGamma[i].SetData(tInLn.Result);
                    postAttnLnGamma[i].SetData(tPALn.Result);
                    mlpGate[i].SetData(tGate.Result);
                    mlpUp[i].SetData(tUp.Result);
                    mlpDown[i].SetData(tDown.Result);
                    W_inProjQKV[i].SetData(tQKV.Result);
                    W_inProjZ[i].SetData(tZ.Result);
                    W_inProjA[i].SetData(tA.Result);
                    W_inProjB[i].SetData(tB.Result);
                    convWeight[i].SetData(tCv.Result);
                    dtBias[i].SetData(tDt.Result);
                    ALog[i].SetData(tAlog.Result);
                    linearNormGamma[i].SetData(tNm.Result);
                    W_outProj[i].SetData(tOut.Result);
                }
            }

            public void Dispose()
            {
                embedLmHead?.Release();
                finalNormGamma?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    inputLnGamma[i]?.Release();
                    postAttnLnGamma[i]?.Release();
                    mlpGate[i]?.Release();
                    mlpUp[i]?.Release();
                    mlpDown[i]?.Release();
                    W_Q[i]?.Release();
                    W_K[i]?.Release();
                    W_V[i]?.Release();
                    W_O[i]?.Release();
                    qNormGamma[i]?.Release();
                    kNormGamma[i]?.Release();
                    W_inProjQKV[i]?.Release();
                    W_inProjZ[i]?.Release();
                    W_inProjA[i]?.Release();
                    W_inProjB[i]?.Release();
                    convWeight[i]?.Release();
                    dtBias[i]?.Release();
                    ALog[i]?.Release();
                    linearNormGamma[i]?.Release();
                    W_outProj[i]?.Release();
                }
            }
        }
    }
}
