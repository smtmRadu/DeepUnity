using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class Gemma3Weights : IDisposable
        {
            public ComputeBuffer embedLmHead;

            public ComputeBuffer[] W_QKV;
            public ComputeBuffer[] W_O;
            public ComputeBuffer[] mlpWeights;
            public ComputeBuffer[] qNormGamma;
            public ComputeBuffer[] kNormGamma;
            public ComputeBuffer[] inputLnGamma;
            public ComputeBuffer[] postAttnLnGamma;
            public ComputeBuffer[] preFfnLnGamma;
            public ComputeBuffer[] postFfnLnGamma;
            public ComputeBuffer finalNormGamma;

            public bool IsReady { get; private set; }

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;

            public Gemma3Weights(string paramsPath)
            {
                numLayers = Gemma3Modeling.Gemma3Config.NUM_LAYERS;
                hiddenSize = Gemma3Modeling.Gemma3Config.HIDDEN_SIZE;
                headDim = Gemma3Modeling.Gemma3Config.HEAD_DIM;
                headsQ = Gemma3Modeling.Gemma3Config.HEADS_Q;
                headsKV = Gemma3Modeling.Gemma3Config.HEADS_KV;
                intermediateSize = Gemma3Modeling.Gemma3Config.MLP_INTERMEDIATE_SIZE;
                vocabSize = Gemma3Modeling.Gemma3Config.VOCAB_SIZE;

                float exp = Gemma3Modeling.Gemma3Config.ATTN_EXPANSION_FACTOR;
                innerEmbDim = (int)(hiddenSize * exp);
                qkvProjDim = innerEmbDim + 2 * (innerEmbDim * headsKV / headsQ);

                embedLmHead = HalfBuf(vocabSize * hiddenSize);

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

                _ = LoadAllAsync(paramsPath);
            }

            // ---- FP16 buffer helpers ----
            static ComputeBuffer HalfBuf(int halfCount)
            {
                return new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            }

            // Reads a .bin of packed FP16 directly into uint[] (2 halves per uint).
            // Avoids the ushort->uint pack loop on the main thread.
            static uint[] ReadFP16Packed(string path, int numHalves)
            {
                byte[] bytes = System.IO.File.ReadAllBytes(path);
                uint[] packed = new uint[numHalves / 2];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
                return packed;
            }

            // ---- async loading ----
            async Task LoadAllAsync(string paramsPath)
            {
                Task embedTask = LoadEmbeddingAsync(paramsPath);
                Task[] layerTasks = new Task[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    int idx = i;
                    layerTasks[i] = LoadLayerAsync(paramsPath, idx);
                }
                Task<uint[]> normTask = Task.Run(() => ReadFP16Packed(paramsPath + "/norm.bin", hiddenSize));

                await Task.WhenAll(embedTask, normTask, Task.WhenAll(layerTasks));
                finalNormGamma.SetData(normTask.Result);
                IsReady = true;
                ConsoleMessage.Info("Gemma3 weights loaded.");
            }

            async Task LoadEmbeddingAsync(string paramsPath)
            {
                int[] partSizes = new int[]
                {
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_722
                };

                Task<uint[]>[] tasks = new Task<uint[]>[14];
                for (int i = 0; i < 14; i++)
                {
                    int size = partSizes[i];
                    string path = $"{paramsPath}/lm_head/part_{i}.bin";
                    tasks[i] = Task.Run(() => ReadFP16Packed(path, size));
                }

                uint[][] results = await Task.WhenAll(tasks);

                // Combine off the main thread; SetData on main thread is the only sync work.
                uint[] combined = await Task.Run(() =>
                {
                    int totalPacked = (vocabSize * hiddenSize) / 2;
                    uint[] r = new uint[totalPacked];
                    int offset = 0;
                    for (int i = 0; i < 14; i++)
                    {
                        Array.Copy(results[i], 0, r, offset, results[i].Length);
                        offset += results[i].Length;
                    }
                    return r;
                });

                embedLmHead.SetData(combined);
            }

            async Task LoadLayerAsync(string paramsPath, int layerIdx)
            {
                string lp = $"{paramsPath}/layer_{layerIdx}";

                int qSize = hiddenSize * innerEmbDim;
                int kvSize = hiddenSize * innerEmbDim * headsKV / headsQ;
                int oSize = innerEmbDim * hiddenSize;
                int mlpPart = hiddenSize * intermediateSize;

                var tQ = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_proj.bin", qSize));
                var tK = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_proj.bin", kvSize));
                var tV = Task.Run(() => ReadFP16Packed(lp + "/self_attn_v_proj.bin", kvSize));
                var tO = Task.Run(() => ReadFP16Packed(lp + "/self_attn_o_proj.bin", oSize));
                var tQN = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_norm.bin", headDim));
                var tKN = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_norm.bin", headDim));
                var tGate = Task.Run(() => ReadFP16Packed(lp + "/mlp_gate_proj.bin", mlpPart));
                var tUp = Task.Run(() => ReadFP16Packed(lp + "/mlp_up_proj.bin", mlpPart));
                var tDown = Task.Run(() => ReadFP16Packed(lp + "/mlp_down_proj.bin", mlpPart));
                var tILn = Task.Run(() => ReadFP16Packed(lp + "/input_layernorm.bin", hiddenSize));
                var tPALn = Task.Run(() => ReadFP16Packed(lp + "/post_attention_layernorm.bin", hiddenSize));
                var tPFLn = Task.Run(() => ReadFP16Packed(lp + "/pre_feedforward_layernorm.bin", hiddenSize));
                var tPPLn = Task.Run(() => ReadFP16Packed(lp + "/post_feedforward_layernorm.bin", hiddenSize));

                await Task.WhenAll(tQ, tK, tV, tO, tQN, tKN, tGate, tUp, tDown, tILn, tPALn, tPFLn, tPPLn);

                // Assemble flat packed buffers off the main thread.
                int qPacked = qSize / 2;
                int kvPacked = kvSize / 2;
                int mlpPacked = mlpPart / 2;
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
                    Array.Copy(tUp.Result, 0, mlp, mlpPacked, mlpPacked);
                    Array.Copy(tDown.Result, 0, mlp, 2 * mlpPacked, mlpPacked);

                    return (qkv, mlp);
                });

                // Main thread: only SetData.
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
                embedLmHead?.Release();
                finalNormGamma?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    W_QKV[i]?.Release(); W_O[i]?.Release(); mlpWeights[i]?.Release();
                    qNormGamma[i]?.Release(); kNormGamma[i]?.Release();
                    inputLnGamma[i]?.Release(); postAttnLnGamma[i]?.Release();
                    preFfnLnGamma[i]?.Release(); postFfnLnGamma[i]?.Release();
                }
            }
        }
    }
}
