using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3GPUFP32Modeling
    {
        public class Gemma3GPUFP32Weights : IDisposable
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

            public Gemma3GPUFP32Weights(string paramsPath)
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

                embedLmHead = new ComputeBuffer(vocabSize * hiddenSize, 4, ComputeBufferType.Structured);

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
                    W_QKV[i] = new ComputeBuffer(hiddenSize * qkvProjDim, 4, ComputeBufferType.Structured);
                    W_O[i] = new ComputeBuffer(innerEmbDim * hiddenSize, 4, ComputeBufferType.Structured);
                    mlpWeights[i] = new ComputeBuffer(hiddenSize * intermediateSize * 3, 4, ComputeBufferType.Structured);
                    qNormGamma[i] = new ComputeBuffer(headDim, 4, ComputeBufferType.Structured);
                    kNormGamma[i] = new ComputeBuffer(headDim, 4, ComputeBufferType.Structured);
                    inputLnGamma[i] = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                    postAttnLnGamma[i] = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                    preFfnLnGamma[i] = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                    postFfnLnGamma[i] = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                }

                finalNormGamma = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);

                _ = LoadAllAsync(paramsPath);
            }

            private async Task LoadAllAsync(string paramsPath)
            {
                Task embedTask = LoadEmbeddingAsync(paramsPath);

                Task[] layerTasks = new Task[numLayers];
                for (int i = 0; i < numLayers; i++)
                {
                    int idx = i;
                    layerTasks[i] = LoadLayerAsync(paramsPath, idx);
                }

                Task<float[]> normTask = Task.Run(() => Utils.ReadWeights(paramsPath + "/norm.bin", hiddenSize));

                await Task.WhenAll(embedTask, normTask, Task.WhenAll(layerTasks));
                finalNormGamma.SetData(normTask.Result);
                IsReady = true;
                ConsoleMessage.Info("Gemma3GPU weights loaded.");
            }

            private async Task LoadEmbeddingAsync(string paramsPath)
            {
                int[] partSizes = new int[]
                {
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_722
                };

                Task<float[]>[] tasks = new Task<float[]>[14];
                for (int i = 0; i < 14; i++)
                {
                    int size = partSizes[i];
                    string path = $"{paramsPath}/lm_head/part_{i}.bin";
                    tasks[i] = Task.Run(() => Utils.ReadWeights(path, size));
                }

                float[][] results = await Task.WhenAll(tasks);

                int totalSize = vocabSize * hiddenSize;
                float[] combined = new float[totalSize];
                int offset = 0;
                for (int i = 0; i < 14; i++)
                {
                    Array.Copy(results[i], 0, combined, offset, results[i].Length);
                    offset += results[i].Length;
                }

                embedLmHead.SetData(combined);
            }

            private async Task LoadLayerAsync(string paramsPath, int layerIdx)
            {
                string lp = $"{paramsPath}/layer_{layerIdx}";

                int qSize = hiddenSize * innerEmbDim;
                int kvSize = hiddenSize * innerEmbDim * headsKV / headsQ;
                int oSize = innerEmbDim * hiddenSize;
                int mlpPart = hiddenSize * intermediateSize;

                var tQ = Task.Run(() => Utils.ReadWeights(lp + "/self_attn_q_proj.bin", qSize));
                var tK = Task.Run(() => Utils.ReadWeights(lp + "/self_attn_k_proj.bin", kvSize));
                var tV = Task.Run(() => Utils.ReadWeights(lp + "/self_attn_v_proj.bin", kvSize));
                var tO = Task.Run(() => Utils.ReadWeights(lp + "/self_attn_o_proj.bin", oSize));
                var tQN = Task.Run(() => Utils.ReadWeights(lp + "/self_attn_q_norm.bin", headDim));
                var tKN = Task.Run(() => Utils.ReadWeights(lp + "/self_attn_k_norm.bin", headDim));
                var tGate = Task.Run(() => Utils.ReadWeights(lp + "/mlp_gate_proj.bin", mlpPart));
                var tUp = Task.Run(() => Utils.ReadWeights(lp + "/mlp_up_proj.bin", mlpPart));
                var tDown = Task.Run(() => Utils.ReadWeights(lp + "/mlp_down_proj.bin", mlpPart));
                var tILn = Task.Run(() => Utils.ReadWeights(lp + "/input_layernorm.bin", hiddenSize));
                var tPALn = Task.Run(() => Utils.ReadWeights(lp + "/post_attention_layernorm.bin", hiddenSize));
                var tPFLn = Task.Run(() => Utils.ReadWeights(lp + "/pre_feedforward_layernorm.bin", hiddenSize));
                var tPPLn = Task.Run(() => Utils.ReadWeights(lp + "/post_feedforward_layernorm.bin", hiddenSize));

                await Task.WhenAll(tQ, tK, tV, tO, tQN, tKN, tGate, tUp, tDown, tILn, tPALn, tPFLn, tPPLn);

                float[] flatQKV = new float[hiddenSize * qkvProjDim];
                Array.Copy(tQ.Result, 0, flatQKV, 0, qSize);
                Array.Copy(tK.Result, 0, flatQKV, qSize, kvSize);
                Array.Copy(tV.Result, 0, flatQKV, qSize + kvSize, kvSize);
                W_QKV[layerIdx].SetData(flatQKV);

                W_O[layerIdx].SetData(tO.Result);

                float[] flatMLP = new float[mlpPart * 3];
                Array.Copy(tGate.Result, 0, flatMLP, 0, mlpPart);
                Array.Copy(tUp.Result, 0, flatMLP, mlpPart, mlpPart);
                Array.Copy(tDown.Result, 0, flatMLP, 2 * mlpPart, mlpPart);
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
                    W_QKV[i]?.Release();
                    W_O[i]?.Release();
                    mlpWeights[i]?.Release();
                    qNormGamma[i]?.Release();
                    kNormGamma[i]?.Release();
                    inputLnGamma[i]?.Release();
                    postAttnLnGamma[i]?.Release();
                    preFfnLnGamma[i]?.Release();
                    postFfnLnGamma[i]?.Release();
                }
            }
        }
    }
}
