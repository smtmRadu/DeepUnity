using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3GPUFP16Modeling
    {
        public class Gemma3GPUFP16Weights : IDisposable
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

            public Gemma3GPUFP16Weights(string paramsPath)
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

            static ushort[] ReadFP16(string path, int numWeights)
            {
                byte[] bytes = System.IO.File.ReadAllBytes(path);
                ushort[] w = new ushort[numWeights];
                Buffer.BlockCopy(bytes, 0, w, 0, bytes.Length);
                return w;
            }

            static void UploadHalfs(ComputeBuffer buf, ushort[] data)
            {
                uint[] packed = new uint[data.Length / 2];
                for (int i = 0; i < packed.Length; i++)
                    packed[i] = (uint)data[2 * i] | ((uint)data[2 * i + 1] << 16);
                buf.SetData(packed);
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
                Task<ushort[]> normTask = Task.Run(() => ReadFP16(paramsPath + "/norm.bin", hiddenSize));

                await Task.WhenAll(embedTask, normTask, Task.WhenAll(layerTasks));
                UploadHalfs(finalNormGamma, normTask.Result);
                IsReady = true;
                ConsoleMessage.Info("Gemma3GPU FP16 weights loaded.");
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

                Task<ushort[]>[] tasks = new Task<ushort[]>[14];
                for (int i = 0; i < 14; i++)
                {
                    int size = partSizes[i];
                    string path = $"{paramsPath}/lm_head/part_{i}.bin";
                    tasks[i] = Task.Run(() => ReadFP16(path, size));
                }

                ushort[][] results = await Task.WhenAll(tasks);

                int totalSize = vocabSize * hiddenSize;
                ushort[] combined = new ushort[totalSize];
                int offset = 0;
                for (int i = 0; i < 14; i++)
                {
                    Array.Copy(results[i], 0, combined, offset, results[i].Length);
                    offset += results[i].Length;
                }

                UploadHalfs(embedLmHead, combined);
            }

            async Task LoadLayerAsync(string paramsPath, int layerIdx)
            {
                string lp = $"{paramsPath}/layer_{layerIdx}";

                int qSize = hiddenSize * innerEmbDim;
                int kvSize = hiddenSize * innerEmbDim * headsKV / headsQ;
                int oSize = innerEmbDim * hiddenSize;
                int mlpPart = hiddenSize * intermediateSize;

                var tQ = Task.Run(() => ReadFP16(lp + "/self_attn_q_proj.bin", qSize));
                var tK = Task.Run(() => ReadFP16(lp + "/self_attn_k_proj.bin", kvSize));
                var tV = Task.Run(() => ReadFP16(lp + "/self_attn_v_proj.bin", kvSize));
                var tO = Task.Run(() => ReadFP16(lp + "/self_attn_o_proj.bin", oSize));
                var tQN = Task.Run(() => ReadFP16(lp + "/self_attn_q_norm.bin", headDim));
                var tKN = Task.Run(() => ReadFP16(lp + "/self_attn_k_norm.bin", headDim));
                var tGate = Task.Run(() => ReadFP16(lp + "/mlp_gate_proj.bin", mlpPart));
                var tUp = Task.Run(() => ReadFP16(lp + "/mlp_up_proj.bin", mlpPart));
                var tDown = Task.Run(() => ReadFP16(lp + "/mlp_down_proj.bin", mlpPart));
                var tILn = Task.Run(() => ReadFP16(lp + "/input_layernorm.bin", hiddenSize));
                var tPALn = Task.Run(() => ReadFP16(lp + "/post_attention_layernorm.bin", hiddenSize));
                var tPFLn = Task.Run(() => ReadFP16(lp + "/pre_feedforward_layernorm.bin", hiddenSize));
                var tPPLn = Task.Run(() => ReadFP16(lp + "/post_feedforward_layernorm.bin", hiddenSize));

                await Task.WhenAll(tQ, tK, tV, tO, tQN, tKN, tGate, tUp, tDown, tILn, tPALn, tPFLn, tPPLn);

                ushort[] flatQKV = new ushort[hiddenSize * qkvProjDim];
                Array.Copy(tQ.Result, 0, flatQKV, 0, qSize);
                Array.Copy(tK.Result, 0, flatQKV, qSize, kvSize);
                Array.Copy(tV.Result, 0, flatQKV, qSize + kvSize, kvSize);
                UploadHalfs(W_QKV[layerIdx], flatQKV);

                UploadHalfs(W_O[layerIdx], tO.Result);

                ushort[] flatMLP = new ushort[mlpPart * 3];
                Array.Copy(tGate.Result, 0, flatMLP, 0, mlpPart);
                Array.Copy(tUp.Result, 0, flatMLP, mlpPart, mlpPart);
                Array.Copy(tDown.Result, 0, flatMLP, 2 * mlpPart, mlpPart);
                UploadHalfs(mlpWeights[layerIdx], flatMLP);

                UploadHalfs(qNormGamma[layerIdx], tQN.Result);
                UploadHalfs(kNormGamma[layerIdx], tKN.Result);
                UploadHalfs(inputLnGamma[layerIdx], tILn.Result);
                UploadHalfs(postAttnLnGamma[layerIdx], tPALn.Result);
                UploadHalfs(preFfnLnGamma[layerIdx], tPFLn.Result);
                UploadHalfs(postFfnLnGamma[layerIdx], tPPLn.Result);
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
