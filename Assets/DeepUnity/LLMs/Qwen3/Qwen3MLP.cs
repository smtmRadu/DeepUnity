using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3Modeling
    {
        public class Qwen3MLP
        {
            private int hidden_size;
            private int intermediate_size;
            public ComputeBuffer weights;
            public ComputeBuffer inputOutputBuffer;
            public ComputeBuffer intermediateBuffer;
            private ComputeShader cs;
            public bool IsInitialized { get; private set; } = false;

            public Qwen3MLP(int hidden_size, int intermediate_size, string layer_params_path)
            {
                this.cs = DeepUnityMeta.GLUInferenceCS;
                this.hidden_size = hidden_size;
                this.intermediate_size = intermediate_size;

                weights = new ComputeBuffer(hidden_size * intermediate_size * 3, 4);
                if (!string.IsNullOrEmpty(layer_params_path))
                {
                    _ = LoadWeightsAsync(layer_params_path);
                }       
            }

            private async Task LoadWeightsAsync(string path)
            {
                Task<float[]>[] tasks = new Task<float[]>[3];
                tasks[0] = Task.Run(() => Utils.ReadWeights(path + "/mlp_gate_proj.bin", hidden_size * intermediate_size));
                tasks[1] = Task.Run(() => Utils.ReadWeights(path + "/mlp_up_proj.bin", hidden_size * intermediate_size));
                tasks[2] = Task.Run(() => Utils.ReadWeights(path + "/mlp_down_proj.bin", hidden_size * intermediate_size));

                float[][] results = await Task.WhenAll(tasks);
                float[] flat = new float[hidden_size * intermediate_size * 3];
                int partLength = hidden_size * intermediate_size;
                Array.Copy(results[0], 0, flat, 0, partLength);
                Array.Copy(results[1], 0, flat, partLength, partLength);
                Array.Copy(results[2], 0, flat, 2 * partLength, partLength);
                weights.SetData(flat);

                //Debug.Log(results[0].ToCommaSeparatedString());
                //Debug.Log(results[1].ToCommaSeparatedString());
                //Debug.Log(results[2].ToCommaSeparatedString());
                IsInitialized = true;
                // ConsoleMessage.Info($"Loaded {path}/mlp");
            }

            ~Qwen3MLP()
            {
                weights?.Release();
                inputOutputBuffer?.Release();
                intermediateBuffer?.Release();
            }
            private void PrepareIntermediateBuffer(int B, int L)
            {
                if (intermediateBuffer == null || intermediateBuffer.count != B * L * this.intermediate_size)
                {
                    intermediateBuffer?.Release();
                    intermediateBuffer = new ComputeBuffer(B * L * this.intermediate_size, 4, ComputeBufferType.Structured);
                }
            }

            private void PrepareIOBuffer(int B, int L)
            {
                if (inputOutputBuffer == null || intermediateBuffer.count != B * L * this.hidden_size)
                {
                    inputOutputBuffer?.Release();
                    inputOutputBuffer = new ComputeBuffer(B * L * this.hidden_size, 4, ComputeBufferType.Structured);
                }
            }
            public Tensor Predict(Tensor x)
            {
                int seq_len = x.Size(-2);
                bool isBatched = x.Rank == 3;
                int batch_size = isBatched ? x.Size(-3) : 1;

                PrepareIOBuffer(B: batch_size, L: seq_len);
                inputOutputBuffer.SetData(x.ToArray());

                int kGateUp = cs.FindKernel(seq_len == 1 && batch_size == 1 ? "GateUp1Vec" : "GateUp");
                int kDown = cs.FindKernel(seq_len == 1 && batch_size == 1 ? "Down1Vec" : "Down");
                cs.SetBuffer(kGateUp, "weights", weights);
                cs.SetBuffer(kGateUp, "input", inputOutputBuffer);
                cs.SetBuffer(kDown, "weights", weights);
                cs.SetBuffer(kDown, "input", inputOutputBuffer); // input buffer is also used to write output in it.

                PrepareIntermediateBuffer(B: batch_size, L: seq_len);
                cs.SetBuffer(kGateUp, "intermediate", intermediateBuffer);

                cs.SetInt("activation_type", 1); // gelu tanh
                cs.SetInt("hidden_size", this.hidden_size);
                cs.SetInt("intermediate_size", this.intermediate_size);
                cs.SetInt("batch_size", batch_size);
                cs.SetInt("seq_len", seq_len);

                if (seq_len == 1 && batch_size == 1)
                    cs.Dispatch(kGateUp, (this.intermediate_size + 255) / 256, seq_len, batch_size);
                else
                    cs.Dispatch(kGateUp, (this.intermediate_size + 63) / 64, (seq_len + 3) / 4, batch_size);

                // Tensor interBufT = Tensor.Constant(interBuf, batch_size, seq_len, intermediate_size);
                // Debug.Log("Intermediate GemmaMLP: " + interBufT.ToArray().ToCommaSeparatedString());
                cs.SetBuffer(kDown, "intermediate", intermediateBuffer);

                if (seq_len == 1 && batch_size == 1)
                    cs.Dispatch(kDown, (this.intermediate_size + 319) / 320, seq_len, batch_size);
                else
                    cs.Dispatch(kDown, (this.hidden_size + 31) / 32, (seq_len + 3) / 4, batch_size);

                Tensor yT = Tensor.Constant(inputOutputBuffer, x.Shape);
 
                return yT;
            }
        }
    }
}

