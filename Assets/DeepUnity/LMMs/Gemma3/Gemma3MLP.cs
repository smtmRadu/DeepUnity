using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class Gemma3MLP
        {
            private int hidden_size;
            private int intermediate_size;
            public ComputeBuffer weights;
            public bool IsInitialized { get; private set; } = false;
            public Gemma3MLP(int hidden_size, int intermediate_size, string layer_params_path)
            {
                this.hidden_size = hidden_size;
                this.intermediate_size = intermediate_size;

                //weights = TensorGPU.Ones(hidden_size * intermediate_size * 3);
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

                IsInitialized = true;
                // ConsoleMessage.Info($"Loaded {path}/mlp");
            }
            ~Gemma3MLP() 
            {
                 weights.Release();
            }
            public Tensor Predict(Tensor x)
            {
                int seq_len = x.Size(-2);
                bool isBatched = x.Rank == 3;
                int batch_size = isBatched ? x.Size(-3) : 1;

                ComputeShader cs = DeepUnityMeta.GLUInferenceCS;

                ComputeBuffer xBuff = new ComputeBuffer(batch_size * seq_len * x.Size(-1), 4, ComputeBufferType.Structured);
                xBuff.SetData(x.ToArray());

                int kGate = cs.FindKernel("GateUp");
                int kDown = cs.FindKernel("Down");
                cs.SetBuffer(kGate, "weights", weights);
                cs.SetBuffer(kGate, "input", xBuff);
                cs.SetBuffer(kDown, "weights", weights);
                cs.SetBuffer(kDown, "input", xBuff);

                ComputeBuffer interBuf = new ComputeBuffer(batch_size * seq_len * this.intermediate_size, 4, ComputeBufferType.Structured);
                cs.SetBuffer(kDown, "intermediate", interBuf);
                cs.SetBuffer(kGate, "intermediate", interBuf);



                cs.SetInt("activation_type", 1); // gelu tanh
                cs.SetInt("hidden_size", this.hidden_size);
                cs.SetInt("intermediate_size", this.intermediate_size);
                cs.SetInt("batch_size", batch_size);
                cs.SetInt("seq_len", seq_len);

                cs.Dispatch(kGate, (this.intermediate_size + 63) / 64, (seq_len + 3) / 4, batch_size);
                cs.Dispatch(kDown, (this.hidden_size + 63) / 64, (seq_len + 3) / 4, batch_size);

                Tensor yT = Tensor.Constant(xBuff, x.Shape);
                xBuff.Release();
                interBuf.Release();

                return yT;
            }
            
        }
    }
}

