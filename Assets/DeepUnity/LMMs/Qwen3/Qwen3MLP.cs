using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3Modeling
    {
        public class Qwen3MLP
        {
            private int hidden_size;
            private int intermediate_size;
            public TensorGPU weights;
            public ComputeBuffer weights_Cb;

            public Qwen3MLP(int hidden_size, int intermediate_size)
            {
                this.hidden_size = hidden_size;
                this.intermediate_size = intermediate_size;

                //weights = TensorGPU.Ones(hidden_size * intermediate_size * 3);
                weights_Cb = new ComputeBuffer(hidden_size * intermediate_size * 3, 4);           
            }
            // ~Qwen3MLP() {
            //     weights_Cb.Release();
            // }
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
                cs.SetBuffer(kGate, "weights", weights_Cb);
                cs.SetBuffer(kGate, "input", xBuff);
                cs.SetBuffer(kDown, "weights", weights_Cb);
                cs.SetBuffer(kDown, "input", xBuff);

                ComputeBuffer interBuf = new ComputeBuffer(batch_size * seq_len * this.intermediate_size, 4, ComputeBufferType.Structured);
                cs.SetBuffer(kDown, "intermediate", interBuf);
                cs.SetBuffer(kGate, "intermediate", interBuf);



                cs.SetInt("activation_type", 0);
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
            public TensorGPU Predict(TensorGPU x)
            {
                int seq_len = x.Size(-2);
                bool isBatched = x.Rank == 3;
                int batch_size = isBatched ? x.Size(-3) : 1;

                ComputeShader cs = DeepUnityMeta.GLUInferenceCS;

                // for now fused kernel is slower.... (keep false)
                if(false && hidden_size == 1024 && intermediate_size == 3072 && batch_size == 1 && seq_len == 1)
                {
                    int kfused = cs.FindKernel("GateUpDown_Fused_1024_3072");
                    cs.SetBuffer(kfused, "weights", weights.data);
                    cs.SetBuffer(kfused, "input", x.data);

                    cs.SetInt("hidden_size", this.hidden_size);
                    cs.SetInt("intermediate_size", this.intermediate_size);
                    cs.SetInt("batch_size", batch_size);
                    cs.SetInt("seq_len", seq_len);

                    cs.Dispatch(kfused, 1, 1, 1);
                }
                else
                {
                    int kGate = cs.FindKernel("GateUp");
                    int kDown = cs.FindKernel("Down");
                    cs.SetBuffer(kGate, "weights", weights.data);
                    cs.SetBuffer(kGate, "input", x.data);
                    cs.SetBuffer(kDown, "weights", weights.data);
                    cs.SetBuffer(kDown, "input", x.data);

                    ComputeBuffer 
                        interBuf = new ComputeBuffer(batch_size * seq_len * this.intermediate_size, 4, ComputeBufferType.Structured);
                        cs.SetBuffer(kDown, "intermediate", interBuf);
                        cs.SetBuffer(kGate, "intermediate", interBuf);



                    cs.SetInt("activation_type", 0);
                    cs.SetInt("hidden_size", this.hidden_size);
                    cs.SetInt("intermediate_size", this.intermediate_size);
                    cs.SetInt("batch_size", batch_size);
                    cs.SetInt("seq_len", seq_len);

                    cs.Dispatch(kGate, (this.intermediate_size + 63) / 64, (seq_len + 3) / 4, batch_size);
                    cs.Dispatch(kDown, (this.hidden_size + 63) / 64, (seq_len + 3) / 4, batch_size);
                    interBuf.Release();
                }

                

                return x;

            }
        }
    }
}

