/*using System;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public class Linear : Learnable, IModule, ISelfOptimizable
    {
        private static int AllocateKernel() => kernel_index++;
        private static int kernel_index = 0;


        private int allocated_kernel;
        private ComputeShader shader;
        private ComputeBuffer weights;
        private ComputeBuffer biases;
        public Linear(int in_features, int out_features, InitType gamma_init = InitType.Glorot_Uniform, InitType beta_init = InitType.Zeros)
           : base(Device.GPU,
                 gamma_init,
                 beta_init,
                 new int[] { out_features, in_features },
                 new int[] { out_features },
                 in_features,
                 out_features)
        {
            if (in_features < 1)
                throw new ArgumentException("In_features cannot be less than 1.");
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

            EditorApplication.playModeStateChanged += FreeGPU;

            shader = DeepUnityMeta.LinearCS;
            allocated_kernel = AllocateKernel();

            weights = new ComputeBuffer(gamma.Count(), 4);
            biases = new ComputeBuffer(beta.Count(), 4);
            shader.SetBuffer(allocated_kernel, "weights", weights);
            shader.SetBuffer(allocated_kernel, "biases", biases);
        }

        public Tensor Predict(Tensor input)
        {
            bool isBatched = input.Rank == 2;
            int batch_size = isBatched ? input.Size(-2) : 1;

            ComputeBuffer i_cb = new ComputeBuffer(input.Count(), 4);
            i_cb.SetData(input.ToArray());
            shader.SetBuffer(kernel_index, "input", i_cb);

            ComputeBuffer o_cb = new ComputeBuffer(batch_size * beta.Size(-1), 4);
            shader.SetBuffer(kernel_index, "output", o_cb);

            shader.Dispatch(allocated_kernel,
                (beta.Size(-1) + 31) / 32,
                (batch_size + 31) / 32,
                1);

            Tensor output = Tensor.Constant(o_cb).Reshape(batch_size, beta.Size(-1));

            i_cb.Release();
            o_cb.Release();
            return output;       
        }

        public void SelfOptimise(float lr)
        {

        }
        public Tensor Forward(Tensor input)
        {
            throw new NotImplementedException();
        }
        public Tensor Backward(Tensor loss)
        {
            throw new NotImplementedException();
        }


        private void FreeGPU(PlayModeStateChange state)
        {
            if(state == PlayModeStateChange.ExitingPlayMode)
            {
                weights.Release();
                biases.Release();
            }
           
        }

    }

}


*/