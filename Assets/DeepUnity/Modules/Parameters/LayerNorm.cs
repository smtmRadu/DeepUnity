using DeepUnity;
using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class LayerNorm: IModule, IParameters
    {

        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
       
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        
        [SerializeField] public float epsilon;

        // Learnable parameters
        [SerializeField] public float runningMean;
        [SerializeField] public float runningVar;
                            
        [SerializeField] public float gamma;
        [SerializeField] public float beta;


        public LayerNorm(int norm_shape, float eps = 1e-5f)
        {
            this.epsilon = eps;

            this.gamma = 1;
            this.beta = 0;
        }
        public Tensor Predict(Tensor input)
        {
            // input [batch, features]

            var input_centered = (input - runningMean) / MathF.Sqrt(runningVar + epsilon);
            var output = gamma * input_centered + beta;

            return output;    
        }

        public Tensor Forward(Tensor input)
        {
            Tensor mu = Tensor.Mean(input, TDim.width, keepDim: true);
            Tensor var = Tensor.Var(input, TDim.width, keepDim: true);

            // standardize
            xCentered = input - mu;
            std = Tensor.Sqrt(var + epsilon);
            xHat = xCentered / std;

            // scale & shift
            Tensor y = gamma * xHat + beta;

            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            var dLdxHat = dLdY * gamma;
            var dLdVar = Tensor.Mean(dLdxHat + xCentered * (-1f / 2f) * Tensor.Pow(std + epsilon, -3f / 2f), TDim.height, true);

            return null;
        }


        public void ZeroGrad()
        {

        }
        public void ClipGradValue(float clip_value)
        {

        }
        public void ClipGradNorm(float norm_value)
        {

        }
        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {

        }

    }
}

