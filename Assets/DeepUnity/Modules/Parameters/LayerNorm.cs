using DeepUnity;
using System;
using System.Runtime.Remoting.Messaging;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class LayerNorm: IModule, IParameters
    {
        // For now the sequence length is 1. This module will be developed further for RNNs.
        // Learable parameters will be converted to Tensors to reach each sequence element.
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
       
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        [SerializeField] public int sequenceLength;
        [SerializeField] public float momentum;
        [SerializeField] public float epsilon;
       

        // Learnable parameters
        [SerializeField] public float runningMean;
        [SerializeField] public float runningVar;
                            
        [SerializeField] public float gamma;
        [SerializeField] public float beta;

        [SerializeField] public float grad_Gamma;
        [SerializeField] public float grad_Beta;

        /// <summary>
        /// Layer normalization for NeuralNetworks. Works for any batch-size.
        /// [sequence, batch, features]
        /// </summary>
        public LayerNorm(int seq_len = 1, float momentum = 0.9f, float eps = 1e-5f)
        {
            if (seq_len > 1)
                throw new NotImplementedException("LayerNorm was not implemented yet for sequencial input.");

            this.sequenceLength = seq_len;
            this.epsilon = eps;
            this.momentum = momentum;

            this.gamma = 1;
            this.beta = 0;

            grad_Gamma = 0;
            grad_Beta = 0;
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

            // compute running mu and sig2
            runningMean = runningMean * momentum + Tensor.Mean(mu, TDim.height)[0] * (1f - momentum);
            runningVar = runningVar * momentum + Tensor.Mean(var, TDim.height)[0] * (1f - momentum);

            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = dLdY.Shape.height;
            var dLdxHat = dLdY * gamma;
            var dLdVar = Tensor.Mean(dLdxHat + xCentered * (-1f / 2f) *
                         Tensor.Pow(std + epsilon, -3f / 2f),
                         TDim.width, true);
           
            var dLdMu = Tensor.Mean(dLdxHat * -1f / (std + epsilon) +
                        dLdVar * -2f * xCentered / m,
                        TDim.width, true);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + epsilon) +
                       dLdVar * 2f * xCentered / m + dLdMu * (1f / m);

            var dLdGamma = Tensor.Mean(dLdY + xCentered, TDim.width);
            var dLdBeta = Tensor.Mean(dLdY, TDim.width);

            grad_Gamma += Tensor.Mean(dLdGamma, TDim.height)[0];
            grad_Beta += Tensor.Mean(dLdBeta, TDim.height)[0];
            
            return dLdX;
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

