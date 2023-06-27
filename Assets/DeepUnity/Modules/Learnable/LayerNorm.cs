using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class LayerNorm: Learnable, IModule
    {
        // Epsilon should be 1e-5f as default, but i keep it on default 1e-8f
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
       
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        [SerializeField] private int input_rank;
        [SerializeField] private int step;
        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;


        /// <summary>
        /// CNN (C, H, W) - (channels, height, width) <br />
        /// RNN (S, F) - (sequence, features) <br />
        /// DNN (F) - (features) <br />
        /// </summary>
        /// <param name="normalized_shape">Input shape from an expected input of size. <br />
        ///  CNN (C, H, W) - (channels, height, width) <br />
        ///  RNN (S, F) - (sequence, features) <br />
        ///  DNN (F) - (features) <br />
        /// </param>
        public LayerNorm(params int[] normalized_shape)
        {
            if (normalized_shape == null || normalized_shape.Length == 0)
                throw new ArgumentException("LayerNorm normalized_shape is not a valid shape.");

            input_rank = normalized_shape.Length;
            step = 0;

            if(input_rank == 1) // BASE DNN
            {
                gamma = Tensor.Ones(1);
                beta = Tensor.Zeros(1);

                gradGamma = Tensor.Zeros(1);
                gradBeta = Tensor.Zeros(1);
            }
            else if(input_rank == 2) // RNN
            {
                gamma = Tensor.Ones(normalized_shape[0], 1);
                beta = Tensor.Zeros(normalized_shape[0], 1);

                gradGamma = Tensor.Zeros(normalized_shape[0], 1);
                gradBeta = Tensor.Zeros(normalized_shape[0], 1);

            }
            else if(input_rank == 3) // CNN
            {
                gamma = Tensor.Ones(normalized_shape[0], 1, 1);
                beta = Tensor.Zeros(normalized_shape[0], 1, 1);

                gradGamma = Tensor.Zeros(normalized_shape[0], 1, 1);
                gradBeta = Tensor.Zeros(normalized_shape[0], 1, 1);
            }
            else if(input_rank == 4)
            {
                throw new Exception("The batch dimension should not be included in normalized_shape.");
            }
        }
        public Tensor Predict(Tensor input)
        {
            var input_centered = (input - runningMean) / Tensor.Sqrt(runningVar + Utils.EPSILON);
            var output = gamma * input_centered + beta;

            return output;    
        }

        public Tensor Forward(Tensor input)
        {

            Tensor mu;
            Tensor var;

            int batch_size = -1;
            if (input.Rank == input_rank + 1)
                batch_size = input.Size(0);
            else if (input.Rank == input_rank)
                batch_size = 1;


            mu = Tensor.Mean(input, TDim.height, keepDim: true);
            var = Tensor.Var(input, TDim.height, keepDim: true);

            if(input_rank == 3) // CNN case
            {
                mu = Tensor.Mean(mu, TDim.width, keepDim: true);
                var = Tensor.Var(var, TDim.width, keepDim: true);
            }
           

            xCentered = input - mu;
            std = Tensor.Sqrt(var + Utils.EPSILON);
            xHat = xCentered / std;

            Tensor y = gamma * xHat + beta;


            int total = step + batch_size;
            runningMean = runningMean * (step / total) + (Tensor.Mean(mu, 0) - runningMean) * (batch_size / total);
            runningVar = runningVar * (step / total) + (Tensor.Mean(var, 0) - runningVar) * (batch_size / total);
            step = total;

            return y;

            // Or updateing with decay
            // runningMean = runningMean * momentum + Tensor.Mean(mu, TDim.height)[0] * (1f - momentum);
            // runningVar = runningVar * momentum + Tensor.Mean(var, TDim.height)[0] * (1f - momentum);
        }
        public Tensor Backward(Tensor dLdY)
        {
            int m = -1;
            if (dLdY.Rank == input_rank + 1)
                m = dLdY.Size(0);
            else if (dLdY.Rank == input_rank)
                m = 1;


            var dLdxHat = dLdY * gamma;
            var dLdVar = Tensor.Mean(dLdxHat + xCentered * (-1f / 2f) *
                          Tensor.Pow(std + Utils.EPSILON, -3f / 2f),
                          TDim.width, true);

            if(input_rank == 3) // CNN CASE (apply mean calculation for both width and height
                dLdVar = Tensor.Mean(dLdVar, TDim.height, true);
            
            
             var dLdMu = Tensor.Mean(dLdxHat * -1f / (std + Utils.EPSILON) +
                             dLdVar * -2f * xCentered / m,
                             TDim.width, true);
            if(input_rank == 3) // CNN CASE
                dLdMu = Tensor.Mean(dLdMu, TDim.height, true);



            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) +
                        dLdVar * 2f * xCentered / m + dLdMu * (1f / m);
            
            var dLdGamma = Tensor.Mean(dLdY + xCentered, TDim.width);
            var dLdBeta = Tensor.Mean(dLdY, TDim.width);
            
            if(input_rank == 3)
            {
                dLdGamma = Tensor.Mean(dLdGamma, TDim.height);
                dLdBeta = Tensor.Mean(dLdBeta, TDim.height);
            }
           
            if(input_rank != 3)
            {
                gradGamma += Tensor.Mean(dLdGamma, TDim.height);
                gradBeta += Tensor.Mean(dLdBeta, TDim.height);
            }
            else
            {
                gradGamma += Tensor.Mean(dLdGamma, TDim.channel);
                gradBeta += Tensor.Mean(dLdBeta, TDim.channel);
            }
           
            
            return dLdX;
        }
    }
}

// Old version

/*// 
        // Learable parameters will be converted to Tensors to reach each sequence element.
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
       
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        
        [SerializeField] public float epsilon;
        [SerializeField] public float momentum;

        // Learnable parameters
        [SerializeField] public float runningMean;
        [SerializeField] public float runningVar;
                            
        [SerializeField] public float gamma;
        [SerializeField] public float beta;

        [SerializeField] public float grad_Gamma;
        [SerializeField] public float grad_Beta;

        /// <summary>
        /// Layer normalization for NeuralNetworks. Sequence length is 1. Works for any batch-size.
        /// </summary>
        public LayerNorm(float momentum = 0.9f, float eps = 1e-5f)
        {
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
 */