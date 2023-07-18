using kbRadu;
using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Windows;

namespace DeepUnity
{
    [Serializable]
    public class LayerNorm: Learnable, IModule
    {
        // Epsilon should be 1e-5f as default, but i keep it on default 1e-8f
        // Just a good reference paper to learn from, i made this just by adapting batchnorm layer.
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
        
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        [SerializeField] private int[] inputShape;
        [SerializeField] private float momentum = 0.1f;
        //[SerializeField] private int step;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;


        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: (Batch, *)      <br />
        /// Output: (Batch, *)     <br />
        /// </summary>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is <b>m</b> where <b>m = batch.size / dataset.size</b></param>
        public LayerNorm(params int[] input_shape) : base(Device.CPU)
        {
            gamma = Tensor.Ones(1);
            beta = Tensor.Zeros(1);

            gammaGrad = Tensor.Zeros(1);
            betaGrad = Tensor.Zeros(1);

            runningMean = Tensor.Zeros(1);
            runningVar = Tensor.Ones(1);

            // step = 0;
            this.inputShape = input_shape.ToArray();
        }
        public Tensor Predict(Tensor input)
        {
            var input_centered = (input - runningMean[0]) / MathF.Sqrt(runningVar[0] + Utils.EPSILON);
            var output = gamma[0] * input_centered + beta[0];

            return output;    
        }

        public Tensor Forward(Tensor input)
        {
            bool isBatched = input.Rank > inputShape.Rank;
            int batch_size = isBatched? input.Size(0) : 1;
            int num_features = input.Size(-1);


            Tensor mu = Tensor.Mean(input, -1, keepDim: true).Expand(-1, input.Size(-1));
            Tensor var = Tensor.Var(input, -1, keepDim: true).Expand(-1, input.Size(-1));

            xCentered = input - mu;
            std = Tensor.Sqrt(var + Utils.EPSILON);
            xHat = xCentered / std;

            Tensor expandedGamma = Tensor.Expand(gamma, 0, num_features).Unsqueeze(0);
            expandedGamma = Tensor.Expand(expandedGamma, 0, batch_size);

            Tensor expandedBeta = Tensor.Expand(beta, 0, num_features).Unsqueeze(0);
            expandedBeta = Tensor.Expand(expandedBeta, 0, batch_size);
            Tensor y = expandedGamma * xHat + expandedBeta;


            float mu_across_batch = isBatched ? Tensor.Mean(mu, 0)[0] : mu[0];
            float var_across_batch = isBatched ? Tensor.Mean(var, 0)[0] : var[0];

            
            // Sharing consistance update approach
            // step += batch_size;
            // float d1 = mu_across_batch - runningMean[0];
            // runningMean += d1 / step;
            // float d2 = mu_across_batch - runningMean[0];
            // runningVar = (runningVar * (step - batch_size) + d1 * d2) / step;
            
            float momentum = 0.9f;
            runningMean = runningMean * momentum + mu_across_batch * (1f - momentum);
            runningVar = runningVar * momentum + var_across_batch * (1f - momentum);

            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank > inputShape.Rank;
            int m = isBatched ? dLdY.Size(0) : 1;

            var dLdxHat = dLdY * gamma[0];
            var dLdVar = Tensor.Mean(dLdxHat + xCentered * (-1f / 2f) *
                         Tensor.Pow(std + Utils.EPSILON, -3f / 2f),
                         -1, true).Expand(-1, dLdY.Size(-1));

            var dLdMu = Tensor.Mean(dLdxHat * -1f / (std + Utils.EPSILON) +
                        dLdVar * -2f * xCentered / m,
                        -1, true).Expand(-1, dLdY.Size(-1));

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) +
                       dLdVar * 2f * xCentered / m + dLdMu * (1f / m);

            // mean along the layer
            var dLdGamma = Tensor.Mean(dLdY + xCentered, -1);
            var dLdBeta = Tensor.Mean(dLdY, -1);

            // mean along the batch
            float dLdGamma_across_batch = isBatched ? Tensor.Mean(dLdGamma, 0)[0] : dLdGamma[0];
            float dLdBeta_across_batch = isBatched ? Tensor.Mean(dLdBeta, 0)[0] : dLdBeta[0];

            gammaGrad += dLdGamma_across_batch;
            betaGrad += dLdBeta_across_batch;

            return dLdX;
        }
    }
}
