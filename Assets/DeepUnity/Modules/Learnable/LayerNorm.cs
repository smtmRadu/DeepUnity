using kbRadu;
using System;
using System.Drawing.Printing;
using Unity.VisualScripting;
using UnityEngine;

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

        [SerializeField] private float momentum;
        //[SerializeField] private int step;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;


        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Forward input shape [batch, features]       <br />
        /// Predict input shape [features]              <br />
        /// </summary>
        /// <param name="momentum">Small batch size (0.9 - 0.99), Big batch size (0.6 - 0.85). Best momentum value is <b>m</b> where <b>m = batch.size / dataset.size</b></param>
        public LayerNorm(float momentum = 0.9f)
        {
            gamma = Tensor.Ones(1);
            beta = Tensor.Zeros(1);

            gradGamma = Tensor.Zeros(1);
            gradBeta = Tensor.Zeros(1);

            runningMean = Tensor.Zeros(1);
            runningVar = Tensor.Ones(1);

            // step = 0;
            this.momentum = momentum;
        }
        public Tensor Predict(Tensor input)
        {
            var input_centered = (input - runningMean[0]) / MathF.Sqrt(runningVar[0] + Utils.EPSILON);
            var output = gamma[0] * input_centered + beta[0];

            return output;    
        }

        public Tensor Forward(Tensor input)
        {
            bool isBatched = input.Rank == 2;

            int batch_size = isBatched ? input.Size(0) : 1;
            int num_features = isBatched ? input.Size(1) : input.Size(0);


            Tensor mu = Tensor.Mean(input, isBatched ? 1 : 0, keepDim: true); // actually we don t know if input is batched or not, so use deprecated version of mean
            Tensor var = Tensor.Var(input, isBatched ? 1 : 0, keepDim: true);

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
            bool isBatched = dLdY.Rank == 2;
            // dLdY (B, OUT)
            int m = isBatched ? dLdY.Size(-2) : 1;
            var dLdxHat = dLdY * gamma[0];
            var dLdVar = Tensor.Mean(dLdxHat + xCentered * (-1f / 2f) *
                         Tensor.Pow(std + Utils.EPSILON, -3f / 2f),
                         isBatched ? 1 : 0, true);

            var dLdMu = Tensor.Mean(dLdxHat * -1f / (std + Utils.EPSILON) +
                        dLdVar * -2f * xCentered / m,
                        isBatched ? 1 : 0, true);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) +
                       dLdVar * 2f * xCentered / m + dLdMu * (1f / m);

            // mean along the layer
            var dLdGamma = Tensor.Mean(dLdY + xCentered, isBatched ? 1 : 0);
            var dLdBeta = Tensor.Mean(dLdY, isBatched ? 1 : 0);

            // mean along the batch
            float dLdGamma_across_batch = isBatched ? Tensor.Mean(dLdGamma, 0)[0] : dLdGamma[0];
            float dLdBeta_across_batch = isBatched ? Tensor.Mean(dLdBeta, 0)[0] : dLdBeta[0];

            gradGamma += dLdGamma_across_batch;
            gradBeta += dLdBeta_across_batch;

            return dLdX;
        }
    }
}
