using kbRadu;
using System;
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
        /// <b>Always placed before activation. </b>    <br />
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
            int batch_size = input.Height;
            int num_features = input.Width;

            Tensor mu = Tensor.Mean(input, TDim.width, keepDim: true);
            Tensor var = Tensor.Var(input, TDim.width, keepDim: true);

            xCentered = input - mu;
            std = Tensor.Sqrt(var + Utils.EPSILON);
            xHat = xCentered / std;

            Tensor expandedGamma = Tensor.Expand(gamma, TDim.width, num_features);
            expandedGamma = Tensor.Expand(expandedGamma, TDim.height, batch_size);

            Tensor expandedBeta = Tensor.Expand(beta, TDim.width, num_features);
            expandedBeta = Tensor.Expand(expandedBeta, TDim.height, batch_size);
            Tensor y = expandedGamma * xHat + expandedBeta;


            float mu_across_batch = Tensor.Mean(mu, TDim.height)[0];
            float var_across_batch = Tensor.Mean(var, TDim.height)[0];

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
            int m = dLdY.Height;
            var dLdxHat = dLdY * gamma[0];
            var dLdVar = Tensor.Mean(dLdxHat + xCentered * (-1f / 2f) *
                         Tensor.Pow(std + Utils.EPSILON, -3f / 2f),
                         TDim.width, true);

            var dLdMu = Tensor.Mean(dLdxHat * -1f / (std + Utils.EPSILON) +
                        dLdVar * -2f * xCentered / m,
                        TDim.width, true);

            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) +
                       dLdVar * 2f * xCentered / m + dLdMu * (1f / m);

            var dLdGamma = Tensor.Mean(dLdY + xCentered, TDim.width);
            var dLdBeta = Tensor.Mean(dLdY, TDim.width);

            // Also get the mean along the batch (cause the learnable parameters are updated by batch_size steps each call)
            dLdGamma = Tensor.Mean(dLdGamma, TDim.height);
            dLdBeta = Tensor.Mean(dLdBeta, TDim.height);

            dLdGamma = Tensor.Squeeze(dLdGamma);
            dLdBeta = Tensor.Squeeze(dLdBeta);

            gradGamma += dLdGamma;
            gradBeta += dLdBeta;

            return dLdX;
        }
    }
}
