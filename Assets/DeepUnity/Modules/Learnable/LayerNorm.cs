using System;
using System.Linq;
using UnityEngine;


namespace DeepUnity
{
    /// <summary>
    /// <b>Placed before the non-linear activation function. </b>    <br />
    /// Input: (B, *) or (*) for unbatched input.<br />
    /// Output: (B, *) or (*) for unbatched input.<br />
    /// where  B = batch_size and * = input_shape.<br />
    /// <b>Applies normalization over all dimensions (*) of the input.</b> 
    /// </summary>
    [Serializable]
    public class LayerNorm: Learnable, IModule, IModuleS
    {
        // Epsilon should be 1e-5f as default, but i keep it on default 1e-8f
        // Just a good reference paper to learn from, i made this just by adapting batchnorm layer.
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
        
        // These caches a flattened shape of the input value
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        [SerializeField] private int[] inputShape;
        [SerializeField] private int step;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;


        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: (B, *) or (*) for unbatched input.<br />
        /// Output: (B, *) or (*) for unbatched input.<br />
        /// where  B = batch_size and * = input_shape.<br />
        /// <b>Applies normalization over all dimensions (*) of the input.</b> 
        /// </summary>
        /// <param name="input_shape">Shape of the input (*), excepting the batch (B) dimension.</param>
        public LayerNorm(params int[] input_shape) : base(Device.CPU)
        {
            if (input_shape == null || input_shape.Length == 0)
                throw new ShapeException("Please specify the input_shape when creating a LayerNorm module.");

            gamma = Tensor.Ones(1);
            beta = Tensor.Zeros(1);

            gammaGrad = Tensor.Zeros(1);
            betaGrad = Tensor.Zeros(1);

            runningMean = Tensor.Zeros(1);
            runningVar = Tensor.Ones(1);

            step = 0;
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

            // This applyes a mean over everything, even the batch.. which is actually ok. I had to 
            // do layernorm over each batch element separately but it works ok like this i think.

            Tensor input_flat = Tensor.Reshape(input, input.Count());

            Tensor mu_flat = Tensor.Mean(input_flat, 0);
            Tensor var_flat = Tensor.Var(input_flat, 0);
            mu_flat = Tensor.Expand(mu_flat, 0, input.Count());
            var_flat = Tensor.Expand(var_flat, 0, input.Count());

            xCentered = input_flat - mu_flat;
            std = Tensor.Sqrt(var_flat + Utils.EPSILON);
            xHat = xCentered / std;

            Tensor y = gamma[0] * xHat + beta[0];
            

            // Update running mean and running var
            int total_samples = batch_size + step;
            float weight_old = step / (float)total_samples;
            float weight_new = batch_size / (float)total_samples;
            runningMean = runningMean * weight_old + Tensor.Mean(mu_flat, 0) * weight_new;
            runningVar = runningVar * weight_old + Tensor.Mean(var_flat, 0) * weight_new;
            step = total_samples;


            return y.Reshape(input.Shape);
        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank > inputShape.Rank;
            int m = isBatched ? dLdY.Size(0) : 1;

            Tensor flattened_dLdY = Tensor.Reshape(dLdY, dLdY.Count());

            var dLdxHat = flattened_dLdY * gamma[0];
            var dLdVar = dLdxHat * xCentered * (-1f / 2f) * Tensor.Pow(std + Utils.EPSILON, -3f / 2f);
            var dLdMu = dLdxHat * -1f / (std + Utils.EPSILON) + dLdVar * -2f * xCentered / m;
            var dLdX = dLdxHat * 1f / Tensor.Sqrt(std + Utils.EPSILON) + dLdVar * 2f * xCentered / m + dLdMu * (1f / m);
            var dLdGamma = Tensor.Mean(flattened_dLdY + xCentered, 0);
            var dLdBeta = Tensor.Mean(flattened_dLdY, 0);

            gammaGrad += dLdGamma;
            betaGrad += dLdBeta;

             return dLdX.Reshape(dLdY.Shape);
        }
    }
}
