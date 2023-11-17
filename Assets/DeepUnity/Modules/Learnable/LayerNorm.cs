using System;
using Unity.VisualScripting;
using UnityEngine;


namespace DeepUnity
{
    /// <summary>
    /// <b>Placed before the non-linear activation function. </b>    <br />
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// where  B = batch_size and H = in_features.<br />
    /// <b>Applies normalization over the last dimension (H) of the input.</b> 
    /// </summary>
    [Serializable]
    public class LayerNorm: ILearnable, IModule, IModule2
    {
        // Epsilon should be 1e-5f as default, but i keep it on default 1e-8f
        // Just a good reference paper to learn from, i made this just by adapting batchnorm layer.
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
        
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        [SerializeField] private int step;

        // Learnable parameters
        [SerializeField] private Tensor runningMean;
        [SerializeField] private Tensor runningVar;

        [SerializeField] private Tensor gamma;
        [SerializeField] private Tensor beta;
        [NonSerialized] private Tensor gammaGrad;
        [NonSerialized] private Tensor betaGrad;


        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where  B = batch_size and H = in_features.<br />
        /// <b>Applies normalization over the last dimension (H) of the input.</b> 
        /// </summary>
        public LayerNorm()
        {
            runningMean = Tensor.Zeros(1);
            runningVar = Tensor.Ones(1);

            gamma = Tensor.Ones(1);
            beta = Tensor.Zeros(1);
            gammaGrad = Tensor.Zeros(1);
            betaGrad = Tensor.Zeros(1);
            step = 0;
        }
        public Tensor Predict(Tensor input)
        {
            Tensor input_centered = (input - runningMean[0]) / MathF.Sqrt(runningVar[0] + Utils.EPSILON);
            Tensor output = gamma[0] * input_centered + beta[0];
            return output;    
        }

        public Tensor Forward(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for LayerNorm. Make sure is of shape (B, H) or (H).");

            bool isBatched = input.Rank == 2;
            int batch_size = isBatched? input.Size(0) : 1;
            int feature_size = input.Size(-1);

            Tensor mu = input.Mean(-1, keepDim: true).Expand(-1, feature_size);
            Tensor var = input.Var(-1, keepDim: true).Expand(-1, feature_size);

            xCentered = input - mu;
            std = Tensor.Sqrt(var + Utils.EPSILON);
            xHat = xCentered / std;

            Tensor y = gamma[0] * xHat + beta[0];

            float mu_over_batch = isBatched ? mu.Mean(-2)[0] : mu[0];
            float var_over_batch = isBatched ? var.Mean(-2)[0] : var[0];

            // Update running mean and running var
            int total_samples = batch_size + step;
            float weight_old = step / (float)total_samples;
            float weight_new = batch_size / (float)total_samples;
            runningMean = runningMean * weight_old + mu_over_batch * weight_new;
            runningVar = runningVar * weight_old + var_over_batch * weight_new;
            step = total_samples;

            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            // check page 4 https://arxiv.org/pdf/1502.03167.pdf for differentiation

            bool isBatched = dLdY.Rank == 2;
            int m = isBatched ? dLdY.Size(0) : 1;

            Tensor dLdxHat = dLdY * gamma[0];
            Tensor dLdVar = dLdxHat * xCentered * (-1f / 2f) * Tensor.Pow(std.Pow(2f) + Utils.EPSILON, -3f / 2f);
            Tensor dLdMu = dLdxHat * -1f / std + dLdVar * -2f * xCentered / m;
            Tensor dLdX = dLdxHat * 1f / std + dLdVar * 2f * xCentered / m + dLdMu * (1f / m);
            Tensor dLdGamma = Tensor.Mean(dLdY + xCentered, 0);
            Tensor dLdBeta = Tensor.Mean(dLdY, 0);

            Tensor.CopyTo(gammaGrad + dLdGamma.Mean(0), gammaGrad);
            Tensor.CopyTo(betaGrad + dLdBeta.Mean(0), betaGrad);

            return dLdX;
        }

        public object Clone()
        {
            LayerNorm laynorm = new LayerNorm();
            laynorm.step = this.step;
            laynorm.gamma = (Tensor)this.gamma.Clone();
            laynorm.beta = (Tensor)this.beta.Clone();
            laynorm.runningMean = (Tensor)this.runningMean.Clone(); 
            laynorm.runningVar = (Tensor)this.runningVar.Clone();
            return laynorm;
        }


        public void SetDevice(Device device) { return; }
        public int ParametersCount()
        {
            return gamma.Count() + beta.Count();
        }
        public Parameter[] Parameters()
        {
            if (gammaGrad == null)
                OnAfterDeserialize();

            var g = new Parameter(gamma, gammaGrad);
            var b = new Parameter(beta, betaGrad);

            return new Parameter[] { g, b };
        }
        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (gamma.Shape == null)
                return;

            if (gamma.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            this.gammaGrad = Tensor.Zeros(gamma.Shape);
            this.betaGrad = Tensor.Zeros(beta.Shape);

        }
    }
}
