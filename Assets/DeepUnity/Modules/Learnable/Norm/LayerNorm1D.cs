using System;
using Unity.VisualScripting;
using UnityEngine;


namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Placed before the non-linear activation function. </b>    <br />
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// where  B = batch_size and H = in_features.<br />
    /// <b>Applies normalization over the last dimension (H) of the input.</b> 
    /// </summary>
    [Serializable]
    public class LayerNorm1D : ILearnable, IModule
    {
        // Epsilon should be 1e-5f as default, but i keep it on default 1e-8f
        // Just a good reference paper to learn from, i made this just by adapting batchnorm layer.
        /// https://proceedings.neurips.cc/paper_files/paper/2019/file/2f4fe03d77724a7217006e5d16728874-Paper.pdf
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;
        private Tensor xCentered { get; set; }
        private Tensor xHat { get; set; }
        private Tensor std { get; set; }

        [SerializeField] private bool affine = true;
        [SerializeField] private bool bias = true;
        [SerializeField] private float epsilon = 1e-5f;
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
        /// <param name="affine">Train gamma and beta parameters (elementwise-affine).</param>
        /// <param name="bias">If using affine parameters, do you want to train the shift?</param>
        public LayerNorm1D(int num_features, float eps = 1e-5f, bool affine = true, bool bias = true)
        {
            this.affine = affine;
            this.bias = bias;
            this.epsilon = eps;
            if (affine)
            {
                // gamma & beta are affine learnable params
                gamma = Tensor.Ones(num_features);
                gammaGrad = Tensor.Zeros(num_features);

                if(bias)
                {
                    beta = Tensor.Zeros(num_features);
                    betaGrad = Tensor.Zeros(num_features);
                }
               
            }         
        }
        private LayerNorm1D() { }
        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for LayerNorm. Make sure is of shape (B, H) or (H).");

            int feature_size = input.Size(-1);
            bool isBatched = input.Rank == 2;

            Tensor mu = input.Mean(-1, keepDim: true).Expand(-1, feature_size);
            
            std = input.Std(-1, correction: 0, keepDim: true).Expand(-1, feature_size);
            xCentered = input - mu;
            xHat = xCentered / (std + epsilon);

            if(!affine) 
                return xHat;

            Tensor expanded_gamma = isBatched ? gamma.Unsqueeze(0).Expand(0, input.Size(0)) : gamma;


            if(!bias)
                return expanded_gamma * xHat;


            Tensor expanded_beta = isBatched ? beta.Unsqueeze(0).Expand(0, input.Size(0)) : beta;

            return expanded_gamma * xHat + expanded_beta;     
        }

        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdY)
        {
            // check page 4 https://arxiv.org/pdf/1502.03167.pdf for differentiation

            bool isBatched = dLdY.Rank == 2;
            int m = isBatched ? dLdY.Size(0) : 1;
            Tensor expanded_gamma = isBatched ? gamma.Unsqueeze(0).Expand(0, m) : gamma;

            Tensor dLdxHat = affine ? dLdY : dLdY * expanded_gamma;
            Tensor dLdVar = dLdxHat * xCentered * (-1f / 2f) * Tensor.Pow(std.Square() + epsilon, -3f / 2f);
            Tensor dLdMu = dLdxHat * -1f / std + dLdVar * -2f * xCentered / m;
            Tensor dLdX = dLdxHat / std + dLdVar * 2f * xCentered / m + dLdMu * (1f / m);
            
            if(RequiresGrad && affine)
            {
                Tensor dLdGamma = dLdY * xCentered;
                Tensor.CopyTo(gammaGrad + (isBatched ? dLdGamma.Mean(0) : dLdGamma), gammaGrad);

                if(bias)
                {
                    Tensor dLdBeta = dLdY;
                    Tensor.CopyTo(betaGrad + (isBatched ? dLdBeta.Mean(0): dLdBeta), betaGrad);
                }
                
            }

            return dLdX;
        }

        public object Clone()
        {
            LayerNorm1D laynorm = new LayerNorm1D();
            laynorm.epsilon = epsilon;
            laynorm.affine = affine;
            laynorm.bias = bias;
            laynorm.Device = Device;
            laynorm.RequiresGrad = RequiresGrad;
            if(affine)
            {
                laynorm.gamma = (Tensor)gamma.Clone();
                laynorm.gammaGrad = (Tensor)gammaGrad.Clone();
            }
            if(bias)
            {
                laynorm.beta = (Tensor)beta.Clone();
                laynorm.betaGrad = (Tensor)betaGrad.Clone();
            }
            return laynorm;            
        }
        public Parameter[] Parameters()
        {
          
            if (gammaGrad == null)
                OnAfterDeserialize();

            if (!affine)
                return new Parameter[0];


            var g = new Parameter(gamma, gammaGrad);

            if(!bias)
                return new Parameter[] { g };


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
            if (!affine)
                return;

            if (gamma.Shape == null)
                return;

            if (gamma.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            gammaGrad = Tensor.Zeros(gamma.Shape);

            if (!bias)
                return;

            betaGrad = Tensor.Zeros(beta.Shape);
        }
    }
}
