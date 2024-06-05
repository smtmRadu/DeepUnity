using System;
using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Windows;

namespace DeepUnity.Modules
{
    /// <summary>
    /// <b>Placed before the non-linear activation function. </b>    <br />
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
    /// where  B = batch_size and H = in_features.<br />
    /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
    /// </summary>
    [Serializable]
    public class RMSNorm1D : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;

        [SerializeField] private float epsilon = 1e-5f;
        [SerializeField] private Tensor gamma;
        [NonSerialized] private Tensor gammaGrad;

        private Tensor xHat { get; set; }
        private Tensor rmsNorm { get; set; }

        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where  B = batch_size and H = in_features.<br />
        /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
        /// </summary>
        public RMSNorm1D(int num_features, float eps = 1e-5f, bool elementwise_affine = true)
        {
            this.epsilon  = eps;
            if(elementwise_affine)
            {
                gamma = Tensor.Ones(num_features);
                gammaGrad = Tensor.Zeros(num_features);
            }
           
        }
        private RMSNorm1D() { }

        public object Clone()
        {
            RMSNorm1D rmsnorm = new RMSNorm1D();
            rmsnorm.epsilon = this.epsilon;
            rmsnorm.Device = Device;
            rmsnorm.RequiresGrad = RequiresGrad;
            if (gamma != null)
            {
                rmsnorm.gamma = (Tensor)gamma.Clone();
                rmsnorm.gammaGrad = (Tensor)gammaGrad.Clone();
            }        
            return rmsnorm;
        }
        public Parameter[] Parameters()
        {
            if (gammaGrad == null)
                OnAfterDeserialize();

            var g = new Parameter(gamma, gammaGrad);

            return new Parameter[] { g };
        }


        public Tensor Predict(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for LayerNorm. Make sure is of shape (B, H) or (H).");

            // x = gamma * x / norm(x)        
            rmsNorm = (input.Square().Mean(-1, keepDim: true) + epsilon).Sqrt().Expand(-1, input.Size(-1));
            xHat = input / rmsNorm;

            if (gamma == null) // no affine
                return xHat;

            bool isBatched = input.Rank == 2;
            Tensor expanded_gamma = isBatched ? gamma.Unsqueeze(0).Expand(0, input.Size(0)) : gamma;
            return expanded_gamma * xHat;
        }
        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank == 2;
            if (RequiresGrad && gamma != null)
            {
                Tensor dLdGamma = dLdY * xHat / rmsNorm;
                gammaGrad += isBatched ? dLdGamma.Mean(0) : dLdGamma;
            }

            if (gamma == null) // no affine
                return dLdY / rmsNorm;

            Tensor expanded_gamma = isBatched ? gamma.Unsqueeze(0).Expand(0, dLdY.Size(0)) : gamma;
            return dLdY * expanded_gamma / rmsNorm;
        }

        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (gamma == null)
                return;

            if (gamma.Shape == null)
                return;

            if (gamma.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            gammaGrad = Tensor.Zeros(gamma.Shape);
        }
    }

}
