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
    /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
    /// </summary>
    [Serializable]
    public class RMSNorm : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;

        [SerializeField] private Tensor gamma;
        [NonSerialized] private Tensor gammaGrad;

        private Tensor InputCache { get; set; }
        private Tensor xHat { get; set; }
        private Tensor rmsNorm { get; set; }

        /// <summary>
        /// <b>Placed before the non-linear activation function. </b>    <br />
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input.<br />
        /// where  B = batch_size and H = in_features.<br />
        /// <b>Applies root mean square normalization over the last dimension (H) of the input.</b> 
        /// </summary>
        public RMSNorm()
        {
            gamma = Tensor.Ones(1);
            gammaGrad = Tensor.Zeros(1);
        }

        public object Clone()
        {
            RMSNorm rmsnorm = new RMSNorm();
            rmsnorm.gamma = (Tensor)gamma.Clone();
            rmsnorm.gammaGrad = (Tensor)gammaGrad.Clone();
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
            InputCache = input.Clone() as Tensor;
            rmsNorm = (input.Square().Mean(-1, keepDim: true) + Utils.EPSILON).Sqrt().Expand(-1, input.Size(-1));
            xHat = input / rmsNorm;

            return gamma[0] * xHat;
        }
        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank == 2;

            Tensor dLdGamma = dLdY * InputCache / rmsNorm;
            gammaGrad[0] += isBatched ?
                dLdGamma.Mean(0).Mean(0)[0]:
                dLdGamma.Mean(0)[0];

            
            Tensor dLdX = dLdY * gamma[0] / rmsNorm;
            return dLdX;
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
            gammaGrad = Tensor.Zeros(gamma.Shape);
        }
    }

}
