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

        [SerializeField] private float p = -1;
        [SerializeField] private Tensor gamma;
        [NonSerialized] private Tensor gammaGrad;
        [SerializeField] private Tensor runningRms;
        [SerializeField] private int step;

        private Tensor InputCache { get; set; }
        private Tensor xHat { get; set; }
        private Tensor rms { get; set; }

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
            runningRms = Tensor.Ones(1);
            step = 0;
        }

        public object Clone()
        {
            RMSNorm rmsnorm = new RMSNorm();
            rmsnorm.gamma = (Tensor)gamma.Clone();
            rmsnorm.gammaGrad = (Tensor)gammaGrad.Clone();
            rmsnorm.runningRms = (Tensor)runningRms.Clone();
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
           
            return input / runningRms[0] * gamma[0];
        }
        public Tensor Forward(Tensor input)
        {
            if (input.Rank > 2)
                throw new InputException($"Input ({input.Shape.ToCommaSeparatedString()}) received is invalid for LayerNorm. Make sure is of shape (B, H) or (H).");

            bool isBatched = input.Rank == 2;
            int batch_size = isBatched ? input.Size(0) : 1;
            int feature_size = input.Size(-1);

            InputCache = input.Clone() as Tensor;
            rms = (input.Square().Mean(-1, keepDim: true) + Utils.EPSILON).Sqrt().Expand(-1, feature_size);
            xHat = input / rms;

            Tensor y = gamma[0] * xHat;


            float rms_over_batch = isBatched ? rms.Mean(-2)[0] : rms[0];

            // Update running rms
            int total_samples = batch_size + step;
            float weight_old = step / (float)total_samples;
            float weight_new = batch_size / (float)total_samples;
            runningRms = runningRms * weight_old + rms_over_batch * weight_new;
            step = total_samples;

            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = dLdY.Rank == 2;

            Tensor dLdGamma = dLdY * - InputCache / (rms * gamma[0] * gamma[0]);
            gammaGrad[0] = isBatched ?
                dLdGamma.Mean(0).Mean(0)[0]:
                dLdGamma.Mean(0)[0];

            
            Tensor dLdX = gamma[0] * rms.Reciprocal() * (dLdY - xHat * (dLdY * xHat).Mean(-1, keepDim: true).Expand(-1, dLdY.Size(-1)));
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
