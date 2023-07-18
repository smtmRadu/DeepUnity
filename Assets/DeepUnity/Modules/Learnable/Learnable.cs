using UnityEngine;

namespace DeepUnity
{
    public abstract class Learnable : ISerializationCallbackReceiver
    {
        [SerializeField] public Device device;

        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;

        [SerializeField] public Tensor gammaGrad;
        [SerializeField] public Tensor betaGrad;

       
        public Learnable(Device device) => this.device = device;

        public void ZeroGrad()
        {
            gammaGrad = Tensor.Zeros(gammaGrad.Shape);
            betaGrad = Tensor.Zeros(betaGrad.Shape);
        }
        public void ClipGradValue(float clip_value)
        {
            Tensor.Clip(gammaGrad, -clip_value, clip_value);
            Tensor.Clip(betaGrad, -clip_value, clip_value);
        }
        public void ClipGradNorm(float max_norm)
        {
            Tensor normG = Tensor.Norm(gammaGrad, NormType.ManhattanL1);

            if (normG[0] > max_norm)
            {
                float scale = max_norm / normG[0];
                gammaGrad *= scale;
            }


            Tensor normB = Tensor.Norm(betaGrad, NormType.ManhattanL1);

            if (normB[0] > max_norm)
            {
                float scale = max_norm / normB[0];
                betaGrad *= scale;
            }
        }

        public void OnBeforeSerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            try
            {
                // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.
                var x = gamma.Shape;
                if (x == null || x.Length == 0)
                    throw new System.Exception("Is not even important...");
            }
            catch
            {
                return;
            }

            this.gammaGrad = Tensor.Zeros(gamma.Shape);
            this.betaGrad = Tensor.Zeros(beta.Shape);
        }
    }
}

