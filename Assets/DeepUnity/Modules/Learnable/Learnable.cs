using UnityEngine;

namespace DeepUnity
{
    public abstract class Learnable : ISerializationCallbackReceiver
    {
        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;

        [SerializeField] public Tensor gradGamma;
        [SerializeField] public Tensor gradBeta;

        public void ZeroGrad()
        {
            gradGamma = Tensor.Zeros(gradGamma.Shape);
            gradBeta = Tensor.Zeros(gradBeta.Shape);
        }
        public void ClipGradValue(float clip_value)
        {
            Tensor.Clip(gradGamma, -clip_value, clip_value);
            Tensor.Clip(gradBeta, -clip_value, clip_value);
        }
        public void ClipGradNorm(float max_norm)
        {
            Tensor normG = Tensor.Norm(gradGamma, NormType.ManhattanL1);

            if (normG[0] > max_norm)
            {
                float scale = max_norm / normG[0];
                gradGamma *= scale;
            }


            Tensor normB = Tensor.Norm(gradBeta, NormType.ManhattanL1);

            if (normB[0] > max_norm)
            {
                float scale = max_norm / normB[0];
                gradBeta *= scale;
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
                    throw new System.Exception();
            }
            catch
            {
                return;
            }

            this.gradGamma = Tensor.Zeros(gamma.Shape);
            this.gradBeta = Tensor.Zeros(beta.Shape);
        }
    }
}

