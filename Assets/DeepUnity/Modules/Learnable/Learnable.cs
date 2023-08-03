using UnityEngine;
using System;
namespace DeepUnity
{
    /// <summary>
    /// Base class for all <see cref="IModule"/> that have learnable parameters.
    /// </summary>
    public abstract class Learnable : ISerializationCallbackReceiver
    {
        [SerializeField] public Device device;

        [SerializeField] public Tensor gamma;
        [SerializeField] public Tensor beta;
        [NonSerialized] public Tensor gammaGrad;
        [NonSerialized] public Tensor betaGrad;

        protected Learnable(Device device) => this.device = device;

        public void ZeroGrad()
        {

            gammaGrad = Tensor.Zeros(gammaGrad.Shape);
            betaGrad = Tensor.Zeros(betaGrad.Shape);

            if (this is RNNCell R)
            {
                R.recurrentGammaGrad = Tensor.Zeros(R.recurrentGammaGrad.Shape);
                R.recurrentBetaGrad = Tensor.Zeros(R.recurrentBetaGrad.Shape);
            }
        }
        public void ClipGradValue(float clip_value)
        {
            Tensor.Clip(gammaGrad, -clip_value, clip_value);
            Tensor.Clip(betaGrad, -clip_value, clip_value);

            if (this is RNNCell R)
            {
                R.recurrentGammaGrad = Tensor.Clip(R.recurrentGammaGrad, -clip_value, clip_value);
                R.recurrentBetaGrad = Tensor.Clip(R.recurrentBetaGrad, -clip_value, clip_value);
            }
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

            if (this is RNNCell R)
            {
                Tensor rnormG = Tensor.Norm(R.recurrentGammaGrad, NormType.ManhattanL1);

                if (rnormG[0] > max_norm)
                {
                    float scale = max_norm / rnormG[0];
                    R.recurrentGammaGrad *= scale;
                }

                Tensor rnormB = Tensor.Norm(R.recurrentBetaGrad, NormType.ManhattanL1);

                if (rnormB[0] > max_norm)
                {
                    float scale = max_norm / rnormB[0];
                    R.recurrentBetaGrad *= scale;
                }
            }
        }
        public virtual int LearnableParametersCount => gamma.Count() + beta.Count();
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

