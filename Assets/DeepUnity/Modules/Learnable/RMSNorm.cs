/*using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    public class RMSNorm : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;

        [SerializeField] private float p = -1;
        [SerializeField] private Tensor gamma;
        [SerializeField] private Tensor beta;
        [NonSerialized] private Tensor gammaGrad;
        [NonSerialized] private Tensor betaGrad;

        public RMSNorm(float? partial = null)
        {
            if (partial != null)
            {
                if (partial.Value < 0 || partial.Value > 1)
                    throw new System.ArgumentException("If using partial RMS norm, partial must be in range [0, 1]");
                p = partial.Value;
            }
            else
                p = partial.Value;

            gamma = Tensor.Ones(1);
            beta = Tensor.Zeros(1);
            gammaGrad = Tensor.Zeros(1);
            betaGrad = Tensor.Zeros(1);
        }

        public object Clone()
        {
            RMSNorm laynorm = new RMSNorm();
            laynorm.gamma = (Tensor)gamma.Clone();
            laynorm.beta = (Tensor)beta.Clone();
            laynorm.gammaGrad = (Tensor)gammaGrad.Clone();
            laynorm.betaGrad = (Tensor)betaGrad.Clone();

            return laynorm;
        }
        public Parameter[] Parameters()
        {
            if (gammaGrad == null)
                OnAfterDeserialize();

            var g = new Parameter(gamma, gammaGrad);
            var b = new Parameter(beta, betaGrad);

            return new Parameter[] { g, b };
        }


        public Tensor Predict(Tensor input)
        {
            Tensor rmsnorm = Tensor.Norm(input).RSqrt();
            return input / rmsnorm;

            throw new ArgumentException();
        }
        public Tensor Forward(Tensor input)
        {
            throw new ArgumentException();
        }
        public Tensor Backward(Tensor input)
        {
            throw new ArgumentException();
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
            betaGrad = Tensor.Zeros(beta.Shape);
        }
    }

}


*/