using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class ELU : IModule, IActivation
    {
        [SerializeField] private float alpha = 1f;

        protected Tensor InputCache { get; set; }
        public ELU(float alpha = 1f) => this.alpha = alpha;

        public Tensor Predict(Tensor x)
        {
            return x.Select(k => k > 0f ? k : alpha * (MathF.Exp(k) - 1f));
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k => k > 0f ? 1f : alpha * (MathF.Exp(k) - 1f));
        }

        public object Clone() => new ELU(alpha);
    
    }
}

