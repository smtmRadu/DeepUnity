using System;
using UnityEngine;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    [Serializable]
    public class LeakyReLU : IModule, IActivation
    {
        [SerializeField] private float alpha = 0.009999999776482582f;

        public LeakyReLU(float alpha = 1e-2f) => this.alpha = alpha;
        protected Tensor InputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(k => k >= 0f ? k : alpha * k);
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k => k >= 0f ? 1f : alpha);
        }
        public object Clone() => new LeakyReLU(alpha);
    }
}
