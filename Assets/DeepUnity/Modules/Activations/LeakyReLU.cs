using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class LeakyReLU : ActivationBase
    {
        [SerializeField] private float alpha = 0.2f;
        protected override Tensor Activation(Tensor x) => x.Select(k => k > 0f ? 1f : k * alpha);
        protected override Tensor Derivative(Tensor x) => x.Select(k => k > 0f ? 1f : alpha);
        public LeakyReLU(float alpha = 0.2f) => this.alpha = alpha;
    }
}
