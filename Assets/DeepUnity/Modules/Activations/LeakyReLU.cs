using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class LeakyReLU : Activation
    {
        [SerializeField] private float alpha = 0.009999999776482582f;
        protected override Tensor Activate(Tensor x) => x.Select(k => k >= 0f ? k : (alpha * k)); 
        protected override Tensor Derivative(Tensor x) => x.Select(k => k >= 0f ? 1f : alpha);
        public LeakyReLU(float alpha = 1e-2f) => this.alpha = alpha;
        public override object Clone() => new LeakyReLU();
    }
}
