using DeepUnity;
using System;
using UnityEngine;

namespace kbRadu
{
    [Serializable]
    public class ELU : ActivationBase
    {
        [SerializeField] private float alpha = 1f;

        public ELU(float alpha = 1f) => this.alpha = alpha;
        protected override void Activation(ref Tensor x) => x.ForEach(k => k > 0f ? k : alpha * (MathF.Exp(k) - 1f));
        protected override void Derivative(ref Tensor x) => x.ForEach(k => k > 0f ? 1f : alpha * (MathF.Exp(k) -1f));
    
    }
}

