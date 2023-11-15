using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class ELU : Activation
    {
        [SerializeField] private float alpha = 1f;

        public ELU(float alpha = 1f) => this.alpha = alpha;
        protected override Tensor Activate(Tensor x) => x.Select(k => k > 0f ? k : alpha * (MathF.Exp(k) - 1f));
        protected override Tensor Derivative(Tensor x) => x.Select(k => k > 0f ? 1f : alpha * (MathF.Exp(k) -1f));

        public override object Clone() => new ELU(alpha);
    
    }
}

