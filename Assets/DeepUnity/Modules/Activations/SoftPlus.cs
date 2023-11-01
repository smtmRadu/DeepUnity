using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Softplus : Activation
    {
        [SerializeField] private float beta = 1f;
        public Softplus(float beta = 1f) => this.beta = beta;
        protected override Tensor Activate(Tensor x)
        {
            return x.Select(x =>
            {
                float ebx = MathF.Exp(beta * x);
                float log = MathF.Log(1f + ebx);
                return  log / beta;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float embx = MathF.Exp(-beta * x);
                return 1f / (1 + embx);
            });
        }
        public override object Clone() => new Softplus(beta);
    }

}