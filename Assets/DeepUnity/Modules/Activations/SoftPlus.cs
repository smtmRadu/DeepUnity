using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class SoftPlus : ActivationBase
    {
        [SerializeField] private float beta = 1f;
        public SoftPlus(float beta = 1f) => this.beta = beta;
        protected override Tensor Activation(Tensor x)
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
    }

}