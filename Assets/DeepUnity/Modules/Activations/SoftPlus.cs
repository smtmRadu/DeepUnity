using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class SoftPlus : ActivationBase
    {
        [SerializeField] private float beta = 1f;
        public SoftPlus(float beta = 1f) => this.beta = beta;
        protected override void Activation(ref Tensor x)
        {
            x.ForEach(x =>
            {
                float ebx = MathF.Exp(beta * x);
                float log = MathF.Log(1f + ebx);
                return log / beta;
            });
        }
        protected override void Derivative(ref Tensor x)
        {
            x.ForEach(x =>
            {
                float embx = MathF.Exp(-beta * x);
                return 1f / (1 + embx);
            });
        }
    }

}