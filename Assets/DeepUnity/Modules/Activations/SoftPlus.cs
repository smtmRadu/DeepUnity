using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Softplus : Activation
    {
        [SerializeField] private float beta = 1f;
        [SerializeField] private float psi = 1f;

        /// <summary>
        /// f(x) = log(1 + exp(beta*x)) * psi / beta <br></br>
        /// f'(x) = psi * exp(beta * x) / (1 + exp(beta * x)
        /// </summary>
        /// <param name="beta"></param>
        /// <param name="psi">Psi is a newly entroduced factor that targets softplus activation towards exp() but lineary.</param>
        public Softplus(float beta = 1f, float psi = 1f)
        {
            this.beta = beta;
            this.psi = psi;
        }
        protected override Tensor Activate(Tensor x)
        {
            return x.Select(x =>
            {
                float log = MathF.Log(1f + MathF.Exp(beta * x));
                return  psi * log / beta;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float exp_bx = MathF.Exp(beta * x);
                return psi * exp_bx / (1 + exp_bx);
            });
        }
        public override object Clone() => new Softplus(beta);
    }

}