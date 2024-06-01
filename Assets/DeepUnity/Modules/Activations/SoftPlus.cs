using System;
using UnityEngine;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class Softplus : ILearnable, IActivation
    {
       
        [SerializeField] private float beta = 1f;
        [SerializeField] private Tensor psi = Tensor.Ones(1);
        [SerializeField] private bool scaleTrainable = false;
        [SerializeField] private bool inPlace = false;
        private Tensor psiGrad;
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        [SerializeField] public bool RequiresGrad { get; set; } = true;
        private Tensor InputCache { get; set; }

        /// <summary>
        /// f(x) = log(1 + exp(<paramref name="beta"/>*x)) * <paramref name="scale"/>  / <paramref name="beta"/> <br></br>
        /// f'(x) = <paramref name="scale"/>  * exp(<paramref name="beta"/> * x) / (1 + exp(<paramref name="beta"/> * x))
        /// <br></br>
        /// <br></br>
        /// <em>Note: <paramref name="beta"/> is fixed, but <paramref name="scale"/> can be trainable.</em>
        /// </summary>
        /// <param name="beta"></param>
        /// <param name="scale">Psi is a newly entroduced factor that targets softplus activation towards exp() but lineary.</param>
        public Softplus(float beta = 1f, float scale = 1f, bool scale_trainable = false, bool in_place = false)
        {
            this.beta = beta;
            this.psi = Tensor.Fill(scale, 1);
            this.scaleTrainable = scale_trainable;
            this.inPlace = in_place;

            if (this.scaleTrainable)
                psiGrad = Tensor.Zeros(1);
        }

       

        public Tensor Predict(Tensor x)
        {
            if(inPlace)
            {
                float log;
                for (int i = 0; i < x.Count(); i++)
                {
                    log = MathF.Log(1f + MathF.Exp(beta * x[i]));
                    x[i] = psi[0] * log / beta;
                }
                return x;
            }
            else
                return x.Select(x =>
                {
                    float log = MathF.Log(1f + MathF.Exp(beta * x));
                    return psi[0] * log / beta;
                });
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            if(scaleTrainable)
                psiGrad[0] = dLdY.Average() * MathF.Log(1f + MathF.Exp(beta * InputCache.Average())) / beta;

            return dLdY * InputCache.Select(x =>
            {
                float exp_bx = MathF.Exp(beta * x);
                return psi[0] * exp_bx / (1 + exp_bx);
            });
        }
        public object Clone() => new Softplus(beta, psi[0], scaleTrainable, inPlace);

        public Parameter[] Parameters()
        {
            if (!scaleTrainable)
                return new Parameter[] { };

            return new Parameter[] { new Parameter(psi, psiGrad)};
        }

        public void OnBeforeSAerialize()
        {

        }
        public void OnAfterDeserialize()
        {
            if (!scaleTrainable)
                return;

            if (psiGrad.Shape == null)
                return;

            if (psiGrad.Shape.Length == 0)
                return;

            psiGrad = Tensor.Zeros(1);
        }
    }

}