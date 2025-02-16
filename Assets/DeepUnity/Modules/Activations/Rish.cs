using DeepUnity.Modules;
using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// An activation I found out a while ago, that manages to converge way faster, even faster than swish.
    /// </summary>
    [Serializable]
    public sealed class Rish : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private Tensor InputCache { get; set; }

        public Rish(bool in_place = false)
        {
            this.inPlace = in_place;
        }


        public Tensor Predict(Tensor x)
        {
            if (inPlace)
            {
                Parallel.For(0, x.Count(), i =>
                {
                    var exp_x = MathF.Exp(x[i]);
                    x[i] = (x[i] - 1f) * exp_x / (1f + exp_x);
                });
                return x;
            }

            Tensor output = Tensor.Zeros(x.Shape);
            Parallel.For(0, x.Count(), i =>
            {
                var exp_x = MathF.Exp(x[i]);
                output[i] = (x[i] - 1f) * exp_x / (1f + exp_x);
            });
            return output;
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }


        public Tensor Backward(Tensor dLdY)
        {
            Tensor inputGrad = Tensor.Zeros(dLdY.Shape);
            Parallel.For(0, InputCache.Count(), i =>
            {
                float exp_x = MathF.Exp(InputCache[i]);
                float drish = exp_x * (InputCache[i] + exp_x) / MathF.Pow(exp_x + 1f, 2f);
                inputGrad[i] = dLdY[i] * drish;
            });

            return inputGrad;
        }

        public object Clone() => new Rish(inPlace);

    }

}


