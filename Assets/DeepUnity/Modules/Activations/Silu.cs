using DeepUnity.Modules;
using System;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class SiLU : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Applies SiLU activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public SiLU(bool in_place = false)
        {
            this.inPlace = in_place;    
        }


        public Tensor Predict(Tensor x)
        {
            if (inPlace)
            {
                Parallel.For(0, x.Count(), i =>
                {
                    x[i] = x[i] / (MathF.Exp(-x[i]) + 1f);
                });
                return x;
            }

            Tensor output = Tensor.Zeros(x.Shape);
            Parallel.For(0, x.Count(), i =>
            {
                output[i] = x[i] / (MathF.Exp(-x[i]) + 1f);
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
                float dswish =  exp_x * (InputCache[i] + exp_x + 1f) * MathF.Pow(exp_x + 1f, -2f); 
                inputGrad[i] = dLdY[i] * dswish;
            });

            return inputGrad;
        }
        
        public object Clone() => new SiLU(inPlace);

    }

}


