using DeepUnity.Modules;
using System;
using UnityEngine;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class SiLU : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Applies Swish activation function. </b><br></br>
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
                for (int i = 0; i < x.Count(); i++)
                {
                    x[i] = x[i] / (MathF.Exp(-x[i]) + 1);
                }
                return x;
            }
            else
                return x.Select(k => k / (MathF.Exp(-k) + 1));
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }


        public Tensor Backward(Tensor dLdY)
        {

            return dLdY * InputCache.Select(x =>
            {
                float exp_x = MathF.Exp(x);
                return exp_x * (x + exp_x + 1) / MathF.Pow(exp_x + 1f, 2f);
            });
        }
        
        public object Clone() => new SiLU(inPlace);

    }

}


