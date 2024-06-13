using System;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// <b>Applies the Logistic Sigmoid activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public sealed class Sigmoid : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private Tensor OutputCache { get; set; }

        /// <summary>
        /// <b>Applies the Logistic Sigmoid activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Sigmoid(bool in_place = false) 
        {
            this.inPlace = in_place;
        }

     
        public Tensor Predict(Tensor x)
        {
            if(inPlace)
            {
                for (int i = 0; i < x.Count(); i++)
                {
                    x[i] = 1f / (1f + MathF.Exp(-x[i]));
                }
                return x;
            }
            else
                return x.Select(x =>
                {
                    float sigmoid = 1f / (1f + MathF.Exp(-x));
                    return sigmoid;
                });
        }

        public Tensor Forward(Tensor x)
        {
            Tensor y = Predict(x);
            OutputCache = y.Clone() as Tensor;
            return y;
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * (1f - OutputCache);
        }

        public object Clone() => new Sigmoid(inPlace);
    }
}
