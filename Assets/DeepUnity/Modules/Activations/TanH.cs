using System;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// <b>Applies the Hyperbolic Tangent activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class Tanh : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        /// <summary>
        /// <b>Applies the Hyperbolic Tangent activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public Tanh(bool in_place = false)
        {
            this.inPlace = in_place;
        }

        protected Tensor OutputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            if(inPlace)
            {
                for (int i = 0; i < x.Count(); i++)
                {
                    x[i] = MathF.Tanh(x[i]);
                }
                return x;
            }
            else
                return x.Select(x =>
                {
                    return MathF.Tanh(x);
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
            return dLdY * (1f - OutputCache.Square());
        }

        public object Clone() => new Tanh(inPlace);
    }

}