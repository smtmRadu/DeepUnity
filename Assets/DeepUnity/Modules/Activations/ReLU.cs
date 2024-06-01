using System;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// <b>Applies the Rectified Linear Unit activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public sealed class ReLU : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Applies the Rectified Linear Unit activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public ReLU(bool in_place = false)
        {
            this.inPlace = in_place;
        }


       
        public Tensor Predict(Tensor x)
        {
            if(inPlace)
            {
                for (int i = 0; i < x.Count(); i++)
                {
                    x[i] = Math.Max(0, x[i]);
                }
                return x;
            }
            else
            return x.Select(k => Math.Max(0f, k));
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k => k > 0f ? 1f : 0f);
        }

        public object Clone() => new ReLU(inPlace);
    }
}