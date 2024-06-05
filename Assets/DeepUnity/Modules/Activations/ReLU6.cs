using System;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// <b>Applies the ReLU6 activation function piece-wise. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public sealed class ReLU6 : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        private Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Applies the ReLU6 activation function piece-wise. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public ReLU6(bool in_place = false)
        {
            this.inPlace = in_place;
        }



        public Tensor Predict(Tensor x)
        {
            if (inPlace)
            {
                for (int i = 0; i < x.Count(); i++)
                {
                    x[i] = Math.Min(Math.Max(0, x[i]), 6f);
                }
                return x;
            }
            else
                return x.Select(k => Math.Min(Math.Max(0, k), 6f));
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k =>
            {
                if (0 < k && k < 6)
                    return 1;
                return 0;
            });
        }

        public object Clone() => new ReLU6(inPlace);
    }
}