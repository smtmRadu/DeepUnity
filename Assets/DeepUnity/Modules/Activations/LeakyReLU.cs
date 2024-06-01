using System;
using UnityEngine;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class LeakyReLU : IModule, IActivation
    {
        [SerializeField] private bool inPlace = false;
        [SerializeField] private float alpha = 0.009999999776482582f;
        private Tensor InputCache { get; set; }


        /// <summary>
        /// <b>Applies the Leaky Rectified Linear Unit activation function using a negative slope of <paramref name="alpha"/>. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape. 
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public LeakyReLU(float alpha = 1e-2f, bool in_place = false)
        {
            this.alpha = alpha;
            this.inPlace = in_place;
        }
        
        public Tensor Predict(Tensor x)
        {
            if (inPlace)
            {
                for (int i = 0; i < x.Count(); i++)
                {
                    x[i] = x[i] >= 0 ? x[i] : alpha * x[i];
                }
                return x;
            }
            else
                return x.Select(k => k >= 0f ? k : alpha * k);
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k => k >= 0f ? 1f : alpha);
        }
        public object Clone() => new LeakyReLU(alpha, inPlace);
    }
}
