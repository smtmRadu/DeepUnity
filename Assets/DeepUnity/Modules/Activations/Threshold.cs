using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Threshold : IModule, IActivation
    {
        [SerializeField] private float threshold;
        [SerializeField] private float value;
        /// <summary>
        /// 
        /// </summary>
        /// <param name="threshold">The value to threshold at.</param>
        /// <param name="value">The value to replace with.</param>
        public Threshold(float threshold, float value)
        {
            this.threshold = threshold;
            this.value = value;
        }

        protected Tensor InputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(k => k > threshold ? threshold : value);
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k => k > threshold ? 1f : 0f);
        }

        public object Clone() => new Threshold(threshold, value);
    }
}

