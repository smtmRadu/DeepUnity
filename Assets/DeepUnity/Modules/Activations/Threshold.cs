using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Threshold : ActivationBase
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
        protected override void Activation(ref Tensor x) => x.ForEach(k => k > threshold ? k : value);

        protected override void Derivative(ref Tensor x) => x.ForEach(k => k > threshold ? 1f : 0f);
    }
}

