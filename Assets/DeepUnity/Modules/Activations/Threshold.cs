using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Threshold : Activation
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
        protected override Tensor Activate(Tensor x) => x.Select(k => k > threshold ? threshold : value);
        protected override Tensor Derivative(Tensor x) => x.Select(k => k > threshold ? 1f : 0f);

        public override object Clone() => new Threshold(threshold, value);
    }
}

