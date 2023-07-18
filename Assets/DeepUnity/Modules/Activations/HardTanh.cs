using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class HardTanh : ActivationBase
    {
        [SerializeField] private float min_value = -1f;
        [SerializeField] private float max_value =  1f;
        protected override Tensor Activation(Tensor x) => x.Select(k =>
        {
            if (k > max_value)
                return max_value;

            if(k < min_value)
                return min_value;

            return k;
        });
        protected override Tensor Derivative(Tensor x) => x.Select(k =>
        {
            if (k > max_value)
                return 0f;

            if (k < min_value)
                return 0f;

            return 1f;
        });
        public HardTanh(float min_val = -1f, float max_val = 1f)
        {
            this.min_value = min_val;
            this.max_value = max_val;
        }
    }
}

