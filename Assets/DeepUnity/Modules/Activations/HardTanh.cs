using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class HardTanh : IModule, IActivation
    {
        [SerializeField] private float min_value = -1f;
        [SerializeField] private float max_value =  1f;

        protected Tensor InputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(k =>
            {
                if (k > max_value)
                    return max_value;

                if (k < min_value)
                    return min_value;

                return k;
            });
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
                if (k > max_value)
                    return 0f;

                if (k < min_value)
                    return 0f;

                return 1f;
            });
        }

        public object Clone() => new HardTanh();
    }
}

