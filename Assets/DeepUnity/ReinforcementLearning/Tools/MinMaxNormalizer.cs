using DeepUnity;
using System;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// MinMax online normalizer. Works well if the input data does not have extreme large values along.
    /// </summary>
    [Serializable]
    public class MinMaxNormalizer
    {
        [SerializeField] private float MIN_RANGE;
        [SerializeField] private float MAX_RANGE;

        [SerializeField] private Tensor min;
        [SerializeField] private Tensor max;
        public MinMaxNormalizer(int size, float min = -1f, float max = 1f)
        {
            MIN_RANGE = min;
            MAX_RANGE = max;

            this.min = Tensor.Fill(min, size);
            this.min = Tensor.Fill(max, size);
        }
        public void Update(Tensor tuple)
        {
            min = Tensor.Minimum(min, tuple);
            max = Tensor.Maximum(max, tuple);
        }
        public Tensor Normalize(Tensor tuple)
        {
            return (tuple - min) / (max - min) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE;
        }
    }
}

