using DeepUnity;
using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RunningNormalizer
    {
        [SerializeField] private float MIN_RANGE;
        [SerializeField] private float MAX_RANGE;

        [SerializeField] private Tensor min;
        [SerializeField] private Tensor max;
        public RunningNormalizer(int size, float min = -1f, float max = 1f)
        {
            MIN_RANGE = min;
            MAX_RANGE = max;

            this.min = Tensor.Fill(min, size);
            this.min = Tensor.Fill(max, size);
        }
        private void Update(Tensor tuple)
        {
            min = Tensor.Minimum(min, tuple);
            max = Tensor.Maximum(max, tuple);
        }
        public Tensor Normalize(Tensor tuple, bool update = true)
        {
            if (update)
                Update(tuple);

            return (tuple - min) / (max - min) * (MAX_RANGE - MIN_RANGE) + MIN_RANGE;
        }
    }
}

