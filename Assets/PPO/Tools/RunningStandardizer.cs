using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RunningStandardizer
    {
        [SerializeField] private int step;
        [SerializeField] private Tensor mean; 
        [SerializeField] private Tensor variance;


        public RunningStandardizer(int size)
        {
            step = 0;
            mean = Tensor.Zeros(size);
            variance = Tensor.Ones(size);
        }


        public Tensor Standardise(Tensor tuple, bool update = true)
        {
            if (update)
                Update(tuple);

            return (tuple - mean) / (Tensor.Sqrt(variance) + Utils.EPSILON);
        }
        private void Update(Tensor tuple)
        {
            step++;

            Tensor d1 = tuple - mean;
            mean += d1 / step;
            Tensor d2 = tuple - mean;
            variance = (variance * (step - 1) + d1 * d2) / step;
            //variance_sum += delta * delta2;
        }
    }
}

