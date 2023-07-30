using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class RunningStandardizer
    {
        [SerializeField] private ulong step;
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
            if (tuple.Rank > 1)
                throw new Exception($"Batched tuple is not allowed!");

            if (update)
                Update(tuple);

            return (tuple - mean) / (Tensor.Sqrt(variance) + Utils.EPSILON);
        }
        private void Update(Tensor tuple)
        {
            //float weightOld = (float)((double)step/(step + 1ul));
            //float weightNew = (float)(1.0/(step + 1ul));
            //mean = mean * weightOld + tuple * weightNew;
            //variance = variance * weightOld + tuple * weightNew;

            step++;
            Tensor d1 = tuple - mean;
            mean += d1 / step;
            Tensor d2 = tuple - mean;
            variance = (variance * (step - 1) + d1 * d2) / step;
        }
    }
}

