using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class ZScoreNormalizer : INormalizer
    {
        [SerializeField] private int step;
        [SerializeField] private Tensor mean; 
        [SerializeField] private Tensor variance;
        [SerializeField] private bool clip;

        public ZScoreNormalizer(int size, bool clip)
        {
            step = 0;
            mean = Tensor.Zeros(size);
            variance = Tensor.Ones(size);
            this.clip = clip;
        }


        public Tensor Normalize(Tensor tuple, bool update = true)
        {
            if (tuple.Rank > 1)
                throw new Exception($"Batched tuple is not allowed!");

            if (update)
                Update(tuple);

            var normalized = (tuple - mean) / (Tensor.Sqrt(variance) + Utils.EPSILON);
            if (!clip)
                return normalized;
            else
            {
                for (int i = 0; i < tuple.Size(-1); i++)
                {
                    normalized[i] = Math.Clamp(normalized[i], -2f * variance[i], 2f * variance[i]);
                }
                return normalized;
            }
        }
        public void Update(Tensor tuple)
        {
            //float weightOld = (float)((double)step/(step + 1));
            //float weightNew = (float)(1.0/(step + 1));
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

