using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Online standardizer. Works well even if input data contains extremly large values.
    /// </summary>
    [Serializable]
    public class ZScoreNormalizer : INormalizer
    {
        [SerializeField] private int step;
        [SerializeField] private Tensor mean; 
        [SerializeField] private Tensor m2;

        public ZScoreNormalizer(int size)
        {
            step = 0;
            mean = Tensor.Zeros(size);
            m2 = Tensor.Zeros(size);
        }

        public Tensor Normalize(Tensor tuple)
        {
            if(step <= 1)
                return Tensor.Identity(tuple);

            Tensor variance = m2 / (step - 1);

            if (tuple.Rank == 1)
                return (tuple - mean) / (variance.Sqrt() + Utils.EPSILON);
            else if (tuple.Rank == 2)
            {
                int batch_size = tuple.Size(0);
                return (tuple - mean.Unsqueeze(0).Expand(0, batch_size)) /
                                       (variance.Sqrt().Unsqueeze(0).Expand(0, batch_size) + Utils.EPSILON);
            }
            else
                throw new ArgumentException("Tuple must have either 1 or 2 dimensions.");
        }
        public void Update(Tensor tuple)
        {
            if(tuple.Rank == 2)
            {
                var tuples = tuple.Split(0, 1);
                foreach (var item in tuples)
                {
                    Update(item);
                }
                return;
            }
            step++;
            Tensor delta1 = tuple - mean;
            mean += delta1 / step;
            Tensor delta2 = tuple - mean;
            m2 += delta1 * delta2;
        }
    }
}

