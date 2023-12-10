using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Online standardizer. Works well even if input data contains extremly large values.
    /// </summary>
    [Serializable]
    public class RunningNormalizer : INormalizer
    {
        [SerializeField] private int step;
        [SerializeField] private Tensor mean; 
        [SerializeField] private Tensor m2;

        public RunningNormalizer(int size)
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

            // Var equal 0 it is replaced with 1.
            variance = variance.Select(x =>
            {
                if(x == 0)
                    return 1;
                return x;
            });

            if (tuple.Rank < 2)
                return (tuple - mean) / variance.Sqrt();
            else if (tuple.Rank == 2)
            {
                int batch_size = tuple.Size(0);
                return (tuple - mean.Unsqueeze(0).Expand(0, batch_size)) /
                                       variance.Sqrt().Unsqueeze(0).Expand(0, batch_size);
            }
            else
                throw new ArgumentException("Tuple must have either 0, 1 or 2 dimensions.");
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

