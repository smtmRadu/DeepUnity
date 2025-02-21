using System;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    
    /// <summary>
    /// Inputs are normalized using the following formula:
    /// x = (x - mu) / sqrt(var + eps).
    /// </summary>
    [Serializable]
    public class RunningNormalizer
    {
        [SerializeField] private int step;
        [SerializeField] private float eps = 1e-8f;
        [SerializeField] private Tensor mean;
        [SerializeField] private Tensor m2;

        public int Step => step;
        public float Epsilon => eps;
        public Tensor Mean => mean;
        public Tensor M2 => m2;
        /// <summary>
        /// m2 / (step-1) + eps
        /// </summary>
        /// <param name="size"></param>
        /// <param name="eps"></param>
        public Tensor VarianceEps => m2 / (step - 1) + eps;
        /// <summary>
        /// sqrt(m2 / (step-1) + eps)
        /// </summary>
        /// <param name="size"></param>
        /// <param name="eps"></param>
        public Tensor StdEps => Tensor.Sqrt(m2 /  (step - 1) + eps);
        public RunningNormalizer(int size, float eps = 1e-8f)
        {
            this.step = 0;
            this.eps = eps;
            this.mean = Tensor.Zeros(size);
            this.m2 = Tensor.Zeros(size);
        }

        public Tensor Normalize(Tensor x)
        {
            if (step == 0)
                throw new ApplicationException("When using Running Normalizer, `Update` first then `Normalize`.");

            if (step <= 1)
                return Tensor.Identity(x);
            // rather than returning the identity of the tuple (that might have unstably large numbers, 
            // better return a stable form of ones. Why not zeros?
            // (running zeros through the network will end up in zeroes, with the except of some activation functions).

            return Tensor.FusedRunningNormalize(x, mean, m2, eps, step); // maybe faster on large num_envs
            // Tensor std_e = Tensor.Sqrt(m2 / (step - 1) + eps);
            // 
            // if (x.Rank < 2)
            //     return (x - mean) / std_e;
            // else if (x.Rank == 2)
            // {
            //     int batch_size = x.Size(0);
            //     return (x - mean.Unsqueeze(0).Expand(0, batch_size)) /
            //                            std_e.Unsqueeze(0).Expand(0, batch_size);
            // }
            // else
            //     throw new ArgumentException("Tuple must have either 0, 1 or 2 dimensions.");
        }
        public void Update(Tensor x)
        {
            if(x.Rank < 2)
            {
                step++;
                Tensor delta1 = x - mean;
                mean += delta1 / step;
                Tensor delta2 = x - mean;
                m2 += delta1 * delta2;
            }
            else if (x.Rank == 2)
            {
                int batch_size = x.Size(0);
                Tensor batch_mean = x.Mean(0); // (H)
                Tensor batch_variance = x.Var(0, correction: 0);

                Tensor delta1 = batch_mean - mean;
                mean = (mean * step + batch_mean * batch_size) / (step + batch_size);
                Tensor delta2 = batch_variance - mean;

                m2 += batch_variance * batch_size
                    + delta1.Pow(2f) * step * batch_size / (step + batch_size);

                step += batch_size;
            }
            else
            {
                throw new ShapeException($"Cannot Normalize batch of Rank higher than 2. (received {x.Rank})");
            }
            
        }
    }
}

