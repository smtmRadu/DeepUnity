using System;

namespace DeepUnity
{
    public static class Metrics
    {
        /// <summary>
        /// Returns the accuracy value in range [0, 1].
        /// </summary>
        public static float Accuracy(Tensor predictions, Tensor targets)
        {
            Tensor errors = predictions.Zip(targets, (p, t) => MathF.Abs(p - t)); // [batch x outputs]

            errors = Tensor.Mean(errors, 0);
            errors = Tensor.Sum(errors, 0);

            float acc = 1.0f - errors[0];
            return acc;
        }
    }
}