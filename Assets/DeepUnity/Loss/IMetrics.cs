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
            Tensor batch_errors = Tensor.Sum(errors, Dim.width); // for each batch, we sum up the errors [batch]
            Tensor batch_mean = Tensor.Mean(errors, Dim.height); // we average batch errors [1]
            float acc = 1.0f - batch_mean[0];
            return acc;
        }
    }
}