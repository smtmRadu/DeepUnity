using System;

namespace DeepUnity
{
    public static class Metrics
    {
        public static float Accuracy(Tensor predictions, Tensor targets)
        {
            Tensor errors = predictions.Zip(targets, (p, t) => MathF.Abs(p - t)); // [batch x outputs]
            Tensor batch_errors = Tensor.Sum(errors, 1); // for each batch, we sum up the errors [batch]
            Tensor batch_mean = Tensor.Mean(errors, 0); // we average batch errors [1]
            float acc = 1.0f - batch_mean[0];
            return acc;
        }
    }
}