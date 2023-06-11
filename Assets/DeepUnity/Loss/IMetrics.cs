using System;

namespace DeepUnity
{
    public static class Metrics
    {
        public static float Accuracy(Tensor predictions, Tensor targets)
        {
            Tensor errors = predictions.Zip(targets, (p, t) => MathF.Abs(p - t)); // [w x h]
            Tensor batch_errors = Tensor.Sum(errors, 0); // for each batch, we sum up the errors [1 x h]
            Tensor batch_mean = Tensor.Mean(errors, 1); // we average batch errors [1 x 1]
            float acc = 1.0f - batch_mean[0];
            return acc;
        }
    }
}