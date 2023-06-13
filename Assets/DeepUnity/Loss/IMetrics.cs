using System;

namespace DeepUnity
{
    public static class Metrics
    {
        public static float Accuracy(NDArray predictions, NDArray targets)
        {
            NDArray errors = predictions.Zip(targets, (p, t) => MathF.Abs(p - t)); // [w x h]
            NDArray batch_errors = NDArray.Sum(errors, 0); // for each batch, we sum up the errors [1 x h]
            NDArray batch_mean = NDArray.Mean(errors, 1); // we average batch errors [1 x 1]
            float acc = 1.0f - batch_mean[0];
            return acc;
        }
    }
}