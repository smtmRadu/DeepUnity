using System;

namespace DeepUnity
{
    public delegate float IMetrics(Tensor<float> prediction, Tensor<float> label);
    public static class Metrics
    {
        public static float Accuracy(Tensor<float> prediction, Tensor<float> label)
        {
            Tensor<float> errors = prediction.Zip(label, (p, t) => MathF.Abs(p - t));
            float accuracy01 = 1.0f - errors.Sum();
            float clampedAccuracy01 = MathF.Max(0f, accuracy01);
            return clampedAccuracy01;
        }
    }
}


