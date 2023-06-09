using System;

namespace DeepUnity
{
    public static class Metrics
    {
        public static float Accuracy(Tensor prediction, Tensor target)
        {
            Tensor errors = prediction.Zip(target, (p, t) => MathF.Abs(p - t));
            float accuracy01 = 1.0f - Tensor.Sum(errors, 0)[0];
            float clampedAccuracy01 = MathF.Max(0f, accuracy01);
            return clampedAccuracy01;
        }
    }
}