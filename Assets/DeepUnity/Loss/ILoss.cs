using System;

namespace DeepUnity
{
    public delegate Tensor ILoss(Tensor predictions, Tensor targets);

    /// <summary>
    /// Computes the derivative of the loss function w.r.t output. The value returned can be directly backpropagated.
    /// </summary>
    public static class Loss
    {
        // Derivatives
        public static Tensor MSE(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => 2f * (p - t));
        public static Tensor MAE(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => p - t > 0 ? 1f : -1f);
        public static Tensor CrossEntropy(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => (t - p) / (p * (p - 1f) + Utils.EPSILON));
        public static Tensor HingeEmbedded(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => 1f - p * t > 0f ? -t : 0f);

    }
    /// <summary>
    /// [Non-use] Computes the error of the targets w.r.t output. The value returned cannot be directly backpropagated. Use Loss class instead.
    /// </summary>
    internal static class Error
    {
        // Functions
        private static Tensor MSE(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => (p - t) * (p - t));
        private static Tensor MAE(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => Math.Abs(p - t));
        private static Tensor CrossEntropy(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) =>
        {
            float error = -t * MathF.Log(p);
            return float.IsNaN(error) ? 0f : error;
        });
        private static Tensor HingeEmbedded(Tensor predicts, Tensor targets) => predicts.Zip(targets, (p, t) => MathF.Max(0f, 1f - p * t));
    }
}