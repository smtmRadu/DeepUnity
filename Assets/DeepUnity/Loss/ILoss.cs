using System;

namespace DeepUnity
{
    public delegate NDArray ILoss(NDArray predictions, NDArray targets);
    public static class Loss
    {
        // Derivatives
        public static NDArray MSE(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => 2 * (p - t));
        public static NDArray MAE(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => p - t > 0 ? 1f : -1f);
        public static NDArray CrossEntropy(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => (t - p) / (p * (p - 1f) + Utils.EPSILON));
        public static NDArray HingeEmbedded(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => 1f - p * t > 0f ? -t : 0f);

    }

    internal static class Error
    {
        // Functions
        private static NDArray MSE(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => (p - t) * (p - t));
        private static NDArray MAE(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => Math.Abs(p - t));
        private static NDArray CrossEntropy(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) =>
        {
            float error = -t * MathF.Log(p);
            return float.IsNaN(error) ? 0f : error;
        });
        private static NDArray HingeEmbedded(NDArray predicts, NDArray targets) => predicts.Zip(targets, (p, t) => MathF.Max(0f, 1f - p * t));
    }
}