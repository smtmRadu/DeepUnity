using System;

namespace DeepUnity
{
    [Serializable]
    public class Sigmoid : ActivationBase
    {
        protected override void Activation(ref Tensor x)
        {
            x = x.Select(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid;
            });
        }

        protected override void Derivative(ref Tensor x)
        {
            x = x.Select(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid * (1f - sigmoid);
            });
        }
    }
}
