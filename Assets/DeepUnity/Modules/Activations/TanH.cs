using System;

namespace DeepUnity
{
    [Serializable]
    public class TanH : ActivationBase
    {
        protected override void Activation(ref Tensor x)
        {
            x.ForEach(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return tanh;
            });
        }
        protected override void Derivative(ref Tensor x)
        {
            x.ForEach(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return 1f - tanh * tanh;
            });
        }
    }

}