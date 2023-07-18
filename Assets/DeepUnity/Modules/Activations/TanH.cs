using System;

namespace DeepUnity
{
    [Serializable]
    public class Tanh : ActivationBase
    {
        protected override Tensor Activation(Tensor x)
        {
            return x.Select(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return tanh;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return  1f - tanh * tanh;
            });
        }
    }

}