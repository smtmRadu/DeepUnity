using System;

namespace DeepUnity
{
    [Serializable]
    public class Sigmoid : ActivationBase
    {
        protected override Tensor Activation(Tensor x)
        {
            return x.Select(x => 
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return  sigmoid * (1f - sigmoid);
            });
        }
    }
}
