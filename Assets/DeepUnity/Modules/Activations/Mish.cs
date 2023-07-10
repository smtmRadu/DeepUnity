using System;

namespace DeepUnity
{
    [Serializable]
    public class Mish : ActivationBase
    {
        protected override void Activation(ref Tensor x)
        {
            x = x.Select(x =>
            {
                float exp = MathF.Exp(x);
                float mish = x * MathF.Tanh(MathF.Log(1f + exp));
                return mish;
            });
        }

        protected override void Derivative(ref Tensor x)
        {
            x = x.Select(x =>
            {
                float exp = MathF.Exp(x);
                float sech = 1f / MathF.Cosh(MathF.Log(1f + exp));
                float mishDerivative = MathF.Tanh(MathF.Log(1f + exp)) + x * sech * sech;
                return mishDerivative;
            });
        }
    }
}
