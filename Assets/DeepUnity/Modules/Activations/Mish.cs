using System;

namespace DeepUnity
{
    [Serializable]
    public class Mish : Activation
    {
        protected override Tensor Activate(Tensor x)
        {
            return x.Select(x => 
            {
                float exp = MathF.Exp(x);
                float mish = x * MathF.Tanh(MathF.Log(1f + exp));
                return mish;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float exp = MathF.Exp(x);
                float sech = 1f / MathF.Cosh(MathF.Log(1f + exp));
                float mishDerivative = MathF.Tanh(MathF.Log(1f + exp)) + x * sech * sech;
                return mishDerivative;
            });
        }

        public override object Clone() => new Mish();
    }
}
