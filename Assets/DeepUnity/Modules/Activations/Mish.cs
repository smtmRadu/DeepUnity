using System;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class Mish : IModule, IActivation
    {

        private Tensor InputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(x =>
            {
                float exp = MathF.Exp(x);
                float mish = x * MathF.Tanh(MathF.Log(1f + exp));
                return mish;
            });
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(x =>
            {
                float exp = MathF.Exp(x);
                float softplus = MathF.Log(1f + exp);
                float sech = 1f / MathF.Cosh(softplus);
                float sigmoid = exp / (1f + exp);
                float mishDerivative = MathF.Tanh(softplus) + x * sech * sech * sigmoid;
                return mishDerivative;
            });
        }


        public object Clone() => new Mish();
    }
}
