using System;
using DeepUnity.Layers;

namespace DeepUnity.Activations
{
    [Serializable]
    public class Mish : IModule, IActivation
    {

        protected Tensor InputCache { get; set; }
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
                float sech = 1f / MathF.Cosh(MathF.Log(1f + exp));
                float mishDerivative = MathF.Tanh(MathF.Log(1f + exp)) + x * sech * sech;
                return mishDerivative;
            });
        }


        public object Clone() => new Mish();
    }
}
