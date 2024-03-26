using DeepUnity.Modules;
using System;
namespace DeepUnity.Activations
{
    [Serializable]
    public class SELU : IModule, IActivation
    {
        public SELU() { }

        private const float alpha = 1.6732632423543772848170429916717f;
        private const float scale = 1.0507009873554804934193349852946f;

        private Tensor InputCache { get; set; }
        public Tensor Predict(Tensor input)
        {
            return input.Select(x => scale * (MathF.Max(0, x) + MathF.Min(0, alpha * (MathF.Exp(x) - 1))));
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = input.Clone() as Tensor;
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(x =>
            {
                if (x < 0)
                    return alpha * scale * MathF.Exp(x);
                else
                    return alpha * scale * MathF.Exp(x) + scale;
            });
        }


        public object Clone() => new SELU();

    }

}


