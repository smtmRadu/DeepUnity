using System;

namespace DeepUnity
{
    public sealed class ReLU : IModule, IActivation
    {
        public Tensor InputCache { get; set; }
        public Tensor Activation { get => InputCache.Select(x => Math.Max(0, x)); }
        public Tensor Derivative { get => InputCache.Select(x => x > 0f ? 1f : 0f); }


        public Tensor Forward(Tensor input)
        {
            InputCache = input.Clone() as Tensor;
            return Activation;
        }
        public Tensor Backward(Tensor loss)
        {
            return Derivative * loss;
        }
    }
}