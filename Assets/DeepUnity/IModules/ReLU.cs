using System;

namespace DeepUnity
{
    public sealed class ReLU : IModule, IActivation
    {
        public Tensor<float> InputCache { get; set; }
        public Tensor<float> Activation { get => InputCache.Select(x => Math.Max(0, x)); }
        public Tensor<float> Derivative { get => InputCache.Select(x => x > 0f ? 1f : 0f); }


        public Tensor<float> Forward(Tensor<float> input)
        {
            InputCache = input.Clone() as Tensor<float>;
            return Activation;
        }
        public Tensor<float> Backward(Tensor<float> loss)
        {
            return Derivative * loss;
        }
    }
}