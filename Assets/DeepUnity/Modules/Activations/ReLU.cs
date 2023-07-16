using System;

namespace DeepUnity
{
    [Serializable]
    public class ReLU : ActivationBase
    {
        protected override Tensor Activation(Tensor x) => x.Select(k => Math.Max(0f, k));
        protected override Tensor Derivative(Tensor x) => x.Select(k => k > 0f ? 1f : 0f);
    }
}