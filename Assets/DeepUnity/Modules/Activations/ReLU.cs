using System;

namespace DeepUnity
{
    [Serializable]
    public class ReLU : ActivationBase
    {
        protected override void Activation(NDArray x) => x.ForEach(k => Math.Max(0f, k));
        protected override void Derivative(NDArray x) => x.ForEach(k => k > 0f ? 1f : 0f);
    }
}