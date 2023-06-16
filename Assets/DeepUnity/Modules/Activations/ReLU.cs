using System;

namespace DeepUnity
{
    [Serializable]
    public class ReLU : ActivationBase
    {
        protected override void Activation(ref Tensor x) => x.ForEach(k => Math.Max(0f, k));
        protected override void Derivative(ref Tensor x) => x.ForEach(k => k > 0f ? 1f : 0f);
    }
}