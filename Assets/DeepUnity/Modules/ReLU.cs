using System;

namespace DeepUnity
{
    [Serializable]
    public class ReLU : ActivationBase, IModule 
    {
        protected override void Activation(Tensor x) => x.ForEach(k => Math.Max(0f, k));
        protected override void Derivative(Tensor x) => x.ForEach(k => k > 0f ? 1f : 0f);
    }
}