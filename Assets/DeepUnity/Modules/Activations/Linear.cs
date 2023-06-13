using System;

namespace DeepUnity
{
    [Serializable]
    public class Linear : ActivationBase
    {
        protected override void Activation(NDArray x) { }
        protected override void Derivative(NDArray x) => x.ForEach(x => 1f);
    }

}

