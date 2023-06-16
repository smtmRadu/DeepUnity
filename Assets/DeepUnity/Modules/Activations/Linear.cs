using System;

namespace DeepUnity
{
    [Serializable]
    public class Linear : ActivationBase
    {
        protected override void Activation(ref Tensor x) { }
        protected override void Derivative(ref Tensor x) => x.ForEach(x => 1f);
    }

}

