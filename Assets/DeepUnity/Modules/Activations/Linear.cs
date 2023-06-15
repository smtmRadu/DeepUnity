using System;

namespace DeepUnity
{
    [Serializable]
    public class Linear : ActivationBase
    {
        protected override void Activation(Tensor x) { }
        protected override void Derivative(Tensor x) => x.ForEach(x => 1f);
    }

}

