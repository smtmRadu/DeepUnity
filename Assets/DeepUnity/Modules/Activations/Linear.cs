using System;

namespace DeepUnity
{
    [Serializable]
    public class Linear : ActivationBase
    {
        protected override Tensor Activation(Tensor x) => x;
        protected override Tensor Derivative(Tensor y) => Tensor.Ones(y.Shape);
    }

}

