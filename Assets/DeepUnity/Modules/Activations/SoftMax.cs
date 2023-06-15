using System;
using Unity.VisualScripting;

namespace DeepUnity
{
    [Serializable]
    public class SoftMax : ActivationBase
    {
        protected override void Activation(Tensor x)
        {
            // [batch, width]
            Tensor exp = Tensor.Exp(x); // [batch, width]
            Tensor exp_sum = Tensor.Sum(exp, 0); // [batch, 1]
            exp_sum = Tensor.Expand(exp_sum, 1, exp.Shape.width); // [batch, width]

            x = exp / exp_sum;
        }
        protected override void Derivative(Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum = Tensor.Sum(exp, 0); // [batch, 1]
            exp_sum = Tensor.Expand(exp_sum, 1, exp.Shape.width); // [batch, width]

            x = (exp * exp_sum - exp * exp) / (exp_sum * exp_sum);
        }
    }

}

