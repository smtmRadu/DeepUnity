using System;
namespace DeepUnity
{
    [Serializable]
    public class SoftMax : ActivationBase
    {
        protected override Tensor Activation(Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);        
            return  exp / exp_sum;
        }
        protected override Tensor Derivative(Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);
            return (exp * exp_sum - exp * exp) / (exp_sum * exp_sum);
        }
    }

}

