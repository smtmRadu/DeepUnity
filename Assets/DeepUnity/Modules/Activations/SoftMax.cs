using System;
namespace DeepUnity
{
    [Serializable]
    public class Softmax : ActivationBase
    {
        protected override Tensor Activation(Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);
            exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));        
            return  exp / exp_sum;
        }
        protected override Tensor Derivative(Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);
            exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));
            return (exp * exp_sum - exp * exp) / (exp_sum * exp_sum);
        }
    }

}

