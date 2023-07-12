using System;
namespace DeepUnity
{
    [Serializable]
    public class SoftMax : ActivationBase
    {
        protected override void Activation(ref Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum;
            if (IsBatchedInput(x))
            {
                exp_sum = Tensor.Sum(exp, 1, true); // [batch, features]
            }
            else
            {
                exp_sum = Tensor.Sum(exp, 0, true); // [features]
            }
            
            x = exp / exp_sum;
        }
        protected override void Derivative(ref Tensor x)
        {
            Tensor exp = Tensor.Exp(x);
            Tensor exp_sum;
            if (IsBatchedInput(x))
            {
                exp_sum = Tensor.Sum(exp, 1, true); // [batch, features]
            }
            else
            {
                exp_sum = Tensor.Sum(exp, 0, true); // [features]
            }
            x = (exp * exp_sum - exp * exp) / (exp_sum * exp_sum);
        }
    }

}

