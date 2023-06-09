using System;
namespace DeepUnity
{
    [Serializable]
    public class SoftMax : ActivationBase, IModule
    {
        protected override void Activation(Tensor x)
        {
            for (int j = 0; j < x.Shape[1]; j++)
            {
                float exp_sum = 0f;
                for (int i = 0; i < x.Shape[0]; i++)
                {
                    float exp = MathF.Exp(x[i, j]);
                    x[i, j] = exp;
                    exp_sum += exp;
                }

                for (int i = 0; i < x.Shape[0]; i++)
                {
                    x[i, j] /= exp_sum;
                }
            }
        }
        protected override void Derivative(Tensor x)
        {
            for (int j = 0; j < x.Shape[1]; j++)
            {
                float exp_sum = 0f;
                for (int i = 0; i < x.Shape[0]; i++)
                {
                    float exp = MathF.Exp(x[i, j]);
                    x[i, j] = exp;
                    exp_sum += exp;
                }

                for (int i = 0; i < x.Shape[0]; i++)
                {
                    float exp = x[i, j];
                    x[i, j] = (exp * exp_sum - exp * exp) / (exp_sum * exp_sum);
                }
            }
        }
    }

}

