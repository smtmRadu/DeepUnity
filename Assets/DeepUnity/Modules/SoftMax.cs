using System;
using Unity.VisualScripting;

namespace DeepUnity
{
    public class SoftMax : ActivationBase, IModule
    {
        protected override Tensor InputCache { get; set; }

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

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            Activation(input);
            return input;
        }
        public Tensor Backward(Tensor loss)
        {
            Derivative(InputCache);
            return InputCache * loss;
        }
    }

}

