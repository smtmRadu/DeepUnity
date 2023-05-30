using System;

namespace DeepUnity
{
    public class SoftMax : IModule
    {
        public Tensor<float> InputCache { get; set; }
        public Tensor<float> Forward(Tensor<float> input)
        {
            InputCache = input.Clone() as Tensor<float>;
            var shape = input.FullShape;
            Tensor<float> output = Tensor<float>.Zeros(shape);
            for (int j = 0; j < shape[1]; j++)
            {
                float exp_sum = 0f;
                for (int i = 0; i < shape[0]; i++)
                {
                    float exp = MathF.Exp(input[i, j]);
                    output[i, j] = exp;
                    exp_sum += exp;
                }

                for (int i = 0; i < shape[0]; i++)
                {
                    output[i,j] /= exp_sum;
                }
            }

            return output;
        }
        public Tensor<float> Backward(Tensor<float> loss)
        {
            var shape = loss.FullShape;
            Tensor<float> back = Tensor<float>.Zeros(shape);
            for (int j = 0; j < shape[1]; j++)
            {
                float exp_sum = 0f;
                for (int i = 0; i < shape[0]; i++)
                {
                    float exp = MathF.Exp(loss[i, j]);
                    back[i, j] = exp;
                    exp_sum += exp;
                }

                for (int i = 0; i < shape[0]; i++)
                {
                    float exp = back[i, j];
                    back[i, j]  = (exp * exp_sum - exp * exp) / (exp_sum * exp_sum);
                }
            }

            return back;
        }
    }

}

