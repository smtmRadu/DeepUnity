using System;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {
        private Tensor InputCache { get; set; }
        public Tensor Predict(Tensor input)
        {
            var shape = input.Shape;

            Tensor output = Tensor.Zeros(shape[0] * shape[1], shape[2]);

            for (int k = 0; k < shape[2]; k++)
            {
                for (int i = 0; i < shape[0]; i++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        output[i * shape[1] + j, k] = input[i, j, k];
                    }
                }
            }

            return output;
        }
        public Tensor Forward(Tensor input)
        {
            // input.shape = (width, height, batch)
            // output.shape = w*h by batch
            InputCache = Tensor.Identity(input);
            var shape = input.Shape;

            Tensor output = Tensor.Zeros(shape[0] * shape[1], shape[2]);

            for (int k = 0; k < shape[2]; k++)
            {
                for (int i = 0; i < shape[0]; i++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        output[i * shape[1] + j, k] = input[i, j, k];
                    }
                }
            }


            return output;
        }
        public Tensor Backward(Tensor loss)
        {
            // input.shape = width x height x batch
            // loss.shape = w*h x batch

            var backShape = InputCache.Shape;
            Tensor back = Tensor.Zeros(backShape);

            for (int k = 0; k < backShape[2]; k++)
            {
                for (int i = 0; i < backShape[0]; i++)
                {
                    for (int j = 0; j < backShape[1]; j++)
                    {
                        back[i, j, k] = loss[i * backShape[1] + j, k];
                    }
                }
            }

            return back;

        }
    }

}