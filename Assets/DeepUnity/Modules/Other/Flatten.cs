using System;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {
        private NDArray InputCache { get; set; }
        public NDArray Predict(NDArray input)
        {
            var shape = input.Shape;

            NDArray output = NDArray.Zeros(shape[0] * shape[1], shape[2]);

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
        public NDArray Forward(NDArray input)
        {
            // input.shape = (width, height, batch)
            // output.shape = w*h by batch
            InputCache = NDArray.Identity(input);
            var shape = input.Shape;

            NDArray output = NDArray.Zeros(shape[0] * shape[1], shape[2]);

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
        public NDArray Backward(NDArray loss)
        {
            // input.shape = width x height x batch
            // loss.shape = w*h x batch

            var backShape = InputCache.Shape;
            NDArray back = NDArray.Zeros(backShape);

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