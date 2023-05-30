namespace DeepUnity
{
    public class Flatten : IModule
    {
        public Tensor<float> InputCache { get; set; }
        public Tensor<float> Forward(Tensor<float> input)
        {
            // input.shape = width x height x batch
            // output.shape = w*h x batch
            InputCache = input.Clone() as Tensor<float>;
            var shape = input.FullShape;
            
            Tensor<float> output = Tensor<float>.Zeros(shape[0] * shape[1], shape[3]);

            for (int k = 0; k < shape[3]; k++)
            {
                for (int i = 0; i < shape[0]; i++)
                {
                    for (int j = 0; j < shape[1]; j++)
                    {
                        output[i * shape[0] + j, k] = input[i, j, k];
                    }
                }
            }


            return output;
        }
        public Tensor<float> Backward(Tensor<float> loss)
        {
            // input.shape = width x height x batch
            // loss.shape = w*h x batch

            var backShape = InputCache.FullShape;
            Tensor<float> back = Tensor<float>.Zeros(backShape);

            for (int k = 0; k < backShape[3]; k++)
            {
                for (int i = 0; i < backShape[0]; i++)
                {
                    for (int j = 0; j < backShape[1]; j++)
                    {
                        back[i, j, k] = loss[i * backShape[0] + j, k];
                    }
                }
            }

            return back;

        }
    }

}

