using System;
namespace DeepUnity
{
    [Serializable]
    public class Conv2D : Learnable, IModule 
    {
        // https://www.youtube.com/watch?v=Lakz2MoHy6o
        private Tensor Input_Cache { get; set; }

        /// Biases are applied over the final output. Biases (out_channels, out_height, out_width).
        // input shape = [batch, Ichannels, Iheight, Iwidth]
        // output_shape = [batch, Ochannels, Iheight - K + 1, Iwidth - K + 1] 
        // In Conv2D, Gamma represents kernels, Beta represents biases
        public Conv2D(int in_channels, int out_channels, int kernel_size)
        {          
            gamma = Tensor.RandomNormal((0, 1), out_channels, in_channels, kernel_size, kernel_size);
            gradGamma = Tensor.Zeros(out_channels, in_channels, kernel_size, kernel_size);

            // beta is initialized on Forward or predict because we do not know the size of the input w and h
        }
        public Tensor Predict(Tensor input)
        {

            if (beta == null)
            {
                int kernel_size = gamma.Width;

                int out_channels = gamma.Batch;
                int h = input.Height - kernel_size + 1;
                int w = input.Width - kernel_size + 1;

                beta = Tensor.RandomNormal((0, 1), out_channels, h, w);
                gradBeta = Tensor.Zeros(out_channels, h, w);
            }

            int batch_size = input.Batch;
            return Tensor.Correlate2D(input, gamma, CorrelationMode.Valid) + Tensor.Expand(beta, Dim.batch, batch_size);
        }
        public Tensor Forward(Tensor input)
        {
            if(beta == null)
            {
                int kernel_size = gamma.Width;

                int out_channels = gamma.Batch;
                int h = input.Height - kernel_size + 1;
                int w = input.Width - kernel_size + 1;

                beta = Tensor.RandomNormal((0, 1), out_channels, h, w);
                gradBeta = Tensor.Zeros(out_channels, h, w);
            }


            Input_Cache = Tensor.Identity(input);
            int batch_size = input.Batch;
            return Tensor.Correlate2D(input, gamma, CorrelationMode.Valid) + Tensor.Expand(beta, Dim.batch, batch_size);
        }
        public Tensor Backward(Tensor loss)
        {
            int batch_size = loss.Batch;
            gradGamma += Tensor.Correlate2D(Input_Cache, loss, CorrelationMode.Valid) / batch_size;
            gradBeta += loss / batch_size;

            return Tensor.Convolve2D(loss, gamma, CorrelationMode.Full);
        }

    }
}

