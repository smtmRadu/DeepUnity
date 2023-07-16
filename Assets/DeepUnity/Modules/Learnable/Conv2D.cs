using System;
using System.Collections.Generic;
using System.Linq;

namespace DeepUnity
{
    [Serializable]
    public class Conv2D : Learnable, IModule 
    {
        // https://www.youtube.com/watch?v=Lakz2MoHy6o
        private Tensor InputCache { get; set; }

        /// Biases are applied over the final output. Biases (out_channels, out_height, out_width).
        // input shape = [batch, Ichannels, Iheight, Iwidth]
        // output_shape = [batch, Ochannels, Iheight - K + 1, Iwidth - K + 1] 
        // In Conv2D, Gamma represents kernels, Beta represents biases
        public Conv2D(IEnumerable<int> input_shape, int out_channels, int kernel_size)
        {          
            if(input_shape.Count() > 3)
                throw new Exception("input_shape cannot have more than 3 dimensions. Please specificy a shape like (C,H,W) or (H, W).");


            int in_width = input_shape.Last();
            int in_height = input_shape.ElementAt(input_shape.Count() - 2);
            int in_channels = input_shape.Count() == 3? input_shape.ElementAt(0) : 1;

            int out_width = in_width - kernel_size + 1;
            int out_height = in_height - kernel_size + 1;

            gamma = Tensor.RandomNormal((0, 1), out_channels, in_channels, kernel_size, kernel_size);
            gradGamma = Tensor.Zeros(out_channels, in_channels, kernel_size, kernel_size);

            beta = Tensor.RandomNormal((0, 1), out_channels, out_height, out_width);
            gradBeta = Tensor.Zeros(out_channels, out_height, out_width);
        }
        public Tensor Predict(Tensor input)
        {
            bool isBatched = input.Rank == 4;
            int batch_size = isBatched ? gamma.Size(-4) : 1;
            return Tensor.Correlate2D(input, gamma, CorrelationMode.Valid) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
        }
        public Tensor Forward(Tensor input)
        {
            bool isBatched = input.Rank == 4;

            InputCache = Tensor.Identity(input);

            int batch_size = isBatched ? input.Size(-4) : 1;
            return Tensor.Correlate2D(input, gamma, CorrelationMode.Valid) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
        }
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batch_size = isBatched ? loss.Size(-4) : 1;
            gradGamma += Tensor.Correlate2D(InputCache, loss, CorrelationMode.Valid) / batch_size;
            gradBeta += loss / batch_size;

            return Tensor.Convolve2D(loss, gamma, CorrelationMode.Full);
        }

    }
}

