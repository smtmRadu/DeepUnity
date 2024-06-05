using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
    /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
    /// where <br></br>
    /// H_out = H_in + 2 * height_padding, <br></br>
    /// W_out = W_in + 2 * width_padding.<br />
    /// </summary>
    [Serializable]
    public class ZeroPad2D : IModule
    {
        [SerializeField] private int hPadding;
        [SerializeField] private int wPadding;

        /// <summary>
        /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
        /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
        /// where <br></br>
        /// H_out = H_in + 2 * <paramref name="padding"/>, <br></br>
        /// W_out = W_in + 2 * <paramref name="padding"/>.<br />
        /// </summary>
        /// <param name="padding">Pad used for height and width.</param>
        public ZeroPad2D(int padding)
        {
            if (padding < 1)
                throw new ArgumentException("Padding cannot be less than 1.");
            this.hPadding = padding;
            this.wPadding = padding;
        }

        /// <summary>
        /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
        /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
        /// where <br></br>
        /// H_out = H_in + 2 * <paramref name="height_padding"/>, <br></br>
        /// W_out = W_in + 2 * <paramref name="width_padding"/>.<br />
        /// </summary>
        public ZeroPad2D(int height_padding, int width_padding)
        {
            if (height_padding < 1 || width_padding < 1)
                throw new ArgumentException("Paddings cannot be less than 1.");
            this.hPadding = height_padding;
            this.wPadding = width_padding;
        }
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batch_size = isBatched ? loss.Size(0) : 1;
            int channels = loss.Size(-3);
            int height = loss.Size(-2);
            int width = loss.Size(-1);
            Tensor inputGrad = isBatched ?
                Tensor.Zeros(batch_size, channels, height - 2*hPadding, width - 2 * wPadding) :
                Tensor.Zeros(channels, height - 2 * hPadding, width - 2 * wPadding);

            Parallel.For(0, batch_size, b =>
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height - 2 * hPadding; h++)
                    {
                        for (int w = 0; w < width - 2 * wPadding; w++)
                        {
                            inputGrad[b, c, h, w] = loss[b, c, h + hPadding, w + wPadding];
                        }
                    }
                }
            });

            return inputGrad;
        }

        public object Clone()
        {
            return new ZeroPad2D(hPadding, wPadding);
        }

        public Tensor Forward(Tensor input)
        {
            return Predict(input);
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 3)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H, W) or (C, H, W).");

            int batch_size = input.Rank == 4 ? input.Size(0) : 1;
            int channels = input.Size(-3);
            int height = input.Size(-2);
            int width = input.Size(-1);

            Tensor padd_input = input.Rank == 4 ?
                Tensor.Zeros(batch_size, channels, height + 2 * hPadding, width + 2 * wPadding) :
                Tensor.Zeros(channels, height + 2 * hPadding, width + 2 * wPadding);

            Parallel.For(0, batch_size, b =>
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height; h++)
                    {
                        for (int w = 0; w < width; w++)
                        {
                            padd_input[b, c, h + hPadding, w + wPadding] = input[b, c, h, w];
                        }
                    }
                }
            });

            return padd_input;
        }
    }
}



