using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
    /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
    /// where <br></br>
    /// H_out = H_in + 2 * padding, <br></br>
    /// W_out = W_in + 2 * padding.<br />
    /// </summary>
    [Serializable]
    public class ZeroPad2D : IModule
    {
        [SerializeField] private int padding;

        /// <summary>
        /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
        /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
        /// where <br></br>
        /// H_out = H_in + 2 * padding, <br></br>
        /// W_out = W_in + 2 * padding.<br />
        /// </summary>
        public ZeroPad2D(int padding)
        {
            if (padding < 1)
                throw new ArgumentException("Padding cannot be less than 1.");
            this.padding = padding;
        }

        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batch_size = isBatched ? loss.Size(0) : 1;
            int channels = loss.Size(-3);
            int height = loss.Size(-2);
            int width = loss.Size(-1);
            Tensor inputGrad = isBatched ?
                Tensor.Zeros(batch_size, channels, height - 2*padding, width - 2*padding) :
                Tensor.Zeros(channels, height - 2 * padding, width - 2 * padding);

            for (int b = 0; b < batch_size; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height - 2 * padding; h++)
                    {
                        for (int w = 0; w < width - 2 * padding; w++)
                        {
                            inputGrad[b,c, h, w] = loss[b,c,h + padding,w + padding];
                        }
                    }
                }
            }



            return inputGrad;
        }

        public object Clone()
        {
            return new ZeroPad2D(padding);
        }

        public Tensor Forward(Tensor input)
        {
            if (input.Rank < 3)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H, W) or (C, H, W).");

            return Tensor.MatPad(input, padding, PaddingType.Zeros);
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 3)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H, W) or (C, H, W).");

            return Tensor.MatPad(input, padding, PaddingType.Zeros);
        }
    }
}



