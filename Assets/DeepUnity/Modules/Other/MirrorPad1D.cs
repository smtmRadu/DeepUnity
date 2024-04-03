using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(B, C, H_in)</b> or <b>(C, H_in)</b> <br></br>
    /// Output: <b>(B, C, H_out)</b> or <b>(C, H_out)</b> <br></br>
    /// where <br></br>
    /// H_out = H_in + 2 * padding, <br></br>
    /// </summary>
    [Serializable]
    public class MirrorPad1D : IModule
    {
        [SerializeField] private int padding;

        /// <summary>
        /// Input: <b>(B, C, H_in)</b> or <b>(C, H_in)</b> <br></br>
        /// Output: <b>(B, C, H_out)</b> or <b>(C, H_out)</b> <br></br>
        /// where <br></br>
        /// H_out = H_in + 2 * padding, <br></br>
        /// </summary>
        public MirrorPad1D(int padding)
        {
            if (padding < 1)
                throw new ArgumentException("Padding cannot be less than 1.");
            this.padding = padding;
        }

        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 3;
            int batch_size = isBatched ? loss.Size(0) : 1;
            int channels = loss.Size(-2);
            int height = loss.Size(-1);

            Tensor inputGrad = isBatched ?
                Tensor.Zeros(batch_size, channels, height - 2 * padding) :
                Tensor.Zeros(channels, height - 2 * padding);

            for (int b = 0; b < batch_size; b++)
            {
                for (int c = 0; c < channels; c++)
                {
                    for (int h = 0; h < height - 2 * padding; h++)
                    {
                        inputGrad[b, c, h] = loss[b, c, h + padding];
                    }
                }
            }



            return inputGrad;
        }

        public object Clone()
        {
            return new MirrorPad1D(padding);
        }

        public Tensor Forward(Tensor input)
        {
            if (input.Rank < 2)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H) or (C, H).");

            return Tensor.VecPad(input, padding, PaddingType.Mirror);
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 2)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H) or (C, H).");

            return Tensor.VecPad(input, padding, PaddingType.Mirror);
        }
    }
}



