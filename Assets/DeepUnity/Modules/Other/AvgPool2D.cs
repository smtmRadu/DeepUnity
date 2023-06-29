using System.Collections.Generic;
using System.Linq;
using UnityEngine;
namespace DeepUnity
{
    public class AvgPool2D : IModule
    {
        [SerializeField] private int kernel_size;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType padding_mode;

        /// <summary>
        /// H out = H in + 2 * padding - (kernel_size - 1) - 1 <br />
        /// W out = W in + 2 * padding - (kernel_size - 1) - 1 <br />
        /// </summary>
        /// <param name="kernel_size"></param>
        /// <param name="padding"></param>
        /// <param name="padding_mode"></param>
        public AvgPool2D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Mirror)
        {
            this.kernel_size = kernel_size;
            this.padding = padding;
        }

        public Tensor Predict(Tensor input)
        {
            int Wout = input.Size(TDim.width) + 2 * padding - (kernel_size - 1) - 1;
            int Hout = input.Size(TDim.height) + 2 * padding - (kernel_size - 1) - 1;
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);
            
            Tensor pooled_input = Tensor.Zeros(input.Size(TDim.batch), input.Size(TDim.channel), Hout, Wout);

            List<float> values_pool = new List<float>();

            for (int b = 0; b < pooled_input.Shape.Batch; b++)
            {
                // Foreach channel
                for (int c = 0; c < pooled_input.Shape.Channels; c++)
                {
                    // foreach pool result
                    for (int i = 0; i < pooled_input.Shape.Width; i++)
                    {
                        for (int j = 0; j < pooled_input.Shape.Height; j++)
                        {

                            // foreach pool element in the pool
                            for (int pi = 0; pi < input.Shape.Width; pi++)
                            {
                                for (int pj = 0; pj < input.Shape.Height; pj++)
                                {
                                    values_pool.Add(input[b,c, j * 2 + pj, i * 2 + pi]);
                                }
                            }

                            pooled_input[b,c,j,i] = values_pool.Average();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return pooled_input;
        }
        public Tensor Forward(Tensor input)
        {
            int Wout = input.Size(TDim.width) + 2 * padding - (kernel_size - 1) - 1;
            int Hout = input.Size(TDim.height) + 2 * padding - (kernel_size - 1) - 1;

            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            Tensor pooled_input = Tensor.Zeros(input.Size(TDim.batch), input.Size(TDim.channel), Hout, Wout);

            List<float> values_pool = new List<float>();

            for (int b = 0; b < pooled_input.Shape.Batch; b++)
            {
                // Foreach channel
                for (int c = 0; c < pooled_input.Shape.Channels; c++)
                {
                    // foreach pool result
                    for (int i = 0; i < pooled_input.Shape.Width; i++)
                    {
                        for (int j = 0; j < pooled_input.Shape.Height; j++)
                        {

                            // foreach pool element in the pool
                            for (int pi = 0; pi < input.Shape.Width; pi++)
                            {
                                for (int pj = 0; pj < input.Shape.Height; pj++)
                                {
                                    values_pool.Add(input[b, c, j * 2 + pj, i * 2 + pi]);
                                }
                            }

                            pooled_input[b, c, j, i] = values_pool.Average();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return pooled_input;
        }
        public Tensor Backward(Tensor input)
        {
            throw new KeyNotFoundException();
        }
    }
}