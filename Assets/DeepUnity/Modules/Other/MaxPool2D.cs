using System.Collections.Generic;
using System.Linq;
using UnityEngine;
namespace DeepUnity
{
    public class MaxPool2D : IModule
    {
        [SerializeField] private int kernel_size;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType padding_mode;

        public MaxPool2D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Mirror)
        {
            this.kernel_size = kernel_size;
            this.padding = padding;
        }

        public Tensor Predict(Tensor input)
        {
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            // 2. Pool by stride
            int[] pooled_i_shape = input.Shape.ToArray();
            pooled_i_shape[3] /= kernel_size;
            pooled_i_shape[2] /= kernel_size;
            Tensor pooled_input = Tensor.Zeros(pooled_i_shape);


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
                                    values_pool.Add(input[i * 2 + pi, j * 2 + pj, c, b]);
                                }
                            }

                            pooled_input[i, j, c, b] = values_pool.Max();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return pooled_input;
        }
        public Tensor Forward(Tensor input)
        {
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            // 2. Pool by stride
            int[] pooled_i_shape = input.Shape.ToArray();
            pooled_i_shape[3] /= kernel_size;
            pooled_i_shape[2] /= kernel_size;
            Tensor pooled_input = Tensor.Zeros(pooled_i_shape);


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
                                    values_pool.Add(input[i * 2 + pi, j * 2 + pj, c, b]);
                                }
                            }

                            pooled_input[i, j, c, b] = values_pool.Max();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return pooled_input;
        }
        public Tensor Backward(Tensor input)
        {
            return null;
        }
    }
}