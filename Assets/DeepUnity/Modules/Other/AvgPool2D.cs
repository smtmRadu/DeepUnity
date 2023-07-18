using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DeepUnity
{
    public class AvgPool2D : IModule
    {

        private Tensor InputCache { get; set; }

        [SerializeField] private int kernel_size;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType padding_mode;

        /// <summary>
        /// H out = Floor((H in + 2 * padding)/kernel_size + 1)<br />
        /// W out = Floor((W in + 2 * padding)/kernel_size + 1)<br />
        /// </summary>
        public AvgPool2D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Mirror)
        {
            this.kernel_size = kernel_size;
            this.padding = padding;
            this.padding_mode = padding_mode;
        }

        public Tensor Predict(Tensor input)
        {
            int Wout = (int)Math.Floor((input.Size(-1) + 2 * padding) / (float)kernel_size + 1f);
            int Hout = (int)Math.Floor((input.Size(-2) + 2 * padding) / (float)kernel_size + 1f);
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            int batch_size = input.Rank == 4 ? input.Size(-4) : 1;
            int channel_size = input.Rank >= 3 ? input.Size(-3) : 1;

            float[,,,] pooled_input = new float[batch_size, channel_size, Hout, Wout];

            List<float> values_pool = new List<float>();

            for (int b = 0; b < pooled_input.GetLength(0); b++)
            {
                // Foreach channel
                for (int c = 0; c < pooled_input.GetLength(1); c++)
                {
                    // foreach pool result
                    for (int i = 0; i < pooled_input.GetLength(3); i++)
                    {
                        for (int j = 0; j < pooled_input.GetLength(2); j++)
                        {

                            // foreach pool element in the pool
                            for (int pi = 0; pi < kernel_size; pi++)
                            {
                                for (int pj = 0; pj < kernel_size; pj++)
                                {
                                    try
                                    {
                                        values_pool.Add(input[b, c, j * kernel_size + pj, i * kernel_size + pi]);
                                    }
                                    catch { }

                                }
                            }

                            pooled_input[b, c, j, i] = values_pool.Average();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return Tensor.Constant(pooled_input);
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            int Wout = (int)Math.Floor((input.Size(-1) + 2 * padding) / (float)kernel_size + 1f);
            int Hout = (int)Math.Floor((input.Size(-2) + 2 * padding) / (float)kernel_size + 1f);
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            int batch_size = input.Rank == 4 ? input.Size(-4) : 1;
            int channel_size = input.Rank >= 3 ? input.Size(-3) : 1;

            float[,,,] pooled_input = new float[batch_size, channel_size, Hout, Wout];

            List<float> values_pool = new List<float>();

            for (int b = 0; b < pooled_input.GetLength(0); b++)
            {
                // Foreach channel
                for (int c = 0; c < pooled_input.GetLength(1); c++)
                {
                    // foreach pool result
                    for (int i = 0; i < pooled_input.GetLength(3); i++)
                    {
                        for (int j = 0; j < pooled_input.GetLength(2); j++)
                        {

                            // foreach pool element in the pool
                            for (int pi = 0; pi < kernel_size; pi++)
                            {
                                for (int pj = 0; pj < kernel_size; pj++)
                                {
                                    try
                                    {
                                        values_pool.Add(input[b, c, j * kernel_size + pj, i * kernel_size + pi]);
                                    }
                                    catch { }

                                }
                            }

                            pooled_input[b, c, j, i] = values_pool.Average();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return Tensor.Constant(pooled_input);
        }
        public Tensor Backward(Tensor loss)
        {
            int Batch = loss.Rank == 4 ? loss.Size(-4) : 1;
            int Channels = loss.Rank >= 3 ? loss.Size(-3) : 1;
            int Height = loss.Size(-2);
            int Width = loss.Size(-1);

            // Initialize a tensor for gradients with the same shape as the input tensor
            float[,,,] gradInput = new float[Batch, Channels, Height, Height];


           

            for (int b = 0; b < Batch; b++)
            {
                // Foreach channel
                for (int c = 0; c < Channels; c++)
                {
                    // foreach pool result
                    for (int i = 0; i < Width; i++)
                    {
                        for (int j = 0; j < Height; j++)
                        {
                            // Get the gradient for the current pooled output element
                            float grad = loss[b, c, j, i];

                            // Compute the average gradient for the pool elements
                            float avgGrad = grad / (kernel_size * kernel_size);

                            // Distribute the average gradient to the corresponding locations in the input tensor
                            for (int pi = 0; pi < kernel_size; pi++)
                            {
                                for (int pj = 0; pj < kernel_size; pj++)
                                {
                                    int inputIndexX = i * kernel_size + pi;
                                    int inputIndexY = j * kernel_size + pj;

                                    // Set the gradient value in the input tensor
                                    gradInput[b, c, inputIndexY, inputIndexX] = avgGrad;
                                }
                            }
                        }
                    }
                }
            }

            return Tensor.Constant(gradInput);
        }
    }
}