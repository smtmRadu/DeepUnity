using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;

namespace DeepUnity
{
    public class AvgPool2D : IModule
    {

        private Tensor Input_Cache { get; set; }

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
            int Wout = (int)Math.Floor((input.Size(TDim.width) + 2 * padding) / (float)kernel_size + 1f);
            int Hout = (int)Math.Floor((input.Size(TDim.height) + 2 * padding) / (float)kernel_size + 1f);
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


            return pooled_input;
        }
        public Tensor Forward(Tensor input)
        {
            Input_Cache = Tensor.Identity(input);

            int Wout = (int)Math.Floor((input.Size(TDim.width) + 2 * padding) / (float)kernel_size + 1f);
            int Hout = (int)Math.Floor((input.Size(TDim.height) + 2 * padding) / (float)kernel_size + 1f);
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


            return pooled_input;
        }
        public Tensor Backward(Tensor loss)
        {
            // Initialize a tensor for gradients with the same shape as the input tensor
            Tensor gradInput = Tensor.Zeros(Input_Cache.Shape.ToArray());

            for (int b = 0; b < loss.Shape.Batch; b++)
            {
                // Foreach channel
                for (int c = 0; c < loss.Shape.Channels; c++)
                {
                    // foreach pool result
                    for (int i = 0; i < loss.Shape.Width; i++)
                    {
                        for (int j = 0; j < loss.Shape.Height; j++)
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

            return gradInput;
        }
    }
}