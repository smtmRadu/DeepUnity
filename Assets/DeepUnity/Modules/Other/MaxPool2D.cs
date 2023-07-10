using System;
using System.Collections.Generic;
using System.Linq;
using UnityEngine;
namespace DeepUnity
{
    public class MaxPool2D : IModule
    {
        private Tensor Input_Cache { get; set; }

        [SerializeField] private int kernel_size;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType padding_mode;


        /// <summary>
        /// H out = Floor((H in + 2 * padding)/kernel_size + 1)<br />
        /// W out = Floor((W in + 2 * padding)/kernel_size + 1)<br />
        /// </summary>
        public MaxPool2D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Mirror)
        {
            this.kernel_size = kernel_size;
            this.padding = padding;
            this.padding_mode = padding_mode;
        }

        public Tensor Predict(Tensor input)
        {
            int Wout = (int)Math.Floor((input.Width + 2 * padding) / (float)kernel_size + 1f);
            int Hout = (int)Math.Floor((input.Height + 2 * padding) / (float)kernel_size + 1f);
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            float[,,,] pooled_input = new float[input.Batch, input.Channels, Hout, Wout];

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

                            pooled_input[b, c, j, i] = values_pool.Max();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return Tensor.Constant(pooled_input);
        }
        public Tensor Forward(Tensor input)
        {
            Input_Cache = Tensor.Identity(input);

            int Wout = (int)Math.Floor((input.Width + 2 * padding)/ (float)kernel_size + 1f);
            int Hout = (int)Math.Floor((input.Height + 2 * padding) / (float)kernel_size + 1f);
            // 1. Apply padding
            input = Tensor.MatPad(input, padding, padding_mode);

            float[,,,] pooled_input = new float[input.Batch, input.Channels, Hout, Wout];
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

                            pooled_input[b, c, j, i] = values_pool.Max();

                            values_pool.Clear();
                        }
                    }
                }
            }


            return Tensor.Constant(pooled_input);
        }
        public Tensor Backward(Tensor input)
        {
            throw new System.NotImplementedException();
        }
    }
}