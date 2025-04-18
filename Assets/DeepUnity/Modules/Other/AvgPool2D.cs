using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    /// <summary>
    /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
    /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
    /// where <br></br>
    /// H_out = Floor((H_in + 2 * padding - kernel_size - 1) / kernel_size + 1)<br />
    /// W_out = Floor((W_in + 2 * padding - kernel_size - 1) / kernel_size + 1)<br />
    /// </summary>
    [Serializable]
    public class AvgPool2D : IModule
    {

        private Tensor InputCache { get; set; }

        [SerializeField] private int kernel_size;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType padding_mode;

        /// <summary>
        /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
        /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
        /// where <br></br>
        /// H_out = Floor((H_in + 2 * padding - kernel_size - 1) / kernel_size + 1)<br />
        /// W_out = Floor((W_in + 2 * padding - kernel_size - 1) / kernel_size + 1)<br />
        /// </summary>
        public AvgPool2D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Mirror)
        {
            if (padding < 0)
                throw new ArgumentException("Padding cannot be less than 0");
            if (kernel_size < 2)
                throw new ArgumentException("Kernel Size cannot be less than 2");

            this.kernel_size = kernel_size;
            this.padding = padding;
            this.padding_mode = padding_mode;
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 3)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H, W) or (C, H, W).");

            int H_in = input.Size(-2);
            int W_in = input.Size(-1);
            int H_out = (int)Math.Floor((H_in + 2 * padding - 1 * (kernel_size - 1) - 1) / (float)kernel_size + 1f);
            int W_out = (int)Math.Floor((W_in + 2 * padding - 1 * (kernel_size - 1) - 1) / (float)kernel_size + 1f);

            // Apply padding
            if (padding > 0)
                input = Tensor.MatPad(input, padding, padding_mode);

            if (H_out < 1 || W_out < 1)
                throw new ShapeException($"The input shape {input.Shape.ToCommaSeparatedString()} is smaller than the kernel {kernel_size}x{kernel_size} in avg2d pooling layer.");


            bool isBatched = input.Rank == 4;
            int batch_size = isBatched ? input.Size(-4) : 1;
            int channel_size = input.Rank >= 3 ? input.Size(-3) : 1;

            Tensor pooled_input = isBatched ?
                Tensor.Zeros(batch_size, channel_size, H_out, W_out) :
                Tensor.Zeros(channel_size, H_out, W_out);

            Parallel.For(0, batch_size, b =>
            {           
                Parallel.For(0, channel_size, c =>
                {
                    LinkedList<float> values_pool = new LinkedList<float>();

                    // foreach pool result
                    for (int j = 0; j < H_out; j++)
                    {
                        for (int i = 0; i < W_out; i++)
                        {
                            // foreach pool element in the pool
                            for (int kj = 0; kj < kernel_size; kj++)
                            {
                                for (int ki = 0; ki < kernel_size; ki++)
                                {
                                    try
                                    {
                                        values_pool.AddLast(input[b, c, j * kernel_size + kj, i * kernel_size + ki]);
                                    }
                                    catch { }

                                }
                            }

                            pooled_input[b, c, j, i] = values_pool.Average();
                            values_pool.Clear();
                        }
                    }
                });
            });

            return pooled_input;
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            return Predict(input);
        }
        public Tensor Backward(Tensor loss)
        {
            // How does backprop of avgpool2d works.
            // Consider Input = 2x2 mapped on Output = 1x1.
            // Example
            // Input = [[1.0,3.0],[2.0, 0.0]]
            // Output = [1.5]
            // Loss = [0.5]
            // GradInput = [[0.125, 0.125],[0.125, 0.125]].
            // gradient value is splitted foreach value in the pool.


            bool isBatched = loss.Rank == 4;
            int Batch = isBatched ? loss.Size(-4) : 1;
            int Channels = loss.Rank >= 3 ? loss.Size(-3) : 1;
            int H_out = loss.Size(-2);
            int W_out = loss.Size(-1);
            int H_in = InputCache.Size(-2);
            int W_in = InputCache.Size(-1);

            Tensor gradInput = isBatched ?
              Tensor.Zeros(Batch, Channels, H_in, W_in) :
              Tensor.Zeros(Channels, H_in, W_in);

            Parallel.For(0, Batch, b =>
            {
                Parallel.For(0, Channels, c =>
                {
                    for (int i = 0; i < W_out; i++)
                    {
                        for (int j = 0; j < H_out; j++)
                        {
                            float averageValue = loss[b, c, j, i] / (kernel_size * kernel_size);

                            for (int pi = 0; pi < kernel_size; pi++)
                            {
                                for (int pj = 0; pj < kernel_size; pj++)
                                {
                                    int rowIndex = j * kernel_size + pj;
                                    int colIndex = i * kernel_size + pi;

                                    if (rowIndex >= 0 && colIndex >= 0 && rowIndex < H_in && colIndex < W_in)
                                    {
                                        gradInput[b, c, rowIndex, colIndex] += averageValue;
                                    }
                                }
                            }
                        }
                    }
                });
            });
            

            return gradInput;
        }


        public object Clone() => new AvgPool2D(kernel_size, padding, padding_mode);
    }
}