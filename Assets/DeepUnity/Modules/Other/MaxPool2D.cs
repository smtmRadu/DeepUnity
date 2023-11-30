using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;
namespace DeepUnity
{
    /// <summary>
    /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
    /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
    /// where <br></br>
    /// H_out = Floor((H_in + 2 * padding - kernel_size - 1) / kernel_size + 1),<br />
    /// W_out = Floor((W_in + 2 * padding - kernel_size - 1) / kernel_size + 1).<br />
    /// </summary>
    [Serializable]
    public class MaxPool2D : IModule
    {
        private Tensor InputCache { get; set; }

        [SerializeField] private int kernel_size;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType padding_mode;

        /// <summary>
        /// Input: <b>(B, C, H_in, W_in)</b> or <b>(C, H_in, W_in)</b> <br></br>
        /// Output: <b>(B, C, H_out, W_out)</b> or <b>(C, H_out, W_out)</b> <br></br>
        /// where <br></br>
        /// H_out = Floor[(H_in + 2 * <paramref name="padding"/> - <paramref name="kernel_size"/>) / <paramref name="kernel_size"/> + 1],<br />
        /// W_out = Floor[(W_in + 2 * <paramref name="padding"/> - <paramref name="kernel_size"/>) / <paramref name="kernel_size"/> + 1].<br />
        /// </summary>
        public MaxPool2D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Mirror)
        {
            if (padding < 0)
                throw new ArgumentException("Padding cannot be less than 0");
            if (kernel_size < 2)
                throw new ArgumentException("Kernel_size cannot be less than 2");

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
            if(padding > 0)
                input = Tensor.MatPad(input, padding, padding_mode);

            int batch_size = input.Rank == 4 ? input.Size(-4) : 1;
            int channel_size = input.Rank >= 3 ? input.Size(-3) : 1;

            Tensor pooled_input = Tensor.Zeros(batch_size, channel_size, H_out, W_out);


            if (batch_size == 1)
            {               
                // Foreach channel
                for (int c = 0; c < channel_size; c++)
                {
                    // foreach pool result
                    for (int j = 0; j < H_out; j++)
                    {
                        for (int i = 0; i < W_out; i++)
                        {
                            float max_in_pool = float.MinValue;
                            // foreach pool element in the pool
                            for (int kj = 0; kj < kernel_size; kj++)
                            {
                                for (int ki = 0; ki < kernel_size; ki++)
                                {
                                    try
                                    {
                                        float elem = input[0, c, j * kernel_size + kj, i * kernel_size + ki];
                                        if (elem > max_in_pool)
                                            max_in_pool = elem;
                                    }
                                    catch { }
                                }
                            }

                            pooled_input[0, c, j, i] = max_in_pool;
                        }
                    }
                }
                
            }
            else
            {
                Parallel.For(0, batch_size, b =>
                {
                    // Foreach channel
                    for (int c = 0; c < channel_size; c++)
                    {
                        // foreach pool result
                        for (int j = 0; j < H_out; j++)
                        {
                            for (int i = 0; i < W_out; i++)
                            {
                                float max_in_pool = float.MinValue;

                                // foreach pool element in the pool
                                for (int kj = 0; kj < kernel_size; kj++)
                                {
                                    for (int ki = 0; ki < kernel_size; ki++)
                                    {
                                        try
                                        {
                                            float elem = input[b, c, j * kernel_size + kj, i * kernel_size + ki];
                                            if (elem > max_in_pool)
                                                max_in_pool = elem;
                                        }
                                        catch { }
                                    }
                                }

                                pooled_input[b, c, j, i] = max_in_pool;
                            }
                        }
                    }

                });
            }
            if (input.Rank == 3)
                return pooled_input.Squeeze(-4);
            else return pooled_input;
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            return Predict(input);
        }
        public Tensor Backward(Tensor loss)
        {
            // How does backprop of maxpool2d works.
            // Consider Input = 2x2 mapped on Output = 1x1.
            // Example
            // Input = [[1.1,3.0],[2.3, 0.3]]
            // Output = [3.0]
            // Loss = [-0.7]
            // GradInput = [[0, -0.7],[0, 0]].
            // Find the max value in the input cache (on index [0,1]). On that place the gradient wrt input will get the loss vlaue. ALl others are 0.

            int Batch = loss.Rank == 4 ? loss.Size(-4) : 1;
            int Channels = loss.Rank >= 3 ? loss.Size(-3) : 1;
            int H_out = loss.Size(-2);
            int W_out = loss.Size(-1);
            int H_in = InputCache.Size(-2);
            int W_in = InputCache.Size(-1);

            Tensor gradInput = Tensor.Zeros(Batch, Channels, H_in, W_in);

            if(Batch == 1)
            {
                for (int c = 0; c < Channels; c++)
                {
                    for (int i = 0; i < W_out; i++)
                    {
                        for (int j = 0; j < H_out; j++)
                        {
                            int maxRowIndex = -1;
                            int maxColIndex = -1;
                            float maxValue = float.MinValue;

                            for (int pi = 0; pi < kernel_size; pi++)
                            {
                                for (int pj = 0; pj < kernel_size; pj++)
                                {
                                    int rowIndex = j * kernel_size + pj;
                                    int colIndex = i * kernel_size + pi;
                                    float value = InputCache[c, rowIndex, colIndex];

                                    if (value > maxValue)
                                    {
                                        maxValue = value;
                                        maxRowIndex = rowIndex;
                                        maxColIndex = colIndex;
                                    }
                                }
                            }

                            // Check if is inside the bounds, and not taken from padding
                            if (maxRowIndex >= 0 && maxColIndex >= 0 && maxRowIndex < H_in && maxColIndex < W_in)
                            {
                                gradInput[c, maxRowIndex, maxColIndex] += loss[c, j, i];
                            }
                        }
                    }
                }
                
            }
            else
            {
                Parallel.For(0, Batch, b =>
                {
                    for (int c = 0; c < Channels; c++)
                    {
                        for (int i = 0; i < W_out; i++)
                        {
                            for (int j = 0; j < H_out; j++)
                            {
                                int maxRowIndex = -1;
                                int maxColIndex = -1;
                                float maxValue = float.MinValue;

                                for (int pi = 0; pi < kernel_size; pi++)
                                {
                                    for (int pj = 0; pj < kernel_size; pj++)
                                    {
                                        int rowIndex = j * kernel_size + pj;
                                        int colIndex = i * kernel_size + pi;
                                        float value = InputCache[b, c, rowIndex, colIndex];

                                        if (value > maxValue)
                                        {
                                            maxValue = value;
                                            maxRowIndex = rowIndex;
                                            maxColIndex = colIndex;
                                        }
                                    }
                                }

                                // Check if is inside the bounds, and not taken from padding
                                if (maxRowIndex >= 0 && maxColIndex >= 0 && maxRowIndex < H_in && maxColIndex < W_in)
                                {
                                    gradInput[b, c, maxRowIndex, maxColIndex] += loss[b, c, j, i];
                                }
                            }
                        }
                    }
                });
            }


            return gradInput;
        }

        public object Clone() => new MaxPool2D(this.kernel_size, this.padding, this.padding_mode);

    }
}