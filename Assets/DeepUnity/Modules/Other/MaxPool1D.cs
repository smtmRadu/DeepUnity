using System;
using Unity.VisualScripting;
using UnityEngine;
using System.Threading.Tasks;


namespace DeepUnity.Modules
{
    [SerializeField]
    public class MaxPool1D : IModule
    {
        private Tensor InputCache { get; set; }

        [SerializeField] private int kernelSize;
        [SerializeField] private int padding;
        [SerializeField] private PaddingType paddingMode;


        public MaxPool1D(int kernel_size, int padding = 0, PaddingType padding_mode = PaddingType.Zeros)
        {
            if (padding < 0)
                throw new ArgumentException("Padding cannot be less than 0");
            if (kernel_size < 2)
                throw new ArgumentException("Kernel_size cannot be less than 2");

            this.kernelSize = kernel_size;
            this.padding = padding;
            this.paddingMode = padding_mode;
        }

        public Tensor Predict(Tensor input)
        {
            if (input.Rank != 2 && input.Rank != 3)
                throw new ShapeException($"Input({input.Shape.ToCommaSeparatedString()}) must either be (B, C, H) or (C, H).");


            int H_in = input.Size(-1);
            int H_out = (int)Math.Floor((H_in + 2 * padding - kernelSize - 1)/(float)kernelSize);

            if (padding > 0)
                input = Tensor.MatPad(input, padding, paddingMode);

            bool isBatched = input.Rank == 3;
            int batch_size = isBatched ? input.Size(-3) : 1;
            int channel_size = input.Rank >= 1 ? input.Size(-2) : 1;



            Tensor output = isBatched?
                Tensor.Zeros(batch_size, channel_size, H_out):
                Tensor.Zeros(channel_size, H_out);

            Parallel.For(0, batch_size, b =>
            {
                Parallel.For(0, channel_size, c =>
                {
                    for (int j = 0; j < H_out; j++)
                    {

                        float max_in_pool = float.MinValue;

                        // foreach pool element in the pool

                        for (int ki = 0; ki < kernelSize; ki++)
                        {
                            try
                            {
                                float elem = input[b, c, j * kernelSize + ki];
                                if (elem > max_in_pool)
                                    max_in_pool = elem;
                            }
                            catch { }
                        }


                        output[b, c, j] = max_in_pool;
                    }
                });
            });

            return output;
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

            bool isBatched = loss.Rank == 4;
            int Batch = isBatched ? loss.Size(-3) : 1;
            int Channels = loss.Rank >= 3 ? loss.Size(-2) : 1;
            int H_out = loss.Size(-1);
            int H_in = InputCache.Size(-1);

            Tensor gradInput = isBatched ?
                Tensor.Zeros(Batch, Channels, H_in):
                Tensor.Zeros(Channels, H_in);


            Parallel.For(0, Batch, b =>
            {
                Parallel.For(0, Channels, c =>
                {
                    for (int j = 0; j < H_out; j++)
                    {
                        int maxRowIndex = -1;
                        float maxValue = float.MinValue;

                        for (int pi = 0; pi < kernelSize; pi++)
                        {
                            int rowIndex = j * kernelSize + pi;
                            float value = InputCache[b, c, rowIndex];

                            if (value > maxValue)
                            {
                                maxValue = value;
                                maxRowIndex = rowIndex;
                            }
                        }

                        // Check if is inside the bounds, and not taken from padding
                        if (maxRowIndex >= 0 && maxRowIndex < H_in)
                        {
                            gradInput[b, c, maxRowIndex] += loss[b, c, j];
                        }
                    }
                });
            });



            return gradInput;
        }



        public object Clone() => new MaxPool1D(kernelSize, padding, paddingMode);

    }


}

