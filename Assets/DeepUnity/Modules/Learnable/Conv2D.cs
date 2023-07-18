using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.Remoting.Channels;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.UIElements;
using UnityEngine.Windows;

namespace DeepUnity
{
    // https://www.youtube.com/watch?v=Lakz2MoHy6o


    [Serializable]
    public class Conv2D : Learnable, IModule 
    {
        
        private Tensor InputCache { get; set; }

        [SerializeField] private int[] inputShape;
        [SerializeField] private int outChannels;
        [SerializeField] private int kernelSize;

        // Biases are applied over the final output. Biases (out_channels, out_height, out_width).
        // input shape  = (B, iC, H, W)
        // output_shape = (B, oC, H - K + 1, W - K + 1] 
        // In Conv2D, Gamma represents kernels, Beta represents biases


        /// <summary>
        /// input: (<b>batch</b>, <b>in_channels</b>, <b>height</b>, <b>width</b>) <br/>
        /// output: (<b>batch</b>, <b>out_channels</b>, <b>height - kernel_size + 1</b>, <b>width - kernel_size + 1</b>)
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="out_channels">C_out</param>
        /// <param name="kernel_size"></param>
        public Conv2D((int, int, int) input_shape, int out_channels, int kernel_size) : base(Device.CPU)
        {
            if (input_shape.Item1 < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if(kernel_size < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.inputShape = new int[] { input_shape.Item1, input_shape.Item2, input_shape.Item3 };
            this.outChannels = out_channels;
            this.kernelSize = kernel_size;


            
            int in_width = input_shape.Item3;
            int in_height = input_shape.Item2;
            int in_channels = input_shape.Item1;

            int out_width = in_width - kernelSize + 1;
            int out_height = in_height - kernelSize + 1;

            gamma = Tensor.RandomNormal((0, 1), out_channels, in_channels, kernelSize, kernelSize);
            gammaGrad = Tensor.Zeros(out_channels, in_channels, kernelSize, kernelSize);

            beta = Tensor.RandomNormal((0, 1), outChannels, out_height, out_width);
            betaGrad = Tensor.Zeros(outChannels, out_height, out_width);
        }

        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Predict(Tensor input)
        {
            bool isBatched = input.Rank == 4;
            int batch_size = isBatched ? gamma.Size(-4) : 1;
            return Correlate2D_input_kernels(input, gamma, CorrelationMode.Valid) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
        }

        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            bool isBatched = input.Rank == 4;
            int batch_size = isBatched ? input.Size(-4) : 1;
            return Correlate2D_input_kernels(input, gamma, CorrelationMode.Valid) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
        }

        /// <param name="loss">(B, C_out, H - K + 1, W - K + 1)</param>
        /// <returns></returns>
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batch_size = isBatched ? loss.Size(-4) : 1;
            gammaGrad += Correlate2DValid_input_loss(InputCache, loss) / batch_size;
            betaGrad += isBatched ? Tensor.Mean(loss,0) : loss;

            return Convolve2DFull_loss_gamma(loss, gamma);
        }



        // These methods were made for more efficient correlation/covolution than Tensor class provided methods.

        /// <summary>
        /// General template.
        /// Performs the 2d correlation required for input with kernels
        /// </summary>
        /// <param name="input">(B, iC, H, W)</param>
        /// <param name="kernels">(oC, iC, K, K)</param>
        /// <param name="correlationType"></param>
        /// <returns>(B, oC, H*, W*)</returns>
        private static Tensor Correlate2D_input_kernels(Tensor input, Tensor kernels, CorrelationMode correlationType = CorrelationMode.Valid)
        {
            Tensor output = null;

            // Output shape : [batch, kern.batch, *W, *H] 

            int outputChannels = kernels.Size(-4);
            int inputChannels = kernels.Size(-3);

            int batchSize = input.Rank == 4 ? input.Size(-4) : 1;
            int inputHeight = input.Size(-2);
            int inputWidth = input.Size(-1);
            int kernelHeight = kernels.Size(-2);
            int kernelWidth = kernels.Size(-1);

            if (correlationType == CorrelationMode.Valid)
            {
                int outputHeight = inputHeight - kernelHeight + 1;
                int outputWidth = inputWidth - kernelWidth + 1;


                output = Tensor.Zeros(batchSize, outputChannels, outputHeight, outputWidth);

                if(batchSize > 1)
                    Parallel.For(0, batchSize, b =>
                    {
                        for (int oc = 0; oc < outputChannels; oc++)
                        {
                            for (int ic = 0; ic < inputChannels; ic++)
                            {
                                for (int h = 0; h < outputHeight; h++)
                                {
                                    for (int w = 0; w < outputWidth; w++)
                                    {
                                        float sum = 0f;

                                        for (int j = 0; j < kernelHeight; j++)
                                        {
                                            for (int i = 0; i < kernelWidth; i++)
                                            {
                                                sum += input[b, ic, h + j, w + i] * kernels[oc, ic, j, i];
                                            }
                                        }

                                        output[b, oc, h, w] = sum;
                                    }
                                }
                            }
                        }
                    });
                else
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int h = 0; h < outputHeight; h++)
                            {
                                for (int w = 0; w < outputWidth; w++)
                                {
                                    float sum = 0f;

                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            sum += input[0, ic, h + j, w + i] * kernels[oc, ic, j, i];
                                        }
                                    }

                                    output[0, oc, h, w] = sum;
                                }
                            }
                        }
                    }
            }
            else if (correlationType == CorrelationMode.Full)
            {
                int outputHeight = inputHeight + kernelHeight - 1;
                int outputWidth = inputWidth + kernelWidth - 1;

                output = Tensor.Zeros(batchSize, outputChannels, outputHeight, outputWidth);

                Parallel.For(0, batchSize, b =>
                {
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int h = 0; h < outputHeight; h++)
                            {
                                for (int w = 0; w < outputWidth; w++)
                                {
                                    float sum = 0f;

                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            int inputRow = h - j;
                                            int inputCol = w - i;

                                            if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth)
                                            {
                                                sum += input[b, ic, inputRow, inputCol] * kernels[oc, ic, j, i];
                                            }
                                        }
                                    }

                                    output[b, oc, h, w] = sum;
                                }
                            }
                        }
                    }
                });

            }
            else if (correlationType == CorrelationMode.Same)
            {
                int outputHeight = inputHeight;
                int outputWidth = inputWidth;

                int paddingHeight = (kernelHeight - 1) / 2;
                int paddingWidth = (kernelWidth - 1) / 2;

                output = Tensor.Zeros(batchSize, outputChannels, outputHeight, outputWidth);

                Parallel.For(0, batchSize, b =>
                {
                    for (int oc = 0; oc < outputChannels; oc++)
                    {
                        for (int ic = 0; ic < inputChannels; ic++)
                        {
                            for (int h = 0; h < outputHeight; h++)
                            {
                                for (int w = 0; w < outputWidth; w++)
                                {
                                    float sum = 0f;

                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            int inputRow = h + j - paddingHeight;
                                            int inputCol = w + i - paddingWidth;

                                            if (inputRow >= 0 && inputRow < inputHeight && inputCol >= 0 && inputCol < inputWidth)
                                            {
                                                sum += input[b, ic, inputRow, inputCol] * kernels[oc, ic, j, i];
                                            }
                                        }
                                    }

                                    output[b, oc, h, w] = sum;
                                }
                            }
                        }
                    }
                });
            }

            return output;
        }

        /// <summary>
        ///  Performs the 2d correlation required for input with loss
        /// </summary>
        /// <param name="input">(B, iC, H, W)</param>
        /// <param name="loss">(B, oC, H*, W*)</param>
        /// <param name="mode"></param>
        /// <returns>kernel_gradient(oC, iC, K, K)</returns>
        private Tensor Correlate2DValid_input_loss(Tensor input, Tensor loss)
        {
            int batchSize = input.Rank == 4 ? input.Size(0) : 1;
            int inChannels = inputShape[0];
            Tensor kernGrad = Tensor.Zeros(outChannels, inChannels, kernelSize, kernelSize);

            
            int lossHeight = loss.Size(-2);
            int lossWidth = loss.Size(-1);   

            // correlation type = valid
            Parallel.For(0, batchSize, b =>
            {
                for (int oc = 0; oc < outChannels; oc++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int h = 0; h < kernelSize; h++)
                        {
                            for (int w = 0; w < kernelSize; w++)
                            {
                                float sum = 0f;

                                for (int j = 0; j < lossHeight; j++)
                                {
                                    for (int i = 0; i < lossWidth; i++)
                                    {
                                        sum += input[b, ic, h + j, w + i] * loss[b, oc, j, i];
                                    }
                                }

                                kernGrad[oc, ic, h, w] += sum;
                            }
                        }
                    }
                }
            }); 

            return kernGrad;
        }

        /// <summary>
        /// Performs the 2d correlation required for loss with gamma
        /// </summary>
        /// <param name="loss">(B, oC, H*, W*)</param>
        /// <param name="kernels">(oC, iC, K, K)</param>
        /// <returns>input_gradient(B, iC, H, W)</returns>
        private Tensor Convolve2DFull_loss_gamma(Tensor loss, Tensor kernels)
        {
            

            int batchSize = loss.Rank == 4 ? loss.Size(-4) : 1;

            int inChannels = inputShape[0];
            int inputGradHeight = loss.Size(-2) + kernelSize - 1;
            int inputGradWidth = loss.Size(-1) + kernelSize - 1;

            int lossHeight = loss.Size(-2);
            int lossWidth = loss.Size(-1);
            /// convolution type == full
            Tensor inputGrad = Tensor.Zeros(batchSize, inChannels, inputGradHeight, inputGradWidth);


            // Rotate the kernels by 180d -------------------------do-not-requires-parralelism-(tested)----------------------
            Tensor rot180dKernels = Tensor.Zeros(kernels.Shape);
            for (int oc = 0; oc < outChannels; oc++)
            {
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int h = 0; h < kernelSize; h++)
                    {
                        for (int w = 0; w < kernelSize; w++)
                        {
                            rot180dKernels[oc, ic, kernelSize - h - 1, kernelSize - w - 1] = kernels[oc, ic, h, w];
                        }
                    }
                }
            }
            kernels = rot180dKernels;
            // ------------------------------------------------------------------------------------



            Parallel.For(0, batchSize, b =>
            {
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int h = 0; h < inputGradHeight; h++)
                        {
                            for (int w = 0; w < inputGradWidth; w++)
                            {
                                float sum = 0f;

                                for (int j = 0; j < kernelSize; j++)
                                {
                                    for (int i = 0; i < kernelSize; i++)
                                    {
                                        int inputRow = h - j;
                                        int inputCol = w - i;

                                        if (inputRow >= 0 && inputRow < lossHeight && inputCol >= 0 && inputCol < lossWidth)
                                        {
                                            sum += loss[b, oc, inputRow, inputCol] * kernels[oc, ic, j, i];
                                        }
                                    }
                                }
                                inputGrad[b, ic, h, w] = sum;
                            }
                        }
                    }
                }
            });

            return inputGrad;
        }
    }
}

