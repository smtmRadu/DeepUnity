using System;
using System.Collections.Generic;
using System.Drawing.Printing;
using System.Linq;
using System.Runtime.Remoting.Channels;
using System.Threading.Tasks;
using Unity.VisualScripting;
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
        [SerializeField] private int kernelWidth;
        [SerializeField] private int kernelHeight;

        // Biases are applied over the final output. Biases (out_channels, out_height, out_width).
        // input shape  = (B, iC, H, W)
        // output_shape = (B, oC, H - K + 1, W - K + 1] 
        // In Conv2D, Gamma represents kernels, Beta represents biases


        /// <summary>
        /// Input: (<b>B</b>, <b>C_in</b>, <b>H_in</b>, <b>W_in</b>) or (<b>C_in</b>, <b>H_in</b>, <b>W_in</b>) for unbatched input.<br/>
        /// Output: <b>(B, C_out, H_out, W_out)</b> or <b>(C_out, H_out, W_out)</b> for unbatched input.<br></br>
        /// <br></br>
        /// where <br></br>
        /// B = batch_size <br></br>
        /// C_in = in_channels <br></br> 
        /// C_out = out_channels, <br></br>
        /// H_out = H_in - kernel_size + 1 <br></br> 
        /// W_out = W_in - kernel_size + 1
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="out_channels">C_out</param>
        /// <param name="kernel_size"></param>
        public Conv2D((int, int, int) input_shape, int out_channels, int kernel_size, Device device = Device.CPU) : base(device)
        {
            if (input_shape.Item1 < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if(kernel_size < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.inputShape = new int[] { input_shape.Item1, input_shape.Item2, input_shape.Item3 };
            this.outChannels = out_channels;
            this.kernelWidth = kernel_size;
            this.kernelHeight = kernel_size;

            
            int in_width = input_shape.Item3;
            int in_height = input_shape.Item2;
            int in_channels = input_shape.Item1;

            int out_width = in_width - kernelWidth + 1;
            int out_height = in_height - kernelWidth + 1;


            float k = 1f / (in_channels * kernelHeight * kernelWidth);
            (float, float) range = (-MathF.Sqrt(k), MathF.Sqrt(k));

            gamma = Tensor.RandomRange(range, out_channels, in_channels, kernelHeight, kernelWidth);
            beta = Tensor.RandomRange(range, outChannels, out_height, out_width);
            gammaGrad = Tensor.Zeros(out_channels, in_channels, kernelHeight, kernelWidth);
            betaGrad = Tensor.Zeros(outChannels, out_height, out_width);
        }
        /// <summary>
        /// Input: (<b>B</b>, <b>C_in</b>, <b>H_in</b>, <b>W_in</b>) or (<b>C_in</b>, <b>H_in</b>, <b>W_in</b>) for unbatched input.<br/>
        /// Output: <b>(B, C_out, H_out, W_out)</b> or <b>(C_out, H_out, W_out)</b> for unbatched input.<br></br>
        /// <br></br>
        /// where <br></br>
        /// B = batch_size <br></br>
        /// C_in = in_channels <br></br> 
        /// C_out = out_channels, <br></br>
        /// H_out = H_in - kernel_shape.Item2 + 1 <br></br> 
        /// W_out = W_in - kernel_shape.Item1 + 1
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        public Conv2D((int, int, int) input_shape, int out_channels, (int, int) kernel_shape, Device device = Device.CPU) : base(device)
        {
            if (input_shape.Item1 < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_shape.Item1 < 2 || kernel_shape.Item2 < 2)
                throw new ArgumentException("Kernel cannot have a dimension < 2.");

            this.inputShape = new int[] { input_shape.Item1, input_shape.Item2, input_shape.Item3 };
            this.outChannels = out_channels;
            this.kernelWidth = kernel_shape.Item2;
            this.kernelHeight = kernel_shape.Item1;


            int in_width = input_shape.Item3;
            int in_height = input_shape.Item2;
            int in_channels = input_shape.Item1;

            int out_width = in_width - kernelWidth + 1;
            int out_height = in_height - kernelWidth + 1;

            float k = 1f / (in_channels * kernelHeight * kernelWidth);
            (float, float) range = (-MathF.Sqrt(k), MathF.Sqrt(k));
            gamma = Tensor.RandomRange(range, out_channels, in_channels, kernelHeight, kernelWidth);
            beta = Tensor.RandomRange(range, outChannels, out_height, out_width);
            gammaGrad = Tensor.Zeros(out_channels, in_channels, kernelHeight, kernelWidth);
            betaGrad = Tensor.Zeros(outChannels, out_height, out_width);
        }

        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Predict(Tensor input)
        {
            if(input.Size(-3) != inputShape[0] || input.Size(-2) != inputShape[1] || input.Size(-1) != inputShape[2])
                throw new ShapeException($"Input shape ({input.Size(-3)}{input.Size(-2)}{input.Size(-1)}) received in Conv2D module must be ({inputShape.ToCommaSeparatedString()})");

            int batch_size = input.Rank == 4 ? input.Size(-4) : 1;

            if(device == Device.CPU)
            {
                if(batch_size == 1)
                {
                    return Correlate2DValid_input_kernels(input, gamma).Squeeze(-4) + beta;
                }
                else
                {
                    return Correlate2DValid_input_kernels(input, gamma) + Tensor.Expand(Tensor.Unsqueeze(beta, 0), 0, batch_size);
                }
            }
            else
            {
                int C_in = input.Size(-3);
                int H_in = input.Size(-2);
                int W_in = input.Size(-1);
                int C_out = outChannels;
                int H_out = H_in - kernelHeight + 1;
                int W_out = W_in - kernelWidth + 1;

                ComputeShader cs = DeepUnityMeta.Conv2DCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer gammaBuffer = new ComputeBuffer(gamma.Count(), 4);
                gammaBuffer.SetData(gamma.ToArray());
                cs.SetBuffer(0, "gamma", gammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(beta.Count(), 4);  
                betaBuffer.SetData(beta.ToArray());
                cs.SetBuffer(0, "beta", betaBuffer);

                ComputeBuffer outputBuffer = new ComputeBuffer(batch_size * C_out * H_out * W_out, 4);
                outputBuffer.SetData(new float[batch_size * C_out * H_out * W_out]);
                cs.SetBuffer(0, "output", outputBuffer);

                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_channels", C_in);
                cs.SetInt("in_height", H_in);
                cs.SetInt("in_width", W_in);
                cs.SetInt("out_channels", C_out);             
                cs.SetInt("out_height", H_out);           
                cs.SetInt("out_width", W_out);
                cs.SetInt("kernel_height", kernelHeight);
                cs.SetInt("kernel_width", kernelWidth);

                cs.Dispatch(0,
                    (W_out + 16 - 1) / 16,
                    (H_out + 16 - 1) / 16,
                    (C_out + 4 - 1) / 4);

                Tensor result = Tensor.Constant(outputBuffer).Reshape(batch_size, outChannels, H_out, W_out).Squeeze(-4);

                inputBuffer.Release();
                gammaBuffer.Release();
                betaBuffer.Release();
                outputBuffer.Release();

                return result;
            }
        }

        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

            return Predict(input);
        }

        /// <param name="loss">(B, C_out, H - K + 1, W - K + 1)</param>
        /// <returns></returns>
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batch_size = isBatched ? loss.Size(-4) : 1;

            if(device == Device.CPU)
            {
                gammaGrad += Correlate2DValid_input_loss(InputCache, loss) / batch_size;
                betaGrad += isBatched ? Tensor.Mean(loss, -4) : loss;
                return Convolve2DFull_loss_gamma(loss, gamma).Squeeze(-4);
            }
            else 
            {
                ComputeShader cs = DeepUnityMeta.Conv2DCS;

                int C_in = InputCache.Size(-3);
                int H_in = InputCache.Size(-2);
                int W_in = InputCache.Size(-1);
                int C_out = loss.Size(-3);
                int H_out = loss.Size(-2);
                int W_out = loss.Size(-1);
              
                cs.SetInt("batch_size", batch_size);
                cs.SetInt("in_channels", C_in);
                cs.SetInt("in_height", H_in);
                cs.SetInt("in_width", W_in);
                cs.SetInt("out_channels", C_out);
                cs.SetInt("out_height", H_out);
                cs.SetInt("out_width", W_out);
                cs.SetInt("kernel_height", kernelHeight);
                cs.SetInt("kernel_width", kernelWidth);


                // Compute the gradients of the loss wrt. parameters ------------------------------------------------------------OK SO COMPUTE SHADER WORKS BUT IT S AN OUTER PROBLEM
                ComputeBuffer lossBuffer = new ComputeBuffer(loss.Count(), 4);
                lossBuffer.SetData(loss.ToArray());
                cs.SetBuffer(1, "loss", lossBuffer);
                
                ComputeBuffer inputBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputBuffer.SetData(InputCache.ToArray());
                cs.SetBuffer(1, "input", inputBuffer);
                
                ComputeBuffer gammaGradBuffer = new ComputeBuffer(gammaGrad.Count(), 4);
                gammaGradBuffer.SetData(gammaGrad.ToArray());
                cs.SetBuffer(1, "gamma_grad", gammaGradBuffer);
                
                cs.Dispatch(1,
                    (gamma.Size(-1) + 5 - 1) / 5,
                    (gamma.Size(-2) + 5 - 1) / 5,
                    (C_out + 32 - 1) / 32);

                gammaGrad = Tensor.Constant(gammaGradBuffer).Reshape(gammaGrad.Shape);
                betaGrad += isBatched ? Tensor.Mean(loss, -4) : loss;  // faster on CPU

                inputBuffer.Release();
                gammaGradBuffer.Release();
                // Compute the gradients of the loss wrt. parameters --------------------------------------------------------------------


                // ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


                // Computes the gradient of the loss wrt input. -------------------------------------------------------------------------
                cs.SetBuffer(2, "loss", lossBuffer);

                ComputeBuffer gammaBuffer = new ComputeBuffer(gamma.Count(), 4);
                gammaBuffer.SetData(gamma.ToArray());
                cs.SetBuffer(2, "gamma", gammaBuffer);

                ComputeBuffer inputGradBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputGradBuffer.SetData(new float[InputCache.Count()]);
                cs.SetBuffer(2, "input_grad", inputGradBuffer);

                cs.Dispatch(2,
                    (W_in + 16 - 1) / 16,
                    (H_in + 16 - 1) / 16,
                    (batch_size + 4 - 1) / 4);

                Tensor inputGrad = Tensor.Constant(inputGradBuffer).Reshape(batch_size, C_in, H_in, W_in).Squeeze(-4);

                lossBuffer.Release();
                gammaBuffer.Release();
                inputGradBuffer.Release();

                return inputGrad;
                // Computes the gradient of the loss wrt input. -------------------------------------------------------------------------
            }
        }



        // These methods were made for more efficient correlation/covolution than Tensor class provided methods.

        /// <summary>
        /// Performs the 2d correlation required for input with kernels
        /// </summary>
        /// <param name="input">(B, iC, H, W)</param>
        /// <param name="kernels">(oC, iC, K, K)</param>
        /// <param name="correlationType"></param>
        /// <returns>(B, oC, H*, W*)</returns>
        private static Tensor Correlate2DValid_input_kernels(Tensor input, Tensor kernels)
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

                                     output[b, oc, h, w] += sum;
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

                                 output[0, oc, h, w] += sum;
                             }
                         }
                     }
                 }
            


            return output;
        }

        /// <summary>
        ///  Performs the 2d correlation required for input with loss
        /// </summary>
        /// <param name="input">(B, C_in, H, W)</param>
        /// <param name="loss">(B, C_out, H*, W*)</param>
        /// <param name="mode"></param>
        /// <returns>kernel_gradient(C_out, C_in, K, K)</returns>
        private Tensor Correlate2DValid_input_loss(Tensor input, Tensor loss)
        {
            int batchSize = input.Rank == 4 ? input.Size(0) : 1;
            int inChannels = inputShape[0];
            Tensor kernGrad = Tensor.Zeros(outChannels, inChannels, kernelHeight, kernelWidth);

            
            int lossHeight = loss.Size(-2);
            int lossWidth = loss.Size(-1);   

            // correlation type = valid
            Parallel.For(0, outChannels, oc =>
            {
                for (int b = 0; b < batchSize; b++)
                {
                    for (int ic = 0; ic < inChannels; ic++)
                    {
                        for (int h = 0; h < kernelHeight; h++)
                        {
                            for (int w = 0; w < kernelWidth; w++)
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
        /// Performs the 2d convolution required for loss with gamma to obtain loss grad wrt input
        /// </summary>
        /// <param name="loss">(B, oC, H*, W*)</param>
        /// <param name="kernels">(oC, iC, K, K)</param>
        /// <returns>input_gradient(B, iC, H, W)</returns>
        private Tensor Convolve2DFull_loss_gamma(Tensor loss, Tensor kernels)
        {
            

            int batchSize = loss.Rank == 4 ? loss.Size(-4) : 1;

            int inChannels = inputShape[0];
            int inputGradHeight = loss.Size(-2) + kernelHeight - 1;
            int inputGradWidth = loss.Size(-1) + kernelWidth - 1;

            int lossHeight = loss.Size(-2);
            int lossWidth = loss.Size(-1);
            /// convolution type == full
            Tensor inputGrad = Tensor.Zeros(batchSize, inChannels, inputGradHeight, inputGradWidth);

            if(batchSize > 1)
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
                
                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            int inputRow = h - j;
                                            int inputCol = w - i;
                
                                            if (inputRow >= 0 && inputRow < lossHeight && inputCol >= 0 && inputCol < lossWidth)
                                            {
                                                // the kernels are rotated by 180 degrees, so we get the inversed index of the kernel.
                                                int jIndexOfRotatedKernel = kernelHeight - j - 1;
                                                int iIndexOfRotatedKernel = kernelWidth - i - 1;
                                                sum += loss[b, oc, inputRow, inputCol] * kernels[oc, ic, jIndexOfRotatedKernel, iIndexOfRotatedKernel];
                                            }
                                        }
                                    }
                                    inputGrad[b, ic, h, w] += sum;
                                }
                            }
                        }
                    }
                });
            else
                for (int ic = 0; ic < inChannels; ic++)
                {
                    for (int oc = 0; oc < outChannels; oc++)
                    {
                        for (int h = 0; h < inputGradHeight; h++)
                        {
                            for (int w = 0; w < inputGradWidth; w++)
                            {
                                float sum = 0f;

                                for (int j = 0; j < kernelHeight; j++)
                                {
                                    for (int i = 0; i < kernelWidth; i++)
                                    {
                                        int inputRow = h - j;
                                        int inputCol = w - i;

                                        if (inputRow >= 0 && inputRow < lossHeight && inputCol >= 0 && inputCol < lossWidth)
                                        {
                                            // the kernels are rotated by 180 degrees, so we get the inversed index of the kernel.
                                            int jIndexOfRotatedKernel = kernelHeight - j - 1;
                                            int iIndexOfRotatedKernel = kernelWidth - i - 1;
                                            sum += loss[0, oc, inputRow, inputCol] * kernels[oc, ic, jIndexOfRotatedKernel, iIndexOfRotatedKernel];
                                        }
                                    }
                                }
                                inputGrad[0, ic, h, w] += sum;
                            }
                        }
                    }
                }

            return inputGrad;
        }
    }
}

