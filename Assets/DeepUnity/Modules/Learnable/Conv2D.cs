using System.Threading.Tasks;
using System;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.Modules
{
    [Serializable]
    public class Conv2D : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        private Tensor InputCache { get; set; }
       
        private int GetOutChannels { get => kernels.Size(-4); }
        private int GetInChannels { get => kernels.Size(-3); }
        private int GetKernelHeight { get => kernels.Size(-2); }
        private int GetKernelWidth { get => kernels.Size(-1); }

        [SerializeField] private Tensor kernels;
        [SerializeField] private Tensor biases;
        [NonSerialized] public Tensor kernelsGrad;
        [NonSerialized] public Tensor biasesGrad;


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
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public Conv2D(int in_channels, int out_channels, int kernel_size, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = Device.CPU)
            : this(in_channels, out_channels, (kernel_size, kernel_size), weight_init, bias_init, device) { }

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
        public Conv2D(int in_channels, int out_channels,(int, int) kernel_shape, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = Device.CPU)
        {
            if (in_channels < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_shape.Item1 < 2 || kernel_shape.Item2 < 2) 
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.Device = device;

            int fanIn = in_channels * kernel_shape.Item1 * kernel_shape.Item2;
            int fanOut = out_channels;

            kernels = Parameter.Create(new int[] { out_channels, in_channels, kernel_shape.Item1, kernel_shape.Item2 }, fanIn, fanOut, weight_init);
            biases = Parameter.Create(new int[] { out_channels }, fanIn, fanOut, bias_init);

            kernelsGrad = Tensor.Zeros(kernels.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
        }


        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 3)
                throw new ShapeException($"The input ({input.Shape.ToCommaSeparatedString()}) in Conv2D module must be (B, C, H, W) or (C, H, W) for unbatched input.");

            if (input.Size(-3) != GetInChannels)
                throw new ShapeException($"Input shape ({input.Shape.ToCommaSeparatedString()}) received in Conv2D module must have {GetInChannels} channels.");

            int batch_size = input.Rank == 4 ? input.Size(-4) : 1;

            if (Device == Device.CPU)
            {
                int outputChannels = GetOutChannels;
                int inputChannels = GetInChannels;

                bool isBatched = input.Rank == 4 ? true : false;
                int batchSize = isBatched ? input.Size(-4) : 1;
             
                int inputHeight = input.Size(-2);
                int inputWidth = input.Size(-1);
                int kernelHeight = GetKernelHeight;
                int kernelWidth = GetKernelWidth;


                int outputHeight = inputHeight - kernelHeight + 1;
                int outputWidth = inputWidth - kernelWidth + 1;

                if(outputHeight < 1 || outputWidth < 1)
                {
                    throw new ShapeException($"Input received shape in Conv2D ({input.Shape.ToCommaSeparatedString()}) is smaller than the kernel {kernelHeight}x{kernelWidth}. Try using padding.");
                }

                Tensor output = input.Rank == 3 ?
                    Tensor.Zeros(outputChannels, outputHeight, outputWidth) :
                    Tensor.Zeros(batchSize, outputChannels, outputHeight, outputWidth);


                Parallel.For(0, batchSize, b =>
                {
                    Parallel.For(0, outputChannels, oc =>
                    {
                        for (int h = 0; h < outputHeight; h++)
                        {
                            for (int w = 0; w < outputWidth; w++)
                            {
                                float sum = biases[oc];

                                for (int ic = 0; ic < inputChannels; ic++)
                                {
                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            sum += input[b, ic, h + j, w + i] * kernels[oc, ic, j, i];
                                        }
                                    }
                                }

                                output[b, oc, h, w] = sum; // summation over input channels
                            }
                        }
                    });
                });


                return output;
            }
            else
            {
                int C_in = input.Size(-3);
                int H_in = input.Size(-2);
                int W_in = input.Size(-1);
                int C_out = GetOutChannels;
                int H_out = H_in - GetKernelHeight + 1;
                int W_out = W_in - GetKernelWidth + 1;

                ComputeShader cs = DeepUnityMeta.Conv2DCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer gammaBuffer = new ComputeBuffer(kernels.Count(), 4);
                gammaBuffer.SetData(kernels.ToArray());
                cs.SetBuffer(0, "gamma", gammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(biases.Count(), 4);
                betaBuffer.SetData(biases.ToArray());
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
                cs.SetInt("kernel_height", GetKernelHeight);
                cs.SetInt("kernel_width", GetKernelWidth);

                cs.Dispatch(0,
                    (W_out + 15) / 16,
                    (H_out + 15) / 16,
                    (C_out + 3) / 4);

                Tensor output = input.Rank == 3 ?
                        Tensor.Constant(outputBuffer, C_out, H_out, W_out) :
                        Tensor.Constant(outputBuffer, batch_size, C_out, H_out, W_out);

                inputBuffer.Release();
                gammaBuffer.Release();
                betaBuffer.Release(); 
                outputBuffer.Release();

                return output;
            }
        }

        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Forward(Tensor input)
        {
            InputCache = input.Clone() as Tensor;

            return Predict(input);
        }

        /// <param name="loss">(B, C_out, H - K_h + 1, W - K_w + 1)</param>
        /// <returns></returns>
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batchSize = isBatched ? loss.Size(-4) : 1;

            int kernelHeight = GetKernelHeight;
            int kernelWidth = GetKernelWidth;

            int inputChannels = InputCache.Size(-3);
            int inputHeight = InputCache.Size(-2);
            int inputWidth = InputCache.Size(-1);

            int outputChannels = kernels.Size(-4);
            int outputHeight = inputHeight - kernelHeight + 1;
            int outputWidth = inputWidth - kernelWidth + 1;

            float grad_scale = batchSize * inputChannels * outputChannels * kernelHeight * kernelWidth * inputHeight * inputWidth; ; // * outputWidth * outputHeight;
            
            // Bias grad
            Parallel.For(0, outputChannels, oc =>
            {
                float sum = 0f;

                for (int b = 0; b < batchSize; b++)
                {
                    for (int h = 0; h < outputHeight; h++)
                    {
                        for (int w = 0; w < outputWidth; w++)
                            sum += loss[b, oc, h, w];
                    }
                }

                biasesGrad[oc] += sum / grad_scale;
            });

         


            if (Device == Device.CPU)
            {
                Tensor inputGrad = isBatched ?
                        Tensor.Zeros(batchSize, inputChannels, inputHeight, inputWidth) :
                        Tensor.Zeros(inputChannels, inputHeight, inputWidth);

                // Compute the gradients of the weights - valid correlation(x,loss)      
                Parallel.For(0, outputChannels, oc =>
                {
                    Parallel.For(0, inputChannels, ic =>
                    {
                        for (int kh = 0; kh < kernelHeight; kh++)
                        {
                            for (int kw = 0; kw < kernelWidth; kw++)
                            {
                                float sum = 0f;

                                for (int b = 0; b < batchSize; b++)
                                {
                                    for (int j = 0; j < outputHeight; j++)
                                    {
                                        for (int i = 0; i < outputWidth; i++)
                                        {
                                            sum += InputCache[b, ic, j + kh, i + kw] * loss[b, oc, j, i];
                                        }
                                    }
                                }
                               
                                kernelsGrad[oc, ic, kh, kw] = sum / grad_scale;
                            }
                        }                                                 
                    });                                                                     
                });
                
                // Compute the gradient of the input - full convolution(loss, kernels) (no pad involved, just checkings)
                Parallel.For(0, batchSize, b =>
                {
                    Parallel.For(0, inputChannels, ic =>
                    {
                        for (int ih = 0; ih < inputHeight; ih++)
                        {
                            for (int iw = 0; iw < inputWidth; iw++)
                            {
                                float sum = 0f;

                                for (int oc = 0; oc < outputChannels; oc++)
                                {
                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            int oh = ih + kh - kernelHeight + 1;
                                            int ow = iw + kw - kernelWidth + 1;
                                            
                                            if (oh >= 0 && oh < outputHeight && ow >= 0 && ow < outputWidth)
                                            {
                                                sum += loss[b, oc, oh, ow] * kernels[oc, ic, kernelHeight - kh - 1, kernelWidth - kw - 1]; // kernel is rotated by 180d
                                            }
                                            // used when appling pad to the loss sum += loss[b, oc, ih + kh, iw + kw] * kernels[oc, ic, kernelHeight - kh - 1, kernelWidth - kw - 1]; // kernel is rotated by 180d
                                        }
                                    }
                                }
                                
                                inputGrad[b, ic, ih, iw] = sum / grad_scale;
                            }
                        }                    
                    });
                });

                return inputGrad;
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

                cs.SetInt("batch_size", batchSize);
                cs.SetInt("in_channels", C_in);
                cs.SetInt("in_height", H_in);
                cs.SetInt("in_width", W_in);
                cs.SetInt("out_channels", C_out);
                cs.SetInt("out_height", H_out);
                cs.SetInt("out_width", W_out);
                cs.SetInt("kernel_height", kernelHeight);
                cs.SetInt("kernel_width", kernelWidth);
                cs.SetFloat("grad_scale", grad_scale);


                // Compute the gradients of the loss w.r.t. weights ------------------------------------------------------------
                int KINDEX = kernelHeight <= 3 ? 1 : 2;

                ComputeBuffer lossBuffer = new ComputeBuffer(loss.Count(), 4);
                lossBuffer.SetData(loss.ToArray());
                cs.SetBuffer(KINDEX, "loss", lossBuffer);

                ComputeBuffer inputBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputBuffer.SetData(InputCache.ToArray());
                cs.SetBuffer(KINDEX, "input", inputBuffer);

                ComputeBuffer gammaGradBuffer = new ComputeBuffer(kernelsGrad.Count(), 4);
                gammaGradBuffer.SetData(kernelsGrad.ToArray());
                cs.SetBuffer(KINDEX, "gamma_grad", gammaGradBuffer);

                if (KINDEX == 1)
                    cs.Dispatch(1,
                        (kernels.Size(-1) + 2) / 3,
                        (kernels.Size(-2) + 2) / 3,
                        (C_out + 63) / 64);
                else
                    cs.Dispatch(2,
                        (kernels.Size(-1) + 4) / 5,
                        (kernels.Size(-2) + 4) / 5,
                        (C_out + 31) / 32);

                Tensor.CopyTo(kernelsGrad + Tensor.Constant(gammaGradBuffer, kernelsGrad.Shape), kernelsGrad);

                inputBuffer.Release();
                gammaGradBuffer.Release();
                // Compute the gradients of the loss wrt. parameters --------------------------------------------------------------------


                // ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


                // Computes the gradient of the loss wrt input. -------------------------------------------------------------------------
                cs.SetBuffer(3, "loss", lossBuffer);

                ComputeBuffer gammaBuffer = new ComputeBuffer(kernels.Count(), 4);
                gammaBuffer.SetData(kernels.ToArray());
                cs.SetBuffer(3, "gamma", gammaBuffer);

                ComputeBuffer inputGradBuffer = new ComputeBuffer(InputCache.Count(), 4);
                inputGradBuffer.SetData(new float[InputCache.Count()]);
                cs.SetBuffer(3, "input_grad", inputGradBuffer);

                cs.Dispatch(3,
                    (W_in + 15) / 16,
                    (H_in + 15) / 16,
                    (batchSize + 3) / 4);

                Tensor inputGrad = isBatched ?
                    Tensor.Constant(inputGradBuffer, batchSize, C_in, H_in, W_in):
                    Tensor.Constant(inputGradBuffer, C_in, H_in, W_in);

                lossBuffer.Release();
                gammaBuffer.Release();
                inputGradBuffer.Release();

                return inputGrad;
            }


            
        }


        public Parameter[] Parameters()
        {
            if (kernelsGrad == null)
                OnAfterDeserialize();

            var k = new Parameter(kernels, kernelsGrad);
            var b = new Parameter(biases, biasesGrad);

            return new Parameter[] { k, b };
        }
        public virtual void OnBeforeSerialize()
        {

        }
        public virtual void OnAfterDeserialize()
        {
            // This function is actually having 2 workers on serialization.
            // If shape int[] was not deserialized, we need to break this worker.
            // In case the shape wasn't already deserialized, we need to stop this worker and let the other instantiate everything.

            if (kernels.Shape == null)
                return;

            if (kernels.Shape.Length == 0)
                return;

            // do not check if gamma is != null...
            kernelsGrad = Tensor.Zeros(kernels.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);

        }
        public object Clone()
        {
            var conv = new Conv2D(GetInChannels, GetOutChannels, kernel_shape: (GetKernelHeight, GetKernelWidth), device: Device);
            conv.kernels = (Tensor)kernels.Clone();
            conv.biases = (Tensor)biases.Clone();
            conv.kernelsGrad = (Tensor)kernelsGrad.Clone();
            conv.biasesGrad = (Tensor)biasesGrad.Clone();

            return conv;
        }


    }


}


