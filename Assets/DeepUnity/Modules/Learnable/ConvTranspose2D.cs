using DeepUnity.Modules;
using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

/// Conv
/// forward -> valid_correlation (input, weights)
/// backward -> w_grad -> valid_correlation (input, loss)
/// backward -> x_grad -> full_convolution (loss, weights) (weights get rotated)

/// ConvTranspose
/// forward -> full_convolution (input, weights)  (weights get rotated)
/// backward -> w_grad -> valid_correlation (loss, input)
/// backward -> x_grad -> valid_correlation (loss, weights) 


namespace DeepUnity.Layers
{
    // https://www.youtube.com/watch?v=Lakz2MoHy6o
    // https://github.com/TheIndependentCode/Neural-Network/blob/master/convolutional.py
    /// <summary>
    /// Input: (<b>B</b>, <b>C_in</b>, <b>H_in</b>, <b>W_in</b>) or (<b>C_in</b>, <b>H_in</b>, <b>W_in</b>) for unbatched input.<br/>
    /// Output: <b>(B, C_out, H_out, W_out)</b> or <b>(C_out, H_out, W_out)</b> for unbatched input.<br></br>
    /// <br></br>
    /// where <br></br>
    /// B = batch_size <br></br>
    /// C_in = in_channels <br></br> 
    /// C_out = out_channels, <br></br>
    /// H_out = H_in + kernel_height - 1<br></br> 
    /// W_out = W_in + kernel_width - 1 
    /// </summary>
    [Serializable]
    public class ConvTranspose2D : ILearnable, IModule
    {
        public Device Device { get; set; } = Device.CPU;
        public bool RequiresGrad { get; set; } = true;
        private Tensor InputCache { get; set; }
        private int GetOutChannels { get => kernels.Size(-3); }
        private int GetInChannels { get => kernels.Size(-4); }
        private int GetKernelHeight { get => kernels.Size(-2); }
        private int GetKernelWidth { get => kernels.Size(-1); }

        [SerializeField] private bool bias;
        [SerializeField] public Tensor kernels;
        [SerializeField] public Tensor biases;
        [NonSerialized] private Tensor kernelsGrad;
        [NonSerialized] private Tensor biasesGrad;

        /// <summary>
        /// Input: (<b>B</b>, <b>C_in</b>, <b>H_in</b>, <b>W_in</b>) or (<b>C_in</b>, <b>H_in</b>, <b>W_in</b>) for unbatched input.<br/>
        /// Output: <b>(B, C_out, H_out, W_out)</b> or <b>(C_out, H_out, W_out)</b> for unbatched input.<br></br>
        /// <br></br>
        /// where <br></br>
        /// B = batch_size <br></br>
        /// C_in = in_channels <br></br> 
        /// C_out = out_channels, <br></br>
        /// H_out = H_in + kernel_height - 1<br></br> 
        /// W_out = W_in + kernel_width - 1 
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="out_channels">C_out</param>
        /// <param name="kernel_size"></param>
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public ConvTranspose2D(int in_channels, int out_channels, (int,int) kernel_shape, bool bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = Device.CPU)
        {
            if (in_channels < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_shape.Item1 < 2 || kernel_shape.Item2 < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.Device = device;
            this.bias = bias;
            int fanIn = in_channels * kernel_shape.Item1 * kernel_shape.Item2;
            int fanOut = out_channels;
            kernels = Parameter.Create(new int[] { in_channels, out_channels, kernel_shape.Item1, kernel_shape.Item2 }, fanIn, fanOut, weight_init);
            // Reduce size (groups = 1), see pytorch initmode https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html
            kernels /= 3f;   
            kernelsGrad = Tensor.Zeros(kernels.Shape);

            if(bias)
            {
                biases = Parameter.Create(new int[] { out_channels }, fanIn, fanOut, bias_init);
                biases /= 3f;
                biasesGrad = Tensor.Zeros(biases.Shape);
            }
        
        }
        /// <summary>
        /// Input: (<b>B</b>, <b>C_in</b>, <b>H_in</b>, <b>W_in</b>) or (<b>C_in</b>, <b>H_in</b>, <b>W_in</b>) for unbatched input.<br/>
        /// Output: <b>(B, C_out, H_out, W_out)</b> or <b>(C_out, H_out, W_out)</b> for unbatched input.<br></br>
        /// <br></br>
        /// where <br></br>
        /// B = batch_size <br></br>
        /// C_in = in_channels <br></br> 
        /// C_out = out_channels, <br></br>
        /// H_out = H_in + kernel_height - 1<br></br> 
        /// W_out = W_in + kernel_width - 1 
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public ConvTranspose2D(int in_channels, int out_channels, int kernel_size, bool bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = Device.CPU)
         : this(in_channels, out_channels, (kernel_size, kernel_size), bias, weight_init, bias_init, device) { }

        private ConvTranspose2D() { }


        /// <param name="input">(B, C_in, H, W)</param>
        /// <returns></returns>
        public Tensor Predict(Tensor input)
        {
            if (input.Rank < 3)
                throw new ShapeException($"The input ({input.Shape.ToCommaSeparatedString()}) in ConvTranspose2D module must be (B, C, H, W) or (C, H, W) for unbatched input.");

            if (input.Size(-3) != GetInChannels)
                throw new ShapeException($"Input shape ({input.Shape.ToCommaSeparatedString()}) received in Conv2D module must have {GetInChannels} channels.");

            int batch_size = input.Rank == 4 ? input.Size(-4) : 1;

            if(Device == Device.CPU)
            {
                int inputChannels = GetInChannels;
                int inputHeight = input.Size(-2);
                int inputWidth = input.Size(-1);
                int outputChannels = GetOutChannels;
                int kernelHeight = GetKernelHeight;
                int kernelWidth = GetKernelWidth;

                int outputHeight = inputHeight + kernelHeight - 1;
                int outputWidth =  inputWidth  + kernelWidth  - 1;

                // Initialize output tensor
                Tensor output = input.Rank == 3 ?
                    Tensor.Zeros(outputChannels, outputHeight, outputWidth) :
                    Tensor.Zeros(batch_size, outputChannels, outputHeight, outputWidth);

                // Perform transposed convolution - full convolution(x, weights)
                Parallel.For(0, batch_size, b =>
                {
                    Parallel.For(0, outputChannels, oc =>
                    {
                        for (int oh = 0; oh < outputHeight; oh++)
                        {
                            for (int ow = 0; ow < outputWidth; ow++)
                            {
                                float sum = bias ? biases[oc] : 0f;

                                for (int ic = 0; ic < inputChannels; ic++)
                                {
                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            int ih = oh + kh - kernelHeight + 1;
                                            int iw = ow + kw - kernelWidth + 1;

                                            if (ih >= 0 && ih < inputHeight && iw >= 0 && iw < inputWidth)
                                            {
                                                sum += input[b, ic, ih, iw] * kernels[ic, oc, kernelHeight - kh - 1, kernelWidth - kw - 1];
                                            }
                                        }
                                    }
                                }

                                output[b, oc, oh, ow] = sum;
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
                int H_out = H_in + GetKernelHeight - 1;
                int W_out = W_in + GetKernelWidth  - 1;

                ComputeShader cs = DeepUnityMeta.ConvTranpose2DCS;

                ComputeBuffer inputBuffer = new ComputeBuffer(input.Count(), 4);
                inputBuffer.SetData(input.ToArray());
                cs.SetBuffer(0, "input", inputBuffer);

                ComputeBuffer gammaBuffer = new ComputeBuffer(kernels.Count(), 4);
                gammaBuffer.SetData(kernels.ToArray());
                cs.SetBuffer(0, "gamma", gammaBuffer);

                ComputeBuffer betaBuffer = new ComputeBuffer(bias ? biases.Count():GetOutChannels, 4);
                betaBuffer.SetData(bias ? biases.ToArray() : new float[GetOutChannels]);
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
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);

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

            // BIases are computed on CPU because is faster. The bias vector is too small in comparison with other stuff.
            if (bias && RequiresGrad)
            {
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
            }

            
            if (Device == Device.CPU)
            {
                Tensor inputGrad = InputCache.Rank == 3 ?
                    Tensor.Zeros(inputChannels, inputHeight, inputWidth) :
                    Tensor.Zeros(batchSize, inputChannels, inputHeight, inputWidth);


                if (RequiresGrad)
                {
                    // Compute the gradients of the weights - valid correlation(loss, x)      
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
                }

                // Input grad -> valid correlation(loss, weights)
                Parallel.For(0, batchSize, (Action<int>)(b =>
                {
                    Parallel.For(0, inputChannels, (Action<int>)(ic =>
                    {
                        for (int ih = 0; ih < inputHeight; ih++)
                        {
                            for (int iw = 0; iw < inputWidth; iw++)
                            {
                                float sum = this.bias ? biases[ic] : 0f;

                                for (int oc = 0; oc < outputChannels; oc++)
                                {
                                    for (int kh = 0; kh < kernelHeight; kh++)
                                    {
                                        for (int kw = 0; kw < kernelWidth; kw++)
                                        {
                                            
                                            sum += loss[b, oc, ih + kh, iw + kw] * kernels[oc, ic, kh, kw]; // kernel is rotated by 180d
                                            
                                        }
                                    }
                                }

                                inputGrad[b, ic, ih, iw] = sum / grad_scale; // summation over input channels
                            }
                        }
                    }));
                }));


                return inputGrad;
            }
            else
                throw new NotImplementedException();
        }



        public object Clone()
        {
            var convt = new ConvTranspose2D();
            convt.bias = bias;
            convt.kernels = (Tensor)kernels.Clone();        
            convt.kernelsGrad = (Tensor)kernelsGrad.Clone();

            if (bias)
            {
                convt.biases = (Tensor)biases.Clone();
                convt.biasesGrad = (Tensor)biasesGrad.Clone();
            }
           

            return convt;
        }
        public Parameter[] Parameters()
        {
            if (kernelsGrad == null)
                OnAfterDeserialize();

            var k = new Parameter(kernels, kernelsGrad);

            if (!bias)
                return new Parameter[] { k };


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

            if(bias)
                biasesGrad = Tensor.Zeros(biases.Shape);

        }
    }
}