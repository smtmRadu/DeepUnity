/*using DeepUnity.Modules;
using System;
using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

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
    /// H_out = (H_in - 1) * kernel_height + (kernel_height - 1) + 1<br></br> 
    /// W_out = (W_in - 1) * kernel_width + (kernel_width - 1) + 1
    /// </summary>
    [Serializable]
    public class Conv2DTranspose : ILearnable, IModule
    {
        [SerializeField] public Device Device { get; set; } = Device.CPU;
        private Tensor InputCache { get; set; }

        private int GetOutChannels { get => kernels.Size(-3); }
        private int GetInChannels { get => kernels.Size(-4); }
        private int GetKernelHeight { get => kernels.Size(-2); }
        private int GetKernelWidth { get => kernels.Size(-1); }

        [SerializeField] private Tensor kernels;
        [SerializeField] private Tensor biases;
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
        /// H_out = (H_in - 1) * kernel_height + (kernel_height - 1) + 1<br></br> 
        /// W_out = (W_in - 1) * kernel_width + (kernel_width - 1) + 1
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="out_channels">C_out</param>
        /// <param name="kernel_size"></param>
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public Conv2DTranspose(int in_channels, int out_channels, int kernel_size, Device device = Device.CPU)
        {
            throw new NotImplementedException("The ConvTranspose2D layer was not implemented yet.");

            if (in_channels < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_size < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.Device = device;

            float k = 1f / (out_channels * kernel_size * kernel_size);
            k = Mathf.Sqrt(k);
            kernels = Tensor.RandomRange((-k, k), in_channels, out_channels, kernel_size, kernel_size);
            biases = Tensor.RandomRange((-k, k), out_channels);
            kernelsGrad = Tensor.Zeros(kernels.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
        }
        /// <summary>
        /// Input: (<b>B</b>, <b>C_in</b>, <b>H_in</b>, <b>W_in</b>) or (<b>C_in</b>, <b>H_in</b>, <b>W_in</b>) for unbatched input.<br/>
        /// Output: <b>(B, C_out, H_out, W_out)</b> or <b>(C_out, H_out, W_out)</b> for unbatched input.<br></br>
        /// <br></br>
        /// where <br></br>
        /// B = batch_size <br></br>
        /// C_in = in_channels <br></br> 
        /// C_out = out_channels, <br></br>
        /// H_out = (H_in - 1) * kernel_height + (kernel_height - 1) + 1<br></br> 
        /// W_out = (W_in - 1) * kernel_width + (kernel_width - 1) + 1
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public Conv2DTranspose(int in_channels, int out_channels, (int, int) kernel_shape, Device device = Device.CPU)
        {
            throw new NotImplementedException("The ConvTranspose2D layer was not implemented yet.");

            if (in_channels < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_shape.Item1 < 2 || kernel_shape.Item2 < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.Device = device;

            float k = 1f / (out_channels * kernel_shape.Item1 * kernel_shape.Item2);
            k = Mathf.Sqrt(k);
            kernels = Tensor.RandomRange((-k, k), in_channels, out_channels, kernel_shape.Item1, kernel_shape.Item2);
            biases = Tensor.RandomRange((-k, k), out_channels);
            kernelsGrad = Tensor.Zeros(kernels.Shape);
            biasesGrad = Tensor.Zeros(biases.Shape);
        }




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

                int outputHeight = inputHeight * kernelHeight;
                int outputWidth = inputWidth * kernelWidth;

                // Initialize output tensor
                Tensor output = input.Rank == 3 ?
                    Tensor.Zeros(outputChannels, outputHeight, outputWidth) :
                    Tensor.Zeros(batch_size, outputChannels, outputHeight, outputWidth);

                // Perform transposed convolution
                Parallel.For(0, batch_size, b =>
                {
                    Parallel.For(0, outputChannels, oc =>
                    {
                        for (int h = 0; h < outputHeight; h++)
                        {
                            for (int w = 0; w < outputWidth; w++)
                            {
                                float sum = 0f;

                                for (int ic = 0; ic < inputChannels; ic++)
                                {
                                    for (int j = 0; j < kernelHeight; j++)
                                    {
                                        for (int i = 0; i < kernelWidth; i++)
                                        {
                                            int inputH = h - j;
                                            int inputW = w - i;

                                            if (inputH >= 0 && inputH < inputHeight && inputW >= 0 && inputW < inputWidth)
                                            {
                                                sum += input[b, ic, inputH, inputW] * kernels[oc, ic, j, i];
                                            }
                                        }
                                    }
                                }

                                output[b, oc, h, w] = sum;
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
            InputCache = Tensor.Identity(input);

            return Predict(input);
        }

        /// <param name="loss">(B, C_out, H - K_h + 1, W - K_w + 1)</param>
        /// <returns></returns>
        public Tensor Backward(Tensor loss)
        {
            bool isBatched = loss.Rank == 4;
            int batch_size = isBatched ? loss.Size(-4) : 1;


            return null;
        }



        public object Clone()
        {
            var conv = new Conv2DTranspose(GetInChannels, GetOutChannels, kernel_shape: (GetKernelHeight, GetKernelWidth), device: Device);
            conv.kernels = (Tensor)kernels.Clone();
            conv.biases = (Tensor)biases.Clone();
            conv.kernelsGrad = (Tensor)kernelsGrad.Clone();
            conv.biasesGrad = (Tensor)biasesGrad.Clone();

            return conv;
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
    }
}
*/