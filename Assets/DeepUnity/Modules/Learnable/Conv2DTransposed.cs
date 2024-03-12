/*using System;
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
    /// H_out = H_in - kernel.height + 1 <br></br> 
    /// W_out = W_in - kernel.width + 1
    /// </summary>
    [Serializable]
    public class Conv2DTransposed : ILearnable, IModule
    {

        private Tensor InputCache { get; set; }

        private int GetOutChannels { get => kernels.Size(-3); }
        private int GetInChannels { get => kernels.Size(-4); }
        private int GetKernelHeight { get => kernels.Size(-2); }
        private int GetKernelWidth { get => kernels.Size(-1); }


        [SerializeField] private Device device;
        [SerializeField] private Tensor kernels;
        [SerializeField] private Tensor biases;
        [NonSerialized] private Tensor kernelsGrad;
        [NonSerialized] private Tensor biasesGrad;

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
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public Conv2DTransposed(int in_channels, int out_channels, int kernel_size, Device device = Device.CPU)
        {
            if (in_channels < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_size < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.device = device;

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
        /// H_out = H_in - kernel_shape.Item1 + 1 <br></br> 
        /// W_out = W_in - kernel_shape.Item2 + 1
        /// </summary>
        /// <param name="input_shape">(C_in, H, W)</param>
        /// <param name="gamma_init">Initializer used for weights.</param>
        /// <param name="beta_init">Initializer used for biases.</param>
        public Conv2DTransposed(int in_channels, int out_channels, (int, int) kernel_shape, Device device = Device.CPU)
        {
            if (in_channels < 1)
                throw new ArgumentException("Cannot have less than 1 input channels.");

            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            if (kernel_shape.Item1 < 2 || kernel_shape.Item2 < 2)
                throw new ArgumentException("Cannot have less than 2 kernel size.");

            this.device = device;

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
            return null;
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
            var conv = new Conv2DTransposed(GetInChannels, GetOutChannels, kernel_shape: (GetKernelHeight, GetKernelWidth), device: device);
            conv.kernels = (Tensor)kernels.Clone();
            conv.biases = (Tensor)biases.Clone();
            conv.kernelsGrad = (Tensor)kernelsGrad.Clone();
            conv.biasesGrad = (Tensor)biasesGrad.Clone();

            return conv;
        }


        public void SetDevice(Device device) => this.device = device;
        public int ParametersCount()
        {
            return kernels.Count() + biases.Count();
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