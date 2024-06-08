using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    [Serializable]
    public class LazyConv2D : ILearnable, IModule
    {
        [SerializeField] public Device Device { get => Conv2D.Device; set { if (Conv2D != null) Conv2D.Device = value; } }
        [SerializeField] public bool RequiresGrad { get => Conv2D.RequiresGrad; set { if (Conv2D != null) Conv2D.RequiresGrad = value; } }
        [SerializeField] Conv2D Conv2D;
        [SerializeField] private bool initialized = false;
        private int outChannels { get; set; }
        private (int, int) kernelShape { get; set; }
        private bool bias { get; set; }
        private InitType weightInit { get; set; }
        private InitType biasInit { get; set; }
        private Device _device { get; set; }

        public LazyConv2D(int out_channels, (int, int) kernel_shape, bool bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = Device.CPU)
        {
            if (out_channels < 1)
                throw new ArgumentException("Cannot have less than 1 output channel.");

            this.outChannels = out_channels;
            this.kernelShape = kernel_shape;
            this.bias = bias;
            this.weightInit = weight_init;
            this.biasInit = bias_init;
            this._device = device;
            this.initialized = false;
        }

        public LazyConv2D(int out_channels, int kernel_size, bool bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = Device.CPU) :
            this(out_channels, (kernel_size, kernel_size), bias, weight_init, bias_init, device){ }
        public Tensor Predict(Tensor input)
        {
            if (!initialized)
            {
                Conv2D = new Conv2D(input.Size(-3), outChannels, kernelShape, bias, weightInit, biasInit, _device);
                initialized = true;
            }

            return Conv2D.Predict(input);
        }

        public Tensor Forward(Tensor input)
        {
            if (!initialized)
            {
                Conv2D = new Conv2D(input.Size(-3), outChannels, kernelShape, bias, weightInit, biasInit, _device);
                initialized = true;
            }

            return Conv2D.Forward(input);
        }
        public Tensor Backward(Tensor input)
        {
            return Conv2D.Backward(input);
        }

        public Parameter[] Parameters()
        {
            if (Conv2D == null)
                throw new Exception("LazyModule was not initialized through inference");

            return Conv2D.Parameters();
        }

        /// <summary>
        /// The cloned version is the base one (not lazy).
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return Conv2D.Clone();
        }

        public void OnBeforeSerialize()
        {
            Conv2D.OnBeforeSerialize();
        }
        public void OnAfterDeserialize()
        {
            Conv2D.OnAfterDeserialize();

        }

    }

}