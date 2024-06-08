using System;
using UnityEngine;

namespace DeepUnity.Modules
{
    // https://www.youtube.com/watch?v=tMjdQLylyGI&t=602s
    // https://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf (251 - 253)
    /// <summary>
    /// Input: <b>(B, H_in)</b> or <b>(H_in)</b> for unbatched input.<br></br>
    /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.<br></br>
    /// where B = batch_size, H_in = in_features and H_out = out_features.
    /// </summary>
    [Serializable]
    public class LazyDense : ILearnable, IModule
    {
        [SerializeField] public Device Device { get => Dense.Device; set { if (Dense != null) Dense.Device = value; } } 
        [SerializeField] public bool RequiresGrad { get => Dense.RequiresGrad; set { if (Dense != null) Dense.RequiresGrad = value; } }
        [SerializeField] Dense Dense = null;
        [SerializeField] private bool initialized = false;
        private int outFeatures { get; set; }
        private bool bias { get; set; }
        private InitType weightInit { get; set; }
        private InitType biasInit { get; set; }
        private Device _device { get; set; }


        /// <summary>
        /// Input: <b>(B, H_in)</b> or <b>(H_in)</b> for unbatched input.<br></br>
        /// Output: <b>(B, H_out)</b> or <b>(H_out)</b> for unbatched input.<br></br>
        /// where B = batch_size, H_in = in_features and H_out = out_features.
        /// </summary>
        public LazyDense(int out_features, bool bias = true, InitType weight_init = InitType.LeCun_Uniform, InitType bias_init = InitType.LeCun_Uniform, Device device = default)
        {
            if (out_features < 1)
                throw new ArgumentException("Out_features cannot be less than 1.");

            this.Device = device;
            this.bias = bias;
            this.outFeatures = out_features;
            this.weightInit = weight_init;
            this.biasInit = bias_init;
            this._device = device;
            this.initialized = false;

        }

        public Tensor Predict(Tensor input)
        {
            if(!initialized)
            {
                Dense = new Dense(input.Size(-1), outFeatures, bias, weightInit, biasInit, _device);
                initialized = true;
            }

            return Dense.Predict(input);
        }

        public Tensor Forward(Tensor input)
        {
            if (!initialized)
            {
                Dense = new Dense(input.Size(-1), outFeatures, bias, weightInit, biasInit, _device);
                initialized = true;
            }

            return Dense.Forward(input);
        }
        public Tensor Backward(Tensor input)
        {
            return Dense.Backward(input);
        }

        public Parameter[] Parameters()
        {
            if (Dense == null)
                throw new Exception("LazyModule was not initialized through inference");

            return Dense.Parameters();
        }

        /// <summary>
        /// The cloned version is the base one (not lazy).
        /// </summary>
        /// <returns></returns>
        public object Clone()
        {
            return Dense.Clone();
        }

        public void OnBeforeSerialize()
        {
            Dense.OnBeforeSerialize();
        }
        public void OnAfterDeserialize()
        {
           Dense.OnAfterDeserialize();

        }

    }

}