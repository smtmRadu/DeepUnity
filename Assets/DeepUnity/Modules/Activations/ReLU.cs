using System;

namespace DeepUnity
{
    /// <summary>
    /// <b>Applies the Rectified Linear Unit activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class ReLU : Activation
    {
        /// <summary>
        /// <b>Applies the Rectified Linear Unit activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public ReLU() { }

        protected override Tensor Activate(Tensor x) => x.Select(k => Math.Max(0f, k));
        protected override Tensor Derivative(Tensor x) => x.Select(k => k > 0f ? 1f : 0f);       
    }
}