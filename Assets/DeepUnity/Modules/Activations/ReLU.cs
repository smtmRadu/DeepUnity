using System;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    /// <summary>
    /// <b>Applies the Rectified Linear Unit activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class ReLU : IModule, IActivation
    {
        /// <summary>
        /// <b>Applies the Rectified Linear Unit activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public ReLU() { }


        protected Tensor InputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(k => Math.Max(0f, k));
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Select(k => k > 0f ? 1f : 0f);
        }

        public object Clone() => new ReLU();
    }
}