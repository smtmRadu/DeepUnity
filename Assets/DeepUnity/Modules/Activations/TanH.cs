using System;
using DeepUnity.Layers;

namespace DeepUnity.Activations
{
    /// <summary>
    /// <b>Applies the Hyperbolic Tangent activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class Tanh : IModule, IActivation
    {
        /// <summary>
        /// <b>Applies the Hyperbolic Tangent activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Tanh() { }

        protected Tensor OutputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return tanh;
            });
        }

        public Tensor Forward(Tensor x)
        {
            Tensor y = Predict(x);
            OutputCache = y.Clone() as Tensor;
            return y;
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * (-OutputCache.Pow(2f) + 1);
        }

        public object Clone() => new Tanh();
    }

}