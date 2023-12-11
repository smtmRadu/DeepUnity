using System;

namespace DeepUnity
{
    /// <summary>
    /// <b>Applies the Hyperbolic Tangent activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class Tanh : Activation
    {
        /// <summary>
        /// <b>Applies the Hyperbolic Tangent activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Tanh() { }
        protected override Tensor Activate(Tensor x)
        {
            return x.Select(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return tanh;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return  1f - tanh * tanh;
            });
        }
        public override object Clone() => new Tanh();
    }

}