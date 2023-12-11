using System;

namespace DeepUnity
{
    /// <summary>
    /// <b>Applies the Logistic Sigmoid activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class Sigmoid : Activation
    {
        /// <summary>
        /// <b>Applies the Logistic Sigmoid activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Sigmoid() { }
        protected override Tensor Activate(Tensor x)
        {
            return x.Select(x => 
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid;
            });
        }
        protected override Tensor Derivative(Tensor x)
        {
            return x.Select(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return  sigmoid * (1f - sigmoid);
            });
        }

        public override object Clone() => new Sigmoid();
    }
}
