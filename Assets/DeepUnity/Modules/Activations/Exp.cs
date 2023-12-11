using System;
namespace DeepUnity
{

    /// <summary>
    /// <b>Applies the Exponential activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class Exp : Activation
    {
        /// <summary>
        /// <b>Applies the Exponential activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Exp() { }
        protected override Tensor Activate(Tensor x) => x.Exp();
        protected override Tensor Derivative(Tensor x) => x.Exp();

        public override object Clone() => new Exp();
    }
}
