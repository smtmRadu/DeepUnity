using System;
using DeepUnity.Modules;
namespace DeepUnity.Activations
{

    /// <summary>
    /// <b>Applies the Exponential activation function. </b><br></br>
    /// Input: (*) <br></br>
    /// Output: (*) <br></br>
    /// where * = any shape.
    /// </summary>
    [Serializable]
    public class Exponential : IModule, IActivation
    {
        /// <summary>
        /// <b>Applies the Exponential activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Exponential() { }

        protected Tensor OutputCache { get; set; }

        public Tensor Predict(Tensor x)
        {
            return x.Exp();
        }

        public Tensor Forward(Tensor x)
        {
            Tensor y = x.Exp();
            OutputCache = y.Clone() as Tensor;
            return y;
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * OutputCache;
        }

        public object Clone() => new Exponential();
    }
}
