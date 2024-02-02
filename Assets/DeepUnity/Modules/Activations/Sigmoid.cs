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
    public class Sigmoid : IModule, IActivation
    {
        /// <summary>
        /// <b>Applies the Logistic Sigmoid activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape.
        /// </summary>
        public Sigmoid() { }

        protected Tensor OutputCache { get; set; }
        public Tensor Predict(Tensor x)
        {
            return x.Select(x =>
            {
                float sigmoid = 1f / (1f + MathF.Exp(-x));
                return sigmoid;
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
            return dLdY * (-OutputCache + 1f);
        }

        public object Clone() => new Sigmoid();
    }
}
