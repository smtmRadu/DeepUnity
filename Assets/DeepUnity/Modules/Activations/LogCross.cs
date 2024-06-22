using System;
using System.Threading.Tasks;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class LogCross : IModule, IActivation
    {
        private Tensor InputCache { get; set; }


        /// <summary>
        /// <b>An experimental activation function. </b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape. 
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public LogCross() { throw new InvalidOperationException("This activation is purely experimental."); }

        public Tensor Predict(Tensor x)
        {
            Tensor output = Tensor.Zeros(x.Shape);
            Parallel.For(0, x.Count(), i =>
            {
                if (x[i] >= 0)
                    output[i] = MathF.Log(x[i] + 1f);
                else
                    output[i] = -MathF.Log(1f - x[i]);
            });
            return output;
        }

        public Tensor Forward(Tensor x)
        {
            InputCache = x.Clone() as Tensor;
            return Predict(x);
        }

        public Tensor Backward(Tensor dLdY)
        {
            Tensor inputGrad = Tensor.Zeros(dLdY.Shape);
            Parallel.For(0, InputCache.Count(), i =>
            {
                if(InputCache[i] >= 0)
                    inputGrad[i] = dLdY[i] / (InputCache[i] + 1f);
                else
                    inputGrad[i] = dLdY[i] / (1f - InputCache[i]);
            });
            return inputGrad;
        }

        public object Clone() => new LogCross();
    }
}




