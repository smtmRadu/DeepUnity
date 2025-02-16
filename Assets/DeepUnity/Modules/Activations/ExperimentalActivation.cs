using System;
using System.Threading.Tasks;
using DeepUnity.Modules;

namespace DeepUnity.Activations
{
    [Serializable]
    public sealed class ExperimentalActivation : IModule, IActivation
    {
        private Tensor InputCache { get; set; }


        /// <summary>
        /// <b>An experimental activation function. DO NOT USE.</b><br></br>
        /// Input: (*) <br></br>
        /// Output: (*) <br></br>
        /// where * = any shape. 
        /// </summary>
        /// <param name="in_place">Modifies the input tensor in place.</param>
        public ExperimentalActivation() { }// { throw new InvalidOperationException("This activation is purely experimental."); }

        public Tensor Predict(Tensor input)
        {
            Tensor output = Tensor.Zeros(input.Shape);
            Parallel.For(0, input.Count(), i =>
            {
                var x = input[i];

                output[i] = x * Utils.Sigmoid(x - (float)Math.Log(x * x));
                // new one
                // output[i] = x * MathF.Exp((x - 1) - MathF.Log(1f + MathF.Pow((x-1), 2f), MathF.E)); 

                // doesn't converge.
                // output[i] = MathF.Sqrt(MathF.Abs(x)) * MathF.Sign(x);
                
                // not good for is non continuous
                //if (x[i] >= 0)
                // output[i] = MathF.Log(x[i] + 1f);
                //else
                // output[i] = -MathF.Log(1f - x[i]);
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
                var x = InputCache[i];
                inputGrad[i] = MathF.Exp(x) * ((x - 1f) * x * x + MathF.Exp(x)) / MathF.Pow(x * x + MathF.Exp(x), 2f);
                
                //inputGrad[i] = (MathF.Exp(x - 1) * (MathF.Pow(x, 3) - 3 * x * x + 2 * x + 2)) / MathF.Pow(x * x - 2 * x + 2, 2);

                // inputGrad[i] = x * MathF.Sign(x) / (2f * MathF.Pow(MathF.Abs(x), 1.5f));



                //  if(InputCache[i] >= 0)
                //      inputGrad[i] = dLdY[i] / (InputCache[i] + 1f);
                //  else
                //      inputGrad[i] = dLdY[i] / (1f - InputCache[i]);
            });
            return inputGrad;
        }

        public object Clone() => new ExperimentalActivation();
    }
}




