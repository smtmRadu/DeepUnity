using System;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Activations
{
    // https://www.youtube.com/watch?v=09c7bkxpv9I

    /// <summary>
    /// <b>Applies the Softmax function over the last input's dimension H (axis: -1).</b> <br></br>
    /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// where * = any shape and H = features_num
    /// </summary>
    [Serializable]
    public class Softmax : IModule, IActivation
    {
        [SerializeField] private float temperature = 1f;
        /// <summary>
        /// <b>Applies the Softmax function over the last input's dimension H (axis: -1).</b> <br></br>
        /// Input: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
        /// Output: <b>(B, H)</b> or <b>(H)</b> for unbatched input <br></br>
        /// where * = any shape and H = features_num
        /// </summary>
        public Softmax(float temperature = 1f) 
        {
            if (temperature <= 0f)
                throw new ArgumentException("Temperature cannot be less or equal than 1");

            this.temperature = temperature;
        }

        private Tensor OutputCache { get; set; }
        public Tensor Predict(Tensor input)
        {
            int iRank = input.Rank;
            if (iRank != 1 && iRank != 2)
                throw new ShapeException("Softmax input must be of shape (B, H) or (H).");

            // softmax(x[i]) = e^x[i] / sum{j:1->H}(e^x[j]])
            Tensor exp = Tensor.Exp(input / temperature);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);
            exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));
            Tensor y = exp / exp_sum;
            return y;
        }
        public Tensor Forward(Tensor input)
        {
            Tensor y = Predict(input);
            OutputCache = y.Clone() as Tensor;
            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            bool isBatched = OutputCache.Rank == 2;
            int B = isBatched ? OutputCache.Size(-2) : 1;
            int H = OutputCache.Size(-1);

            Tensor jacobian_softmax = isBatched ? Tensor.Zeros(B, H, H) : Tensor.Zeros(H, H);

            for (int b = 0; b < B; b++)
            {
                for (int j = 0; j < H; j++)
                {
                    for (int i = 0; i < H; i++)
                    {
                        float delta = i == j ? 1 : 0;
                        jacobian_softmax[b, j, i] = OutputCache[b, i] * (delta - OutputCache[b, j]);
                    }
                }
            }

            return Tensor.MatMul(dLdY, jacobian_softmax); //  (B, H) * (H, H)
        }

        public object Clone() => new Softmax();
    }

}
