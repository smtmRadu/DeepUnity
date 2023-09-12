
using System;
namespace DeepUnity
{
    // https://www.youtube.com/watch?v=09c7bkxpv9I
    /// <summary>
    /// <b>Applies the Softmax function over the last input's dimension H (-1).</b> <br></br>
    /// Input: <b>(*, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// Output: <b>(*, H)</b> or <b>(H)</b> for unbatched input <br></br>
    /// where * = any shape and H = features_num
    /// </summary>
    [Serializable]
    public class Softmax : IModule
    {
        private Tensor InputCache { get; set; }

        /// <summary>
        /// <b>Applies the Softmax function over the last input's dimension H (-1).</b> <br></br>
        /// Input: <b>(*, H)</b> or <b>(H)</b> for unbatched input <br></br>
        /// Output: <b>(*, H)</b> or <b>(H)</b> for unbatched input <br></br>
        /// where * = any shape and H = features_num
        /// </summary>
        public Softmax() { }
        public Tensor Predict(Tensor input)
        {
            // softmax(x[i]) = e^x[i] / sum{j:1->N}(e^x[j]])
            Tensor exp = Tensor.Exp(input);
            Tensor exp_sum = Tensor.Sum(exp, -1, true);
            exp_sum = Tensor.Expand(exp_sum, -1, exp.Size(-1));
            return exp / exp_sum;
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = input;
            return Predict(input);
        }
        public Tensor Backward(Tensor dLdY)
        {
            // Use InputCache to calculate dLdX
            Tensor s = Predict(InputCache);

            bool isBatched = dLdY.Rank == 2;
            int B = isBatched ? s.Size(-2) : 1;
            int C = s.Size(-1);

            Tensor dLdX = Tensor.Zeros(InputCache.Shape);

            for (int b = 0; b < B; b++)
            {
                for (int i = 0; i < C; i++)
                {
                    for (int j = 0; j < C; j++)
                    {
                        float dSi_dxj;
                        if (i == j)
                        {
                            dSi_dxj = s[b, i] * (1f - s[b, i]);
                        }
                        else
                        {
                            dSi_dxj = -s[b, i] * s[b, j];
                        }

                        // dL/dX = dL/dY * dY/dX
                        dLdX[b, i] += dLdY[b, j] * dSi_dxj;
                    }
                }
            }
            

            return dLdX;
        }
    }

}

