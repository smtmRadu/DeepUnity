using System;
using DeepUnity.Layers;

namespace DeepUnity.Activations
{
    // https://arxiv.org/pdf/1602.02068.pdf
    // https://www.youtube.com/watch?v=09c7bkxpv9I
    // https://medium.com/deeplearningmadeeasy/sparsemax-from-paper-to-code-351e9b26647b

    /// <summary>
    /// <b>Applies the Sparsemax function over the last input's dimension H (axis: -1).</b> <br></br>
    /// Input: <b>(B, K)</b> or <b>(K)</b> for unbatched input <br></br>
    /// Output: <b>(B, K)</b> or <b>(K)</b> for unbatched input <br></br>
    /// where * = any shape and K = features_num
    /// </summary>
    [Serializable]
    public class Sparsemax : IModule, IActivation
    {
        private Tensor OutputCache { get; set; }
        /// <summary>
        /// <b>Applies the Sparsemax function over the last input's dimension H (axis: -1).</b> <br></br>
        /// Input: <b>(B, K)</b> or <b>(K)</b> for unbatched input <br></br>
        /// Output: <b>(B, K)</b> or <b>(K)</b> for unbatched input <br></br>
        /// where * = any shape and K = features_num
        /// </summary>
        public Sparsemax()
        {

        }

        public Tensor Predict(Tensor z)
        {
            Tensor z_sorted = z.Sort(-1, false); // (B, H)

            Tensor cumsum = z_sorted.CumSum(-1); // (B, H)
            Tensor col_range = Tensor.Arange(1, z.Size(-1) + 1); // (H)

            bool isBatched = z.Rank == 2;
            int batch_size = isBatched ? z.Size(0) : 1;
            int K = z.Size(-1);

            Tensor isGreater = Tensor.Zeros(z.Shape); // (B, H)
            for (int b = 0; b < batch_size; b++)
            {
                for (int k = 0; k < K; k++)
                {
                    if (1 + col_range[k] * z_sorted[b, k] > cumsum[b, k])
                        isGreater[b, k] = 1;
                }
            }

            Tensor kz = isGreater.Sum(-1, true); // (B, 1)

            Tensor some = isBatched ? Tensor.Zeros(batch_size, 1) : Tensor.Zeros(1);
            for (int i = 0; i < batch_size; i++)
            {
                int indx = (int)(kz[i, 0] - 1f);
                some[i, 0] = cumsum[i, indx] - 1;
            }
            Tensor tau_z = (some - 1) / kz;


            Tensor y = Tensor.Maximum(z - tau_z.Expand(-1, K), Tensor.Zeros(z.Shape));
            return y;
        }
        public Tensor Forward(Tensor z)
        {
            Tensor y = Predict(z);
            OutputCache = y.Clone() as Tensor;
            return y;
        }
        public Tensor Backward(Tensor dLdY)
        {
            int K = OutputCache.Size(-1);

            Tensor non_zeros = Tensor.Ne(OutputCache, Tensor.Zeros(OutputCache.Shape)); // (B, K)
            Tensor support_size = non_zeros.Sum(-1, true);
            Tensor v_hat = (dLdY * non_zeros).Sum(-1, true) / support_size;

            return non_zeros * (dLdY - v_hat.Expand(-1, K));
        }

        public object Clone() => new Sparsemax();
    }

}
