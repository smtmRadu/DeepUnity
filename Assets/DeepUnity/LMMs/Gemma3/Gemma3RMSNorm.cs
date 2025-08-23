using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class Gemma3RMSNorm
        {
            private int num_features;
            public float eps;
            public float[] gamma;

            public Gemma3RMSNorm(int num_features, float eps = 1e-6f)
            {
                this.num_features = num_features;
                this.eps = eps;
                this.gamma = new float[num_features];
            }


            public Tensor Predict(Tensor x)
            {
                Tensor ms = x.Square().Mean(-1, keepDim: true).Expand(-1, x.Size(-1));
                Tensor x_norm = x / Tensor.Sqrt(ms + eps);

                Tensor y = Tensor.Zeros(x.Shape);
                if (x.Rank == 2)
                {
                    int seq_len = x.Size(-2);
                    int emb_dim = x.Size(-1);
                    for (int l = 0; l < seq_len; l++)
                    {
                        for (int e = 0; e < emb_dim; e++)
                        {
                            y[l, e] = x_norm[l, e] * gamma[e];
                        }
                    }
                }
                else if (x.Rank == 3)
                {
                    int bat_siz = x.Size(-3);
                    int seq_len = x.Size(-2);
                    int emb_dim = x.Size(-1);
                    for (int b = 0; b < bat_siz; b++)
                    {
                        for (int l = 0; l < seq_len; l++)
                        {
                            for (int e = 0; e < emb_dim; e++)
                            {
                                y[l, e] = x_norm[l, e] * gamma[e];
                            }
                        }

                    }

                }

                return y;

            }



        }

    }
}
