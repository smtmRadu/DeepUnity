using System.Threading.Tasks;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        /// <summary>
        /// Gemma RMS norm inits with gamma=0 and the gain is y = x * (1 + w)
        /// </summary>
        public class Gemma3RMSNorm
        {
            private int num_features;
            public float eps;
            public float[] gamma;
            public bool IsInitialized { get; private set; } = false;

            public Gemma3RMSNorm(int num_features, float eps = 1e-6f, string weights_path = null)
            {
                this.num_features = num_features;
                this.eps = eps;
                this.gamma = new float[num_features];// let it here so it is readable for params num
                if (!string.IsNullOrEmpty(weights_path))
                {
                    _ = LoadWeightsAsync(weights_path);
                }
            }

            private async Task LoadWeightsAsync(string path)
            {
                this.gamma = await Task.Run(() => Utils.ReadWeights(path, num_features));
                IsInitialized = true;
                // ConsoleMessage.Info($"Loaded {path}");
                //Debug.Log(this.gamma.ToCommaSeparatedString());
            }


            public Tensor Predict(Tensor x)
            {
                Tensor ms = x.Square().Mean(-1, keepDim: true).Expand(-1, x.Size(-1));
                Tensor x_norm = x * Tensor.RSqrt(ms + eps);

                Tensor y = Tensor.Zeros(x.Shape);
                if (x.Rank == 2)
                {
                    int seq_len = x.Size(-2);
                    int emb_dim = x.Size(-1);
                    for (int l = 0; l < seq_len; l++)
                    {
                        for (int e = 0; e < emb_dim; e++)
                        {
                            y[l, e] = x_norm[l, e] * (1f + gamma[e]);
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
                                y[b, l, e] = x_norm[b, l, e] * (1f + gamma[e]);
                            }
                        }

                    }

                }

                return y;

            }



        }

    }
}
