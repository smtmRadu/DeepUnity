using System;
using System.Threading.Tasks;

namespace DeepUnity
{
    public partial class Tensor : IEquatable<Tensor>, IEquatable<TensorGPU>, ICloneable
    {
        /// <summary>
        /// A fused implementation for AdamW as an optimization (+5%) that avoids tensor creations.
        /// It is also faster than Adam w/o wd.
        /// </summary>
        /// <param name="param"></param>
        /// <param name="g"></param>
        /// <param name="m"></param>
        /// <param name="v"></param>
        /// <param name="vMax"></param>
        /// <param name="gamma"></param>
        /// <param name="betas"></param>
        /// <param name="betas_t"></param>
        /// <param name="lambda"></param>
        /// <param name="eps"></param>
        /// <param name="maximize"></param>
        /// <param name="amsgrad"></param>
        public static void FusedAdamW(Tensor param, Tensor g, Tensor m, Tensor v, Tensor vMax, float gamma, (float, float) betas, (float, float) betas_t, float lambda, float eps, bool maximize, bool amsgrad)
        {
            // let this parallel, tests were made
            Parallel.For(0, param.data.Length, i =>
            {
                if (maximize)
                    g.data[i] = -g.data[i];

                param.data[i] = param.data[i] - gamma * lambda * param.data[i];

                m.data[i] = betas.Item1 * m.data[i] + (1f - betas.Item1) * g.data[i];
                v.data[i] = betas.Item2 * v.data[i] + (1f - betas.Item2) * g.data[i] * g.data[i];

                float mhat = m.data[i] / (1f - betas_t.Item1);
                float vhat = v.data[i] / (1f - betas_t.Item2);

                if (amsgrad)
                {
                    vMax.data[i] = Math.Max(vMax.data[i], vhat);
                    param.data[i] = param.data[i] - gamma * mhat / (MathF.Sqrt(vMax.data[i]) + eps);
                }
                else
                    param.data[i] = param.data[i] - gamma * mhat / (MathF.Sqrt(vhat) + eps);
            });
        }

        /// <summary>
        /// A faster implementation for running normalizer.
        /// </summary>
        /// <param name="x"></param>
        /// <param name="mean"></param>
        /// <param name="variance"></param>
        /// <returns></returns>
        public static Tensor FusedRunningNormalize(Tensor x, Tensor mean, Tensor m2, int step)
        {
            Tensor output = new Tensor(x.shape);
            int dim = x.Size(-1);
            if(x.Rank == 1)
            {
                for (int i = 0; i < dim; i++)
                {
                    output.data[i] = (m2.data[i] - mean.data[i]) / (step - 1);
                }
            }
            else if(x.Rank == 2)
            {
                int batch_size = x.Size(0);
                Parallel.For(0, batch_size, b =>
                {
                    for (int i = 0; i < dim; i++)
                    {
                        output[b, i] = (m2.data[i] - mean.data[i]) / (step - 1);
                    }
                });
            }
            else
            {
                throw new ShapeException("Cannot handle tensors of Rank higher than 2");
            }
            return output;
                
        }
    }

}
