using System.Threading.Tasks;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{

    // Stable and low-precision training for large-scale vision-language models Mitchell Wortsman∗1 Tim Dettmers∗1 Luke Zettlemoyer12 Ari Morcos†2
    // Ali Farhadi†1 Ludwig Schmidt†134
    // https://optimi.benjaminwarner.dev/optimizers/stableadamw/
    public sealed class StableAdamW : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly bool fused;
        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;

        /// <summary>
        /// StableAdamW is a drop-in replacement for AdamW and uses the same hyperparameters, with one exception: StableAdamW removes the need for gradient clipping.
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"> If training on large batch sizes or still observing training loss spikes, consider reducing it to [0.95, 0.99] </param>
        /// <param name="eps"></param>
        /// <param name="weight_decay"></param>
        /// <param name="amsgrad"></param>
        /// <param name="maximize"></param>
        public StableAdamW(
            Parameter[] parameters, 
            float lr = 0.001f, 
            float beta1 = 0.9f, 
            float beta2 = 0.99f, 
            float eps = 1e-6F, 
            float weight_decay = 0.01f,
            bool maximize = false,
            bool fused = false) : base(parameters, lr, eps, weight_decay, maximize)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.maximize = maximize;
            this.fused = fused;
            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].param.Shape);
                v[i] = Tensor.Zeros(parameters[i].param.Shape);
            }
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            Parallel.For(0, parameters.Length, i =>
            {
                // The fused implementations is a bit strange, like it shows sort of fluctations idk why.
                // if (fused)
                // {
                //     Tensor.FusedStableAdamW(
                //         param: parameters[i].param,
                //         g: parameters[i].g,
                //         m: m[i],
                //         v: v[i],
                //         gamma: gamma,
                //         betas: (beta1, beta2),
                //         betas_t: (beta1_t, beta2_t),
                //         lambda: lambda,
                //         eps: epsilon,
                //         maximize: maximize);
                //     return;
                // }

                if (maximize)
                    Tensor.CopyTo(-parameters[i].g, parameters[i].g);

                Tensor g_squared = parameters[i].g.Square();

                Tensor.CopyTo(beta1 * m[i] + (1f - beta1) * parameters[i].g, m[i]);
                Tensor.CopyTo(beta2 * v[i] + (1f - beta2) * g_squared, v[i]);

                Tensor mHat = m[i] / (1f - beta1_t);
                Tensor vHat = v[i] / (1f - beta2_t);

                Tensor RMS = Tensor.Sqrt(g_squared / Tensor.Maximum(v[i], Tensor.Fill(epsilon * epsilon, v[i].Shape)));

                Tensor eta = gamma / Tensor.Maximum(Tensor.Ones(RMS.Shape), RMS);

                Tensor.CopyTo(parameters[i].param - eta * (mHat / (vHat.Sqrt() + epsilon) + lambda * parameters[i].param), parameters[i].param);
            });
        }
    }
}