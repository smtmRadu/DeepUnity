using System.Diagnostics;
using System.Threading.Tasks;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    public sealed class AdamW : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly bool amsgrad;
        private readonly bool maximize;
        private readonly bool fused;

        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;
        private readonly Tensor[] vHatMax;
        /// <summary>
        /// Adam optimizer with decoupled weight decay. If training on larger batch sizes, use a beta_2 between [0.95, 0.99].
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="eps"></param>
        /// <param name="weight_decay"></param>
        /// <param name="amsgrad"></param>
        /// <param name="maximize"></param>
        /// <param name="fused"> Either to use a fused call for adam or not. It might be faster.</param>
        public AdamW(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8F, float weight_decay = 0.01f, bool amsgrad = false, bool maximize = false, bool fused = false) : base(parameters, lr, eps, weight_decay)
        {
            this.amsgrad = amsgrad;
            this.maximize = maximize;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.fused = fused;

            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];

            if (amsgrad)
                vHatMax = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].param.Shape);
                v[i] = Tensor.Zeros(parameters[i].param.Shape);

                if (amsgrad)
                    vHatMax[i] = Tensor.Zeros(parameters[i].param.Shape);           
            }
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            Parallel.For(0, parameters.Length, i =>
            {
                if(fused)
                {
                    Tensor.FusedAdamW(
                        param: parameters[i].param,
                        g: parameters[i].g,
                        m: m[i],
                        v: v[i],
                        vMax: amsgrad ? vHatMax[i] : null,

                        gamma: gamma,
                        betas: (beta1, beta2),
                        betas_t: (beta1_t, beta2_t),
                        lambda: lambda,
                        eps: epsilon,
                        maximize: maximize,
                        amsgrad: amsgrad);
                    return;
                }
                 
                if (maximize)
                    Tensor.CopyTo(-parameters[i].g, parameters[i].g);

                Tensor.CopyTo(parameters[i].param - gamma * lambda * parameters[i].param, parameters[i].param);

                Tensor.CopyTo(beta1 * m[i] + (1f - beta1) * parameters[i].g, m[i]);
                Tensor.CopyTo(beta2 * v[i] + (1f - beta2) * parameters[i].g.Square(), v[i]);

                Tensor mHat = m[i] / (1f - beta1_t);
                Tensor vHat = v[i] / (1f - beta2_t);

                if (amsgrad)
                {
                    vHatMax[i] = Tensor.Maximum(vHatMax[i], vHat);
                    Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHatMax[i].Sqrt() + epsilon), parameters[i].param);
                }
                else
                    Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHat.Sqrt() + epsilon), parameters[i].param);            
            });
        }
    }

    // 3 dense model test (128 hidden size)
    // amsgrad = 2.22s
    // amsgrad + fused = 2.19
    // amsgrad + fused + parallel = 2.08s



}