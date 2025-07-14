using System.Threading.Tasks;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    /// <summary>
    /// AdaBelief optimizer, with [decoupled] weight decay.(https://arxiv.org/pdf/2010.07468)
    /// </summary>
    public sealed class AdaBelief : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly bool amsgrad;
        private readonly bool maximize;
        private readonly bool decoupledWd;

        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] s;

        /// <summary>
        /// AdaBelief optimizer, with [decoupled] weight decay.(https://arxiv.org/pdf/2010.07468)
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="eps"></param>
        /// <param name="weight_decay"></param>
        /// <param name="decoupled_wd">Whether or not the WD is decoupled.</param>
        /// <param name="amsgrad"></param>
        /// <param name="maximize"></param>
        public AdaBelief(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8F, float weight_decay = 0f, bool decoupled_wd = true, bool amsgrad = false, bool maximize = false) : base(parameters, lr, eps, weight_decay, maximize)
        {
            this.amsgrad = amsgrad;
            this.maximize = maximize;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.decoupledWd = decoupled_wd;

            m = new Tensor[parameters.Length];
            s = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].param.Shape);
                s[i] = Tensor.Zeros(parameters[i].param.Shape);
            }
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            Parallel.For(0, parameters.Length, i =>
            {
                if (maximize)
                    Tensor.CopyTo(-parameters[i].g, parameters[i].g);

                // Decoupled lambda
                if (!decoupledWd)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].param, parameters[i].g);

                Tensor s_prev = s[i].Clone() as Tensor; // cache s[t-1]

                Tensor.CopyTo(beta1 * m[i] + (1f - beta1) * parameters[i].g, m[i]);
                Tensor.CopyTo(beta2 * s[i] + (1f - beta2) * (parameters[i].g - m[i]).Pow(2f) + epsilon, s[i]);

                // AMSGrad
                s[i] = amsgrad ? Tensor.Maximum(s[i], s_prev) : s[i];

                // Bias Correction
                Tensor mHat = m[i] / (1f - beta1_t);
                Tensor sHat = s[i] / (1f - beta2_t);

                // Update
                if(decoupledWd)
                    Tensor.CopyTo(parameters[i].param - mHat * gamma / (s[i].Sqrt() + epsilon) - gamma * lambda * parameters[i].param, parameters[i].param);
                else
                    Tensor.CopyTo(parameters[i].param - mHat * gamma / (s[i].Sqrt() + epsilon), parameters[i].param);
            });
        }
    }
}