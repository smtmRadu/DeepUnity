using System;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    // https://openreview.net/forum?id=OM0jvwB8jIp57ZJjtNEZ
    // https://pytorch.org/docs/stable/generated/torch.optim.NAdam.html#torch.optim.NAdam
    // 
    public sealed class NAdam : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly float psi;
        private readonly bool decoupled_weight_decay;

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;

        // Mu Product 1 -> t
        private float mu_prod_t = 1f;



        public NAdam(Parameter[] parameters, float lr = 0.002f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 0f, float momentum_decay = 0.004f, bool decoupled_weight_decay = false) : base(parameters, lr, eps, weightDecay)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            psi = momentum_decay;
            this.decoupled_weight_decay = decoupled_weight_decay;

            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];


            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].theta.Shape);
                v[i] = Tensor.Zeros(parameters[i].theta.Shape);
            }
        }

        public override void Step()
        {
            t++;

            float mu_t = beta1 * (1f - 0.5f * MathF.Pow(0.96f, t * psi));
            float mu_next_t = beta1 * (1f - 0.5f * MathF.Pow(0.96f, (t + 1f) * psi));
            mu_prod_t *= mu_t;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                {
                    if (decoupled_weight_decay)
                        Tensor.CopyTo(parameters[i].theta - gamma * lambda * parameters[i].theta, parameters[i].theta);
                    else
                        Tensor.CopyTo(parameters[i].g + lambda * parameters[i].theta, parameters[i].g);
                }

                m[i] = beta1 * m[i] + (1f - beta1) * parameters[i].g;
                v[i] = beta2 * v[i] + (1f - beta2) * parameters[i].g.Pow(2f);

                Tensor mHat = mu_next_t * m[i] / (1f - mu_prod_t * mu_next_t)
                              + (1f - mu_t) * parameters[i].g / (1f - mu_prod_t);
                Tensor vHat = v[i] / (1f - MathF.Pow(beta2, t));

                Tensor.CopyTo(parameters[i].theta - gamma * mHat / (vHat.Sqrt() + epsilon), parameters[i].theta);
            });

        }
    }

}