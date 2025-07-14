using DeepUnity.Modules;
using System.Threading.Tasks;

namespace DeepUnity.Optimizers
{
    // https://arxiv.org/pdf/2302.06675.pdf
    public class Lion : Optimizer
    {
        private readonly bool cautious; // check https://arxiv.org/pdf/2411.16085 for cautious optimizers
        private readonly float beta1;
        private readonly float beta2;

        private readonly Tensor[] m;

        public Lion(Parameter[] parameters, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.99f, float weightDecay = 0f, bool cautious = false)
            : base(parameters, lr, 0, weightDecay, false)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.cautious = cautious;
            m = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].param.Shape);
            }

           
        }

        public override void Step()
        {
            t++;

            Parallel.For(0, parameters.Length, i =>
            {
                Tensor g = parameters[i].g;
                Tensor c = Tensor.Sign(beta1 * m[i] + (1f - beta1) * g);

                if (cautious)
                {
                    Tensor phi_t = (c * g).Select(x => x >= 0 ? x : 0f); // alginment mask
                    Tensor gamma_scaled = Tensor.Fill(gamma, phi_t.Shape) * (phi_t.Count() / (phi_t.Norm(NormType.NonZeroL0)[0] + 1f));
                    Tensor theta_t = parameters[i].param - gamma_scaled * (phi_t * c + lambda * g);
                    Tensor.CopyTo(theta_t, parameters[i].param);
                }           
                else
                {
                    // update model parameters
                    Tensor theta_t = parameters[i].param - gamma * (c + lambda * g);
                    Tensor.CopyTo(theta_t, parameters[i].param);
                }


                // update ema of gt
                m[i] = beta2 * m[i] + (1f - beta2) * g;
            });
        }

    }

}
