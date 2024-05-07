using DeepUnity.Modules;
using System.Threading.Tasks;
namespace DeepUnity.Optimizers
{
    // https://arxiv.org/pdf/2302.06675.pdf
    public class Lion : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;

        private readonly Tensor[] m;

        public Lion(Parameter[] parameters, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.99f, float weightDecay = 0f)
            : base(parameters, lr, 0, weightDecay)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
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
                Tensor ct = beta1 * m[i] + (1f - beta1) * parameters[i].g;

                // update model parameters
                Tensor theta_t = parameters[i].param - gamma * (Tensor.Sign(ct) + lambda * parameters[i].g);
                Tensor.CopyTo(theta_t, parameters[i].param);

                // update ema of gt
                m[i] = beta2 * m[i] + (1f - beta2) * parameters[i].g;    
            });
        }

    }

}
