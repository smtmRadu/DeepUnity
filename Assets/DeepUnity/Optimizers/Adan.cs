using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    // The implementation is took from the paper https://arxiv.org/pdf/2208.06677.pdf
    // The restart condition for momentum is not introduced for simplicity, had to be checked for each parameter separaterly and it didn't worth the effort
    // lr = sqrt(batch_size / 256) * 6.25e-3 for lr scale depending on the batch sizes (check what Adan authors did for details in appendix)
    public class Adan : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly float beta3;

        private readonly Tensor[] m;
        private readonly Tensor[] v;
        private readonly Tensor[] n;

        private readonly Tensor[] gOld;

        /// <summary>
        /// ADAptive Nesterov momentum optimizer.<br></br>
        /// Paper uses lr = 1e-2, with weight decay in [1e-3, 1e-2] range, and 3e-4 in RL with 0 weight decay.
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="beta3"></param>
        /// <param name="weightDecay"></param>
        public Adan(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.02f, float beta2 = 0.08f, float beta3 = 0.01f, float eps = 1e-7f, float weightDecay = 0f)
            : base(parameters, lr, eps, weightDecay)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.beta3 = beta3;

            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];
            n = new Tensor[parameters.Length];
            gOld = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].g.Shape);
                v[i] = Tensor.Zeros(parameters[i].g.Shape);
                n[i] = Tensor.Zeros(parameters[i].g.Shape);
                gOld[i] = Tensor.Zeros(parameters[i].g.Shape);
            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {

                if (t == 1)
                {
                    m[i] = parameters[i].g.Clone() as Tensor;
                    n[i] = parameters[i].g.Pow(2f);
                    // v[i] = 0 by default from init
                }

                if (t == 2)
                {
                    v[i] = parameters[i].g - gOld[i];
                }

                m[i] = (1 - beta1) * m[i] + beta1 * parameters[i].g;
                v[i] = (1 - beta2) * v[i] + beta2 * (parameters[i].g - gOld[i]);
                n[i] = (1 - beta3) * n[i] + beta3 * (parameters[i].g + (1f - beta2) * (parameters[i].g - gOld[i])).Pow(2);
                Tensor eta = Tensor.Fill(gamma, n[i].Shape) / (n[i].Sqrt() + epsilon);

                // Update theta
                Tensor.CopyTo((lambda * eta + 1f).Pow(-1f) * (parameters[i].param - eta * (m[i] + (1f - beta2) * v[i])), parameters[i].param);


                gOld[i] = parameters[i].g.Clone() as Tensor;

            });
        }
    }
}

