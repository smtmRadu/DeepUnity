using DeepUnity.Modules;
using UnityEngine;
namespace DeepUnity.Optimizers
{
    // The implementation is took from the paper https://arxiv.org/pdf/2208.06677.pdf
    // The restart condition for momentum is not introduced for simplicity, had to be checked for each parameter separaterly and it didn't worth the effort
    // lr = sqrt(batch_size / 256) * 6.25e-3 for lr scale depending on the batch sizes (check what Adan authors did for details in appendix)
    public class Adan : Optimizer
    {
        [SerializeField]private float beta1;
        [SerializeField]private float beta2;
        [SerializeField] private float beta3;

        [SerializeField]private Tensor[] m;
        [SerializeField]private Tensor[] v;
        [SerializeField] private Tensor[] n;

        [SerializeField] private Tensor[] gOld;

        /// <summary>
        /// ADAptive Nesterov momentum optimizer.<br></br>
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="beta1"></param>
        /// <param name="beta2"></param>
        /// <param name="beta3"></param>
        /// <param name="weight_decay"></param>
        public Adan(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.02f, float beta2 = 0.08f, float beta3 = 0.01f, float eps = 1e-8f, float weight_decay = 0f)
            : base(parameters, lr, eps, weight_decay, false)
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
                    n[i] = parameters[i].g.Square();
                    // v[i] = 0 by default from init
                }

                if (t == 2)
                {
                    v[i] = parameters[i].g - gOld[i];
                }

                m[i] = (1f - beta1) * m[i] + beta1 * parameters[i].g;
                v[i] = (1f - beta2) * v[i] + beta2 * (parameters[i].g - gOld[i]);
                n[i] = (1f - beta3) * n[i] + beta3 * (parameters[i].g + (1f - beta2) * (parameters[i].g - gOld[i])).Square();
                Tensor eta = gamma / (n[i].Sqrt() + epsilon);

                // Update theta
                Tensor.CopyTo((1f + lambda * eta).Pow(-1f) * (parameters[i].param - eta * (m[i] + (1f - beta2) * v[i])), parameters[i].param);
                Tensor.CopyTo(parameters[i].g, gOld[i]);
            });
        }
    }
}

