using UnityEngine;
using DeepUnity.Modules;

namespace DeepUnity.Optimizers
{
    /// <summary>
    /// Note that paper algorithm of Adamax is missing an epsilon, thus we wuse pytorch algorithm for this,
    /// </summary>
    public class Adamax : Optimizer
    {
        [SerializeField] private readonly float beta1;
        [SerializeField] private readonly float beta2;

        [SerializeField] private float beta1_t = 1f; // beta1^t caching

        private readonly Tensor[] m;
        private readonly Tensor[] u;

        public Adamax(Parameter[] parameters, float lr = 0.002f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-7f, float weight_decay = 0f) : base(parameters, lr, eps, weight_decay, false)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;

            m = new Tensor[parameters.Length];
            u = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].g.Shape);
                u[i] = Tensor.Zeros(parameters[i].g.Shape);
            }

        }



        public override void Step()
        {
            t++;

            beta1_t *= beta1;


            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].param, parameters[i].g);

                // Update biased first momentum estimate
                m[i] = beta1 * m[i] + (1f - beta1) * parameters[i].g;

                // Update the exponentially weighted infinity norm
                u[i] = Tensor.Maximum(u[i] * beta2, parameters[i].g.Abs() + epsilon);

                // Update parameters 
                Tensor.CopyTo(parameters[i].param - gamma * m[i] / (u[i] * (1 - beta1_t)), parameters[i].param);
            });
        }
    }
}



