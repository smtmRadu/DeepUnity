using DeepUnity.Modules;
using System;
using UnityEngine;
namespace DeepUnity.Optimizers
{
    // https://pytorch.org/docs/stable/generated/torch.optim.RAdam.html#torch.optim.RAdam
    public class RAdam : Optimizer
    {
        [SerializeField]private float beta1;
        [SerializeField]private float beta2;
        [SerializeField] private bool decoupledWeightDecay;


        [SerializeField]private float beta1_t = 1f; // beta1^t caching
        [SerializeField]private float beta2_t = 1f; // beta2^t caching
        [SerializeField] private float rho_inf;

        // 1st momentum buffer
        [SerializeField] private Tensor[] m;

        // 2nd momentum buffer 
        [SerializeField] private Tensor[] v;

        public RAdam(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f, float weightDecay = 0f, bool decoupled_weight_decay = false)
            : base(parameters, lr, eps, weightDecay, false)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.decoupledWeightDecay = decoupled_weight_decay;
            this.rho_inf = 2f / (1f - beta2) - 1f;

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


            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                {
                    if(decoupledWeightDecay)
                        Tensor.CopyTo(parameters[i].param - gamma *lambda * parameters[i].param, parameters[i].param);
                    else
                        Tensor.CopyTo(parameters[i].g + lambda * parameters[i].g, parameters[i].param); 
                }

                Tensor.CopyTo(beta1 * m[i] + (1f - beta1) * parameters[i].g, m[i]);
                Tensor.CopyTo(beta1 * v[i] + (1f - beta1) * parameters[i].g.Square(), v[i]);
                Tensor mHat = m[i] / (1 - beta1_t);
                float rho_t = rho_inf - 2f * t * beta2_t / (1 - beta2_t);
                if (rho_t > 5)
                {
                    Tensor l_t = MathF.Sqrt(1 - beta2_t) * Tensor.Reciprocal(v[i].Sqrt() + epsilon);
                    float r_t = MathF.Sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t));
                    Tensor.CopyTo(parameters[i].param - gamma * mHat * r_t * l_t, parameters[i].param);
                }
                else
                    Tensor.CopyTo(parameters[i].param - gamma * mHat, parameters[i].param);
            });
        }

    }   

}
