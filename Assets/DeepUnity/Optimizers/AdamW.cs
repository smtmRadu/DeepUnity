using System.Threading.Tasks;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Optimizers
{
    [System.Serializable]
    public sealed class AdamW : Optimizer
    {
       
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private bool amsgrad;
        [SerializeField] private bool fused;
        [SerializeField] private bool cautious; // check https://arxiv.org/pdf/2411.16085 for cautious optimizers
        // note that cautious fused adamw cannot be implemented because each param is modified individually.
        
        [SerializeField] private float beta1_t = 1f; // beta1^t caching
        [SerializeField] private float beta2_t = 1f;
        
        // 1st momentum buffer
        [SerializeField] private Tensor[] m;
        
        // 2nd momentum buffer 
        [SerializeField] private Tensor[] v;
        [SerializeField] private Tensor[] vHatMax;
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
        /// <param name="cautious">Either use cautious version of AdamW or not (Cautious Optimizers: Improving Training with One Line of Code).<br></br> <b> It is not available for fused AdamW</b></param>
        public AdamW(
            Parameter[] parameters, 
            float lr = 0.001f, 
            float beta1 = 0.9f, 
            float beta2 = 0.999f, 
            float eps = 1e-8f, 
            float weight_decay = 0.01f, 
            bool amsgrad = false, 
            bool cautious = false, 
            bool maximize = false, 
            bool fused = true) 
            : base(parameters, lr, eps, weight_decay,maximize)
        {
            this.amsgrad = amsgrad;
            this.maximize = maximize;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.fused = fused;
            this.cautious = cautious;

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

                // apply decoupled wd
                if(!cautious)
                    Tensor.CopyTo(parameters[i].param - gamma * lambda * parameters[i].param, parameters[i].param);

                Tensor.CopyTo(beta1 * m[i] + (1f - beta1) * parameters[i].g, m[i]);
                Tensor.CopyTo(beta2 * v[i] + (1f - beta2) * parameters[i].g.Square(), v[i]);

                Tensor mHat = m[i] / (1f - beta1_t);
                Tensor vHat = v[i] / (1f - beta2_t);

                if (amsgrad)
                {
                    if(cautious) // same as without ams grad but vhatmax takes place of vhat
                    {
                        vHatMax[i] = Tensor.Maximum(vHatMax[i], vHat); 
                        Tensor u_t = mHat / (vHatMax[i].Sqrt() + epsilon);
                        Tensor phi_t = (u_t * parameters[i].g).Select(x => x >= 0 ? x : 0f); // alginment mask
                        Tensor gamma_scaled = Tensor.Fill(gamma, phi_t.Shape) * (phi_t.Count() / (phi_t.Norm(NormType.NonZeroL0)[0] + 1f));
                        Tensor.CopyTo(parameters[i].param - gamma_scaled * (phi_t * u_t + lambda * parameters[i].g), parameters[i].param);

                    }
                    else
                    {
                        vHatMax[i] = Tensor.Maximum(vHatMax[i], vHat);
                        Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHatMax[i].Sqrt() + epsilon), parameters[i].param);
                    }       
                }
                else
                {
                    if(cautious)
                    {
                        Tensor u_t = mHat / (vHat.Sqrt() + epsilon);
                        Tensor phi_t = (u_t * parameters[i].g).Select(x => x >= 0 ? x : 0f); // alginment mask
                        Tensor gamma_scaled = Tensor.Fill(gamma, phi_t.Shape) * (phi_t.Count() / (phi_t.Norm(NormType.NonZeroL0)[0] + 1f));
                        Tensor.CopyTo(parameters[i].param - gamma_scaled * (phi_t * u_t + lambda * parameters[i].g), parameters[i].param);
                    }
                    else
                        Tensor.CopyTo(parameters[i].param - gamma * mHat / (vHat.Sqrt() + epsilon), parameters[i].param);
                }
                             
            });
        }
    }

    // 3 dense model test (128 hidden size)
    // amsgrad = 2.22s
    // amsgrad + fused = 2.19
    // amsgrad + fused + parallel = 2.08s

    

}