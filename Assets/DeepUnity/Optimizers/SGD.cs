using DeepUnity.Modules;
using UnityEngine;
namespace DeepUnity.Optimizers
{
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    public class SGD : Optimizer
    {
        [SerializeField] private float mu;
        [SerializeField]private float tau;
        [SerializeField]private bool nesterov;

        [SerializeField] private Tensor[] m;

        /// <summary>
        /// Stochastic Gradient Descent.
        /// </summary>
        /// <param name="parameters"></param>
        /// <param name="lr"></param>
        /// <param name="momentum"></param>
        /// <param name="weightDecay"></param>
        /// <param name="dampening">Reduces the effect of momentum.</param>
        /// <param name="nesterov"></param>
        /// <param name="maximize"></param>
        public SGD(Parameter[] parameters, float lr = 0.01f, float momentum = 0.9f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false, bool maximize = false) 
            : base(parameters, lr, 0, weightDecay, maximize)
        {
            this.mu = momentum;
            this.tau = dampening;
            this.nesterov = nesterov;

            if(mu != 0)
            {
                m = new Tensor[parameters.Length];

                for (int i = 0; i < parameters.Length; i++)
                {
                    m[i] = Tensor.Zeros(parameters[i].param.Shape);
                }
            }          
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].param, parameters[i].g);
                
                
                if (mu != 0)
                {
                    if (t > 1)
                        Tensor.CopyTo(mu * m[i] + (1f - tau) * parameters[i].g, m[i]);
                    else
                        Tensor.CopyTo(parameters[i].g, m[i]);
                        
                
                    if (nesterov)
                        Tensor.CopyTo(parameters[i].g + mu * m[i], parameters[i].g);
                    else
                        Tensor.CopyTo(m[i], parameters[i].g);
                }
                
                if (maximize)
                    Tensor.CopyTo(parameters[i].param + gamma * parameters[i].g, parameters[i].param);
                else
                    Tensor.CopyTo(parameters[i].param - gamma * parameters[i].g, parameters[i].param);
            });
        }

    }

}
