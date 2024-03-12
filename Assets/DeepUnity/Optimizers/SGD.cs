using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    public class SGD : Optimizer
    {
        private readonly float mu;
        private readonly float tau;
        private readonly bool nesterov;
        private readonly bool maximize;

        private readonly Tensor[] b;

        public SGD(Parameter[] parameters, float lr = 0.01f, float momentum = 0.9f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false, bool maximize = false) 
            : base(parameters, lr, 0, weightDecay)
        {
            this.mu = momentum;
            this.tau = dampening;
            this.nesterov = nesterov;
            this.maximize = maximize;

            b = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                b[i] = Tensor.Zeros(parameters[i].g.Shape);
            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].theta, parameters[i].g);
                

                if (mu != 0)
                {
                    if (t > 1)
                        b[i] = mu * b[i] + (1f - tau) * parameters[i].g;
                    else
                        b[i] = parameters[i].g.Clone() as Tensor;

                    if (nesterov)
                        Tensor.CopyTo(parameters[i].g + mu * b[i], parameters[i].g);
                    else
                        Tensor.CopyTo(b[i], parameters[i].g);
                }

                if (maximize)
                    Tensor.CopyTo(parameters[i].theta + gamma * parameters[i].g, parameters[i].theta);
                else
                    Tensor.CopyTo(parameters[i].theta - gamma * parameters[i].g, parameters[i].theta);
            });
        }

    }

}
