using DeepUnity.Layers;
namespace DeepUnity.Optimizers
{
    // https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    public class RMSProp : Optimizer
    {
        private readonly float alpha;
        private readonly float mu;
        private readonly bool centered;

        private readonly Tensor[] v;
        private readonly Tensor[] b;
        private readonly Tensor[] gAve;

        public RMSProp(Parameter[] parameters, float lr = 0.01f, float alpha = 0.99f, float eps = 1e-7f, float weightDecay = 0f, float momentum = 0f, bool centered = false) : base(parameters, lr, eps, weightDecay)
        {
            this.alpha = alpha;
            mu = momentum;
            this.centered = centered;

            v = new Tensor[parameters.Length];
            b = new Tensor[parameters.Length];
            gAve = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                v[i] = Tensor.Zeros(parameters[i].g.Shape);
                b[i] = Tensor.Zeros(parameters[i].g.Shape);
                gAve[i] = Tensor.Zeros(parameters[i].g.Shape);
            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].theta, parameters[i].g);

                v[i] = alpha * v[i] + (1f - alpha) * parameters[i].g.Pow(2f);

                var tilde_v = v[i].Clone() as Tensor;

                if (centered)
                {
                    gAve[i] = gAve[i] * alpha + (1f - alpha) * parameters[i].g;
                    tilde_v = tilde_v - gAve[i].Pow(2f);
                }

                if (mu > 0f)
                {
                    b[i] = mu * b[i] + parameters[i].g / (tilde_v.Sqrt() + epsilon);
                    Tensor.CopyTo(parameters[i].theta - gamma * b[i], parameters[i].theta);
                }
                else
                    Tensor.CopyTo(parameters[i].theta - gamma * parameters[i].g / (tilde_v.Sqrt() + epsilon), parameters[i].theta);

            });

        }
    }

}


