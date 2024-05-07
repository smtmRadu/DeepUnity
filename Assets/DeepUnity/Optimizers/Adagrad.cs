using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
    public class Adagrad : Optimizer
    {

        private readonly float eta;

        private readonly Tensor[] state_sum;

        public Adagrad(Parameter[] parameters, float lr = 0.01f, float lrDecay = 0f, float eps = 1e-7f, float weightDecay = 0f)
            : base(parameters, lr, eps, weightDecay)
        {
            eta = lrDecay;

            state_sum = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
                state_sum[i] = Tensor.Zeros(parameters[i].g.Shape);

        }

        public override void Step()
        {
            t++;


            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                float tilde_gamma = gamma / (1f + (t - 1f) * eta);

                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].param, parameters[i].g);


                state_sum[i] = state_sum[i] + parameters[i].g.Pow(2f);

                Tensor.CopyTo(parameters[i].param - tilde_gamma * parameters[i].g / (state_sum[i].Sqrt() + epsilon), parameters[i].param);
            });
        }
    }

}




