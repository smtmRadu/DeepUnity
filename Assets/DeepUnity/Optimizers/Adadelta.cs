namespace DeepUnity
{
    public class Adadelta : Optimizer
    {
        private readonly float rho;

        private readonly Tensor[] v; // sqr avg
        private readonly Tensor[] u; // accumulate variables
        public Adadelta(Parameter[] parameters, float lr = 1f, float rho = 0.9f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {
            this.rho = rho;

            v = new Tensor[parameters.Length];
            u = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                v[i] = Tensor.Zeros(parameters[i].g.Shape);
                u[i] = Tensor.Zeros(parameters[i].g.Shape);
            }



        }
        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (lambda != 0)
                    Tensor.CopyTo(parameters[i].g + lambda * parameters[i].theta, parameters[i].g);

                v[i] = v[i] * rho + parameters[i].g.Pow(2f) * (1f - rho);

                var delta_x = Tensor.Sqrt(u[i] + Utils.EPSILON) / Tensor.Sqrt(v[i] + Utils.EPSILON) * parameters[i].g;

                u[i] = u[i] * rho + delta_x.Pow(2f) * (1f - rho);

                Tensor.CopyTo(parameters[i].theta - gamma * delta_x, parameters[i].theta);               
            });
        }
    }
}



