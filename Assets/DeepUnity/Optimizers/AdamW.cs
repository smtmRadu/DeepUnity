using System.Threading.Tasks;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    public sealed class AdamW : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly bool amsgrad;
        private readonly bool maximize;

        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;
        private readonly Tensor[] vHatMax;

        public AdamW(Parameter[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-7f, float weightDecay = 0.01f, bool amsgrad = false, bool maximize = false) : base(parameters, lr, eps, weightDecay)
        {
            this.amsgrad = amsgrad;
            this.maximize = maximize;
            this.beta1 = beta1;
            this.beta2 = beta2;

            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];

            if (amsgrad)
                vHatMax = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].theta.Shape);
                v[i] = Tensor.Zeros(parameters[i].theta.Shape);

                if (amsgrad)
                    vHatMax[i] = Tensor.Zeros(parameters[i].theta.Shape);           
            }
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            Parallel.For(0, parameters.Length, i =>
            {
                if (maximize)
                    Tensor.CopyTo(-parameters[i].g, parameters[i].g);

                Tensor.CopyTo(parameters[i].theta - gamma * lambda * parameters[i].theta, parameters[i].theta);

                m[i] = beta1 * m[i] + (1f - beta1) * parameters[i].g;
                v[i] = beta2 * v[i] + (1f - beta2) * parameters[i].g.Pow(2f);
                Tensor mHat = m[i] / (1f - beta1_t);
                Tensor vHat = v[i] / (1f - beta2_t);

                if (amsgrad)
                {
                    vHatMax[i] = Tensor.Maximum(vHatMax[i], vHat);
                    Tensor.CopyTo(parameters[i].theta - gamma * mHat / (vHatMax[i].Sqrt() + epsilon), parameters[i].theta);
                }
                else
                    Tensor.CopyTo(parameters[i].theta - gamma * mHat / (vHat.Sqrt() + epsilon), parameters[i].theta);            
            });
        }
    }
}