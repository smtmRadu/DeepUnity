using System.Threading.Tasks;
using DeepUnity.Modules;
namespace DeepUnity.Optimizers
{
    /// <summary>
    /// THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER https://arxiv.org/pdf/2409.03137 Matteo Pagliardini et al. 2024
    /// </summary>
    public sealed class AdEMAMix : Optimizer
    {
        private readonly float beta1;
        private readonly float beta2;
        private readonly float beta3;
        private readonly float alpha;
        private readonly bool amsgrad;
        private readonly bool decoupledWd;

        private float beta1_t = 1f; // beta1^t caching
        private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] m1;

        // 2nd momentum buffer 
        private readonly Tensor[] m2;

        private readonly Tensor[] nu;

        /// <summary>
        /// THE ADEMAMIX OPTIMIZER: BETTER, FASTER, OLDER https://arxiv.org/pdf/2409.03137 Matteo Pagliardini et al. 2024
        /// </summary>
        public AdEMAMix(Parameter[] parameters, float lr = 1e-3f, float beta1 = 0.9f, float beta2 = 0.999f, float beta3 = 0.9999f, float alpha = 5f, float eps = 1e-8F, float weight_decay = 0) : base(parameters, lr, eps, weight_decay, false)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.beta3 = beta3;
            this.alpha = alpha; 
            m1 = new Tensor[parameters.Length];
            m2 = new Tensor[parameters.Length];
            nu = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m1[i] = Tensor.Zeros(parameters[i].param.Shape);
                m2[i] = Tensor.Zeros(parameters[i].param.Shape);
                nu[i] = Tensor.Zeros(parameters[i].param.Shape);
            }
        }

        public override void Step()
        {
            t++;

            // Optional

            // Schedule beta_3
            // Schedule alpha

            beta1_t *= beta1;
            beta2_t *= beta2;

            Parallel.For(0, parameters.Length, i =>
            {
                // Update fast EMA
                Tensor.CopyTo(beta1 * m1[i] + (1f - beta1) * parameters[i].g, m1[i]);

                // Update slow EMA
                Tensor.CopyTo(beta3 * m2[i] + (1f - beta3) * parameters[i].g, m2[i]);

                // Update the second moment estimate
                Tensor.CopyTo(beta2 * nu[i] + (1f - beta2) * parameters[i].g.Pow(2f), nu[i]);

                // Apply bias correction
                Tensor m1Hat = m1[i] / (1f - beta1_t);
                Tensor nuHat = nu[i] / (1f - beta2_t);

                // Update Parameters
                Tensor.CopyTo(parameters[i].param - gamma * ((m1[i] + alpha * m2[i]) / (nu[i].Sqrt() + epsilon)), parameters[i].param);
            });
        }
    }
}