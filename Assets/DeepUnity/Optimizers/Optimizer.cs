using System.Linq;

namespace DeepUnity
{
    // Note that parameters and gradients value should not be reassigned elsewhere, only the values of the tensors inside
    public abstract class Optimizer
    {
        protected Parameter[] parameters;

        public float gamma;
        protected float lambda;
        protected int t; // step counter

        protected Optimizer(Parameter[] parameters, float lr, float weightDecay)
        {
            this.parameters = parameters;
            this.gamma = lr;
            this.lambda = weightDecay;
            t = 0;       
        }
        public abstract void Step();

        /// <summary>
        /// Resets all gradients of a <see cref="Learnable"/> layer to 0.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var param in parameters)
            {
                Tensor.CopyTo(Tensor.Zeros(param.g.Shape), param.g);
            }
        }
        /// <summary>
        /// Clips the gradients of a <see cref="Learnable"/> layer in the range [-clip_value, clip_value]
        /// </summary>
        public void ClipGradValue(float clip_value)
        {
            foreach (var param in parameters)
            {
                Tensor.CopyTo(param.g.Clip(-clip_value, clip_value), param.g);
            }
        }
        /// <summary>
        /// Computes the clip grad norm globally over all <see cref="Learnable"/> layers. If <paramref name="max_norm"/> = 0, no changes are made.
        /// </summary>
        /// <param name="max_norm">If is 0, nothing is changed</param>
        public void ClipGradNorm(float max_norm, NormType normType = NormType.EuclideanL2)
        {
            if (max_norm == 0)
                return;

            int no_params = parameters.Sum(x => x.theta.Count());

            // Concatenate all gradients in a single tensor vector
            Tensor vector = Tensor.Zeros(no_params);
            int index = 0;
            foreach (var grad_t in parameters)
            {
                foreach (var grad in grad_t.g.ToArray())
                {
                    vector[index++] = grad;
                }
            }

            // Compute norm
            Tensor norm = Tensor.Norm(vector, normType);

            if (norm[0] <= max_norm)
                return;

            float scale = max_norm / norm[0];

            foreach (var param in parameters)
            {
                Tensor.CopyTo(param.g * scale, param.g);
            }     
        }
        
    }
}
