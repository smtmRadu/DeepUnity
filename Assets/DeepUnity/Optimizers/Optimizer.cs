using System.Linq;
using System.Threading.Tasks;
using DeepUnity.Modules;

namespace DeepUnity.Optimizers
{
    // Note that parameters and gradients value should not be reassigned elsewhere, only the values of the tensors inside.
    // Note that LazyLayers parameters must be infered before initializing the optimizer.
    public abstract class Optimizer
    {
        protected Parameter[] parameters;

        /// <summary>
        /// learning rate
        /// </summary>
        public float gamma;
        /// <summary>
        /// regularization strength
        /// </summary>
        protected float lambda;
        /// <summary>
        /// numerical stability value
        /// </summary>
        protected float epsilon;
        /// <summary>
        /// step counter
        /// </summary>
        protected int t;

        protected Optimizer(Parameter[] parameters, float lr, float eps, float weightDecay)
        {
            this.parameters = parameters;
            gamma = lr;
            lambda = weightDecay;
            epsilon = eps;
            t = 0;
        }
        public abstract void Step();

        /// <summary>
        /// Resets all gradients of all <see cref="Parameter"/>s to 0.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var param in parameters)
            {
                Tensor.CopyTo(Tensor.Zeros(param.g.Shape), param.g);
            }
        }
        /// <summary>
        /// Clips the gradients of all <see cref="Parameter"/>s in the range [-clip_value, clip_value]
        /// </summary>
        public void ClipGradValue(float clip_value)
        {
            foreach (var param in parameters)
            {
                Tensor.CopyTo(param.g.Clip(-clip_value, clip_value), param.g);
            }
        }
        /// <summary>
        /// Gradient Clipping by norm for all <see cref="Parameter"/>s. If <paramref name="max_norm"/> = 0, no changes are made.
        /// </summary>
        /// <param name="max_norm">If is 0, the clipping is aborted.</param>
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

            float c = max_norm / norm[0];

            foreach (var param in parameters)
            {
                Tensor.CopyTo(param.g * c, param.g);
            }
        }

    }
}
