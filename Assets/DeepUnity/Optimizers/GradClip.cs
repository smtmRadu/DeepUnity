using System.Linq;
namespace DeepUnity.Optimizers
{
    public abstract partial class Optimizer
    {
        /// <summary>
        /// Clips the gradients of all <see cref="Parameter"/>s in the range [-clip_value, clip_value]
        /// </summary>
        public void ClipGradValue(float clip_value)
        {
            foreach (var param in parameters)
            {
                if (param.Device == Device.CPU)
                    Tensor.CopyTo(param.g.Clip(-clip_value, clip_value), param.g);
                else
                    TensorGPU.Clip_(param.gGPU, -clip_value, clip_value);
            }
        }
        /// <summary>
        /// Gradient Clipping by norm for all <see cref="Parameter"/>s. If <paramref name="max_norm"/> is equal or less than 0, no changes are made.
        /// </summary>
        /// <param name="max_norm">If is 0, the clipping is aborted.</param>
        /// <param name="eps">Value for numerical stability if computing <see cref="NormType.EuclideanL2"/>.</param>
        public void ClipGradNorm(float max_norm, NormType normType = NormType.EuclideanL2, float eps = 1e-12f)
        {
            if (max_norm <= 0)
                return;

            int no_params = parameters.Sum(x =>
            {
                if (x.Device == Device.CPU)
                    return x.param.Count();
                else
                    return x.paramGPU.Count();
            });

            // Concatenate all gradients in a single tensor vector
            Tensor vector = Tensor.Zeros(no_params);
            int index = 0;
            foreach (var grad_t in parameters)
            {
                if (grad_t.Device == Device.CPU)
                    foreach (var grad in grad_t.g.ToArray())
                    {
                        vector[index++] = grad;
                    }
                else
                    foreach (var grad in grad_t.gGPU.ToArray())
                    {
                        vector[index++] = grad;
                    }
            }

            // Compute norm
            Tensor norm = Tensor.Norm(vector, normType, eps);

            if (norm[0] <= max_norm)
                return;

            float c = max_norm / norm[0];

            foreach (var param in parameters)
            {
                if (param.Device == Device.CPU)
                    Tensor.CopyTo(param.g * c, param.g);
                else
                    TensorGPU.Multiply_(param.gGPU, c);
            }
        }
    }

}


