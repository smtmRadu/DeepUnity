using System;
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
        /// Gradient Clipping by norm for all <see cref="Parameter"/>s. If <paramref name="max_norm"/> is equal to or less than 0, no changes are made.
        /// </summary>
        /// <param name="max_norm">If this value is <= 0, the clipping is aborted.</param>
        /// <param name="eps">Value for numerical stability when computing <see cref="NormType.EuclideanL2"/>.</param>
        /// <param name="global">Indicates whether the norm is calculated globally or per-parameter.</param>
        /// <returns>The total norm of the parameter's gradients.</returns>
        public float ClipGradNorm(float max_norm, NormType normType = NormType.EuclideanL2,  float eps = 1e-12f, bool global = true)
        {
            if(global)
            {
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
                Tensor global_norm = Tensor.Norm(vector, normType, eps);

                // APPLY NORM HERE
                if (max_norm > 0f && global_norm[0] > max_norm)
                {
                    float c = max_norm / global_norm[0];

                    foreach (var param in parameters)
                    {
                        if (param.Device == Device.CPU)
                            Tensor.CopyTo(param.g * c, param.g);
                        else
                            TensorGPU.Multiply_(param.gGPU, c);
                    }
                }

                return global_norm[0];
            }
            else
            {
                foreach (var param in parameters)
                {
                    if (param.Device == Device.CPU)
                        Tensor.CopyTo(param.g * param.g.Norm(norm: normType, eps: eps), param.g);
                    else
                        throw new NotImplementedException("Local gradient clipping by norm not implemented for GPU params");
                }
                return -1f; 
            }
        }

        /// <summary>
        /// Compute the norm of all gradients.
        /// The norm is computed over the norms of the individual gradient tensors, as if the norms of the individual tensors were concatenated into a single vector.
        /// </summary>
        /// <param name="normType"></param>
        /// <param name="eps"></param>
        /// <returns></returns>
        public float GetTotalNorm(NormType normType = NormType.EuclideanL2, float eps=1e-12f)
        {
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

            return Tensor.Norm(vector, normType, eps)[0];
        }
    }

}


