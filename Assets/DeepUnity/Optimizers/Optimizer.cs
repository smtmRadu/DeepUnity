using DeepUnity.Modules;

namespace DeepUnity.Optimizers
{
    // Note that parameters and gradients value should not be reassigned elsewhere, only the values of the tensors inside.
    // Note that LazyLayers parameters must be infered before initializing the optimizer.
    public abstract partial class Optimizer
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

        protected Optimizer(Parameter[] parameters, float lr, float eps, float weight_decay)
        {
            this.parameters = parameters;
            gamma = lr;
            lambda = weight_decay;
            epsilon = eps;
            t = 0;
        }
        public abstract void Step();

        /// <summary>
        /// Reset all gradients of all <see cref="Parameter"/>s to 0.
        /// </summary>
        public void ZeroGrad()
        {
            foreach (var param in parameters)
            {
                if (param.Device == Device.CPU)
                    Tensor.CopyTo(Tensor.Zeros(param.g.Shape), param.g);
                else
                    TensorGPU.Zero_(param.gGPU);
            }
        }   

    }
}
