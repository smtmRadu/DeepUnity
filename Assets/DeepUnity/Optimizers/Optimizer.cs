using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.Optimizers
{
    // Note that parameters and gradients value should not be reassigned elsewhere, only the values of the tensors inside.
    // Note that LazyLayers parameters must be infered before initializing the optimizer.
    [System.Serializable]
    public abstract partial class Optimizer
    {
        public Parameter[] parameters;

        /// <summary>
        /// the description of this optimizer. Used as a metadata when serialized.
        /// </summary>
        [SerializeField] public string description;
        /// <summary>
        /// learning rate
        /// </summary>
        [SerializeField] public float gamma;
        /// <summary>
        /// regularization strength
        /// </summary>
        [SerializeField] protected float lambda;
        /// <summary>
        /// numerical stability value
        /// </summary>
        [SerializeField] protected float epsilon;
        /// <summary>
        /// either maximize or minimize loss.
        /// </summary>
        [SerializeField] protected bool maximize;
        /// <summary>
        /// step counter
        /// </summary>
        [SerializeField] protected int t;
        


        protected Optimizer(Parameter[] parameters, float lr, float eps, float weight_decay, bool maximize)
        {
            this.parameters = parameters;
            gamma = lr;
            lambda = weight_decay;
            epsilon = eps;
            this.maximize = maximize;
            this.description = this.GetType().Name;
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