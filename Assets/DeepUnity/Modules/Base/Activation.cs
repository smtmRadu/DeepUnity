using Unity.VisualScripting;

namespace DeepUnity
{
    /// <summary>
    /// Base class for all <see cref="IModule"/> layers that are non-linear activation functions.
    /// </summary>
    public abstract class Activation : IModule
    {
        protected Tensor InputCache { get;set; }
        protected abstract Tensor Activate(Tensor x);
        protected abstract Tensor Derivative(Tensor y);

        public virtual Tensor Predict(Tensor input)
        {
            return Activate(input);
        }
        public virtual Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            return Activate(input);

        }
        public virtual Tensor Backward(Tensor loss)
        {
            return Derivative(InputCache) * loss;
        }
        public abstract object Clone();
    }
}


