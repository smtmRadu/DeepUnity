
namespace DeepUnity
{
    public abstract class ActivationBase
    {
        protected abstract Tensor InputCache { get; set; }
        protected abstract void Activation(Tensor x);
        protected abstract void Derivative(Tensor x);
    }
}


