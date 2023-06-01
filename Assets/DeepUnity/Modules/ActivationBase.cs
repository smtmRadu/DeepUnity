
namespace DeepUnity
{
    public abstract class ActivationBase
    {
        protected abstract Tensor Activation(Tensor x);
        protected abstract Tensor Derivative(Tensor x);
        
    }
}


