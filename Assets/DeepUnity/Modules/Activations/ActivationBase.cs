
namespace DeepUnity
{
    public abstract class ActivationBase : IModule
    {
        private Tensor InputCache { get;set; }
        protected abstract Tensor Activation(Tensor x);
        protected abstract Tensor Derivative(Tensor y);

        public Tensor Predict(Tensor input)
        {
            return Activation(input);
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            return Activation(input);

        }
        public Tensor Backward(Tensor loss)
        {
            return Derivative(InputCache) * loss;
        }
    }
}


