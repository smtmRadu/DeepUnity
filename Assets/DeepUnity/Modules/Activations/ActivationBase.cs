
namespace DeepUnity
{
    public abstract class ActivationBase : IModule
    {
        private NDArray InputCache { get; set; }
        protected abstract void Activation(NDArray x);
        protected abstract void Derivative(NDArray x);

        public NDArray Predict(NDArray input)
        {
            Activation(input);
            return input;
        }
        public NDArray Forward(NDArray input)
        {
            InputCache = NDArray.Identity(input);
            Activation(input);
            return input;

        }
        public NDArray Backward(NDArray loss)
        {
            Derivative(InputCache);
            return InputCache * loss;
        }
    }
}


