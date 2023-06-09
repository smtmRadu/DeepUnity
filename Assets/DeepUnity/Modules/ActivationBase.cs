
namespace DeepUnity
{
    public abstract class ActivationBase
    {
        private Tensor InputCache { get; set; }
        protected abstract void Activation(Tensor x);
        protected abstract void Derivative(Tensor x);

        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            Activation(input);
            return input;

        }
        public Tensor Backward(Tensor loss)
        {
            Derivative(InputCache);
            return InputCache * loss;
        }
    }
}


