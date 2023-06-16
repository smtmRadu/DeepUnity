
namespace DeepUnity
{
    public abstract class ActivationBase : IModule
    {
        private Tensor InputCache;
        protected abstract void Activation(ref Tensor x);
        protected abstract void Derivative(ref Tensor x);

        public Tensor Predict(Tensor input)
        {
            Activation(ref input);
            return input;
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            Activation(ref input);
            return input;

        }
        public Tensor Backward(Tensor loss)
        {
            Derivative(ref InputCache);
            return InputCache * loss;
        }
    }
}


