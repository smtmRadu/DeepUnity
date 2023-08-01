namespace DeepUnity
{
    public abstract class Activation : IModule
    {
        protected Tensor InputCache { get;set; }
        protected abstract Tensor Activate(Tensor x);
        protected abstract Tensor Derivative(Tensor y);

        public Tensor Predict(Tensor input)
        {
            return Activate(input);
        }
        public Tensor Forward(Tensor input)
        {
            InputCache = Tensor.Identity(input);
            return Activate(input);

        }
        public Tensor Backward(Tensor loss)
        {
            return Derivative(InputCache) * loss;
        }
    }
}


