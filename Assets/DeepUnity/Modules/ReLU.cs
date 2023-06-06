using System;

namespace DeepUnity
{
    [Serializable]
    public class ReLU : ActivationBase, IModule 
    {
        protected override Tensor InputCache { get; set; }
        protected override void Activation(Tensor x) => x.ForEach(k => Math.Max(0f, k));
        protected override void Derivative(Tensor x) => x.ForEach(k => k > 0f ? 1f : 0f);

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