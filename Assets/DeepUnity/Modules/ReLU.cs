using System;

namespace DeepUnity
{
    [Serializable]
    public class ReLU : ActivationBase, IModule 
    {
        public Tensor InputCache { get; set; }
        protected override Tensor Activation(Tensor x) => x.Select(k => Math.Max(0, k));
        protected override Tensor Derivative(Tensor x) => x.Select(k => k > 0f ? 1f : 0f); 


        public Tensor Forward(Tensor input)
        {
            InputCache = input.Clone() as Tensor;
            return Activation(InputCache);
        }
        public Tensor Backward(Tensor loss)
        {
            return Derivative(InputCache) * loss;
        }
    }
}