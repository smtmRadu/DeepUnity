
namespace DeepUnity
{
    public interface IActivation
    {
        public Tensor<float> Activation { get; }
        public Tensor<float> Derivative { get; }
        
    }
}


