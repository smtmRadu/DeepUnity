
namespace DeepUnity
{
    public interface IActivation
    {
        public Tensor Activation { get; }
        public Tensor Derivative { get; }
        
    }
}


