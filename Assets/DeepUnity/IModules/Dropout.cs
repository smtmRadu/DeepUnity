
namespace DeepUnity
{
    public class Dropout : IModule
    {
        public Tensor<float> InputCache { get; set; }
        public Tensor<float> Forward(Tensor<float> input)
        {
            return null;
        }
        public Tensor<float> Backward(Tensor<float> loss)
        {
            return null;
        }
    }

}
