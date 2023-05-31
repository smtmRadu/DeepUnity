
namespace DeepUnity
{
    public class Dropout : IModule
    {
        public Tensor InputCache { get; set; }
        public Tensor Forward(Tensor input)
        {
            return null;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }
    }

}
