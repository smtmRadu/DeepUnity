namespace DeepUnity
{
    public class Conv2D : IModule
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