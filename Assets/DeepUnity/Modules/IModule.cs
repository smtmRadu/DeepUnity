namespace DeepUnity
{
    public interface IModule
    {
        public Tensor InputCache { get; set; }
        public Tensor Forward(Tensor input);
        public Tensor Backward(Tensor loss);
    }
}
