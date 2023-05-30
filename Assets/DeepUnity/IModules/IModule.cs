namespace DeepUnity
{
    public interface IModule
    {
        public Tensor<float> InputCache { get; set; }
        public Tensor<float> Forward(Tensor<float> input);
        public Tensor<float> Backward(Tensor<float> loss);
    }
}
