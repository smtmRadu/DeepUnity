namespace DeepUnity
{
    public class Linear : IModule
    {
        public Tensor<float> InputCache { get; set; }
        public Tensor<float> Forward(Tensor<float> input) => input;
        public Tensor<float> Backward(Tensor<float> loss) => loss;
    }

}

