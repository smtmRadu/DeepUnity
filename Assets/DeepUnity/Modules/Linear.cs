using System;

namespace DeepUnity
{
    [Serializable]
    public class Linear : IModule
    {
        public Tensor InputCache { get; set; }
        public Tensor Forward(Tensor input) => input;
        public Tensor Backward(Tensor loss) => loss;
    }

}

