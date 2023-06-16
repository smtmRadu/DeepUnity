using System;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {
        public Tensor Predict(Tensor input) => null;
        public Tensor Forward(Tensor input) => null;
        public Tensor Backward(Tensor loss) => null;

    }

}