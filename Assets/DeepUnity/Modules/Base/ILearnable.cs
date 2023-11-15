
namespace DeepUnity
{
    public interface ILearnable
    {
        public Tensor[] Parameters();
        public Tensor[] Gradients();
        public int ParametersCount();
        public void SetDevice(Device device);
    }

}


