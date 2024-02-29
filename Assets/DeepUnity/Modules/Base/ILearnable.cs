
namespace DeepUnity.Layers
{
    public interface ILearnable
    {
        public Parameter[] Parameters();
        public int ParametersCount();
        public void SetDevice(Device device);
    }

}


