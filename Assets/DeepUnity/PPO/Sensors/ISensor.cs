using System.Collections;

namespace DeepUnity
{
    public interface ISensor
    {
        public IEnumerable GetObservations();
    }
    public enum World
    {
        World3d,
        World2d,
    }
    public enum CaptureType
    {
        RGB,
        Greyscale,
    }
}

