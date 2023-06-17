
namespace DeepUnity
{
    public class SensorBuffer
    {
        private Tensor observations;
        private int position;
       
 	    public SensorBuffer(int capacity)
        {
            observations = Tensor.Zeros(capacity);
            position = 0;
        }
        public void AddObservation(float observation)
        {
            if (Capacity - position < 1)
                throw new System.Exception("SensorBuffer is full.");

            observations[position++] = observation;
        }
        public void Clear()
        {
            observations.ForEach(x => 0f);
            position = 0;
        }
        public override string ToString()
        {
            return $"(Observations {observations})";

        }
        public int Capacity => observations.Shape.width;
    }
}

