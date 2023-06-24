
using System.Collections;
using System.Collections.Generic;
using System.Linq;

namespace DeepUnity
{
    public class SensorBuffer
    {
        public float[] values;
        private int position;
       
 	    public SensorBuffer(int capacity)
        {
            values = Enumerable.Repeat(float.NaN, capacity).ToArray();
            position = 0;
        }
        public void Clear()
        {
            values = Enumerable.Repeat(float.NaN, values.Length).ToArray();
            position = 0;
        }
        public override string ToString()
        {
            return $"(Observations {values})";

        }
        public int Capacity => values.Length;


        // Overloads
        public void AddObservation(float observation)
        {
            if (Capacity - position < 1)
                throw new System.InsufficientMemoryException("SensorBuffer is full.");

            values[position++] = observation;
        }
        public void AddObservation(IEnumerable observationsCollection)
        {
            IEnumerable<float> castedObservationCollection = observationsCollection.Cast<float>();

            if (Capacity - position < castedObservationCollection.Count())
                throw new System.InsufficientMemoryException($"SensorBuffer available space is {Capacity - position}. IEnumerable<float> observations is too large.");

            foreach (var item in castedObservationCollection)
            {
                AddObservation(item);
            }
        }  
    }
}

