
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class SensorBuffer
    {
        public int Capacity => Observations.Length;

        public float[] Observations;
        private int position_index;
       
 	    public SensorBuffer(int capacity)
        {
            Observations = Enumerable.Repeat(float.NaN, capacity).ToArray();
            position_index = 0;
        }
        public void Clear()
        {
            Observations = Enumerable.Repeat(float.NaN, Observations.Length).ToArray();
            position_index = 0;
        }
        public override string ToString()
        {
            return $"[Observations [{Observations.ToCommaSeparatedString()}]]";

        }



        // Overloads
        public void AddObservation(float observation)
        {
            if (Capacity - position_index < 1)
                throw new System.InsufficientMemoryException($"SensorBuffer overflow. Please add observations considering a capacity of {Capacity}.");

            Observations[position_index++] = observation;
        }
        public void AddObservation(int observation)
        {
            AddObservation((float)observation);
        }     
        public void AddObservation(Vector2 observation2)
        {
            AddObservation(observation2.x);
            AddObservation(observation2.y);
        }
        public void AddObservation(Vector3 observation3)
        {
            AddObservation(observation3.x);
            AddObservation(observation3.y);
            AddObservation(observation3.z);
        }
        public void AddObservation(Quaternion observation4)
        {
            AddObservation(observation4.x);
            AddObservation(observation4.y);
            AddObservation(observation4.z);
            AddObservation(observation4.w);
        }
        public void AddObservation(IEnumerable observationsN)
        {
            IEnumerable<float> castedObservationCollection = observationsN.Cast<float>();

            if (Capacity - position_index < castedObservationCollection.Count())
                throw new System.InsufficientMemoryException($"SensorBuffer available space is {Capacity - position_index}. IEnumerable<float> observations is too large.");

            foreach (var item in castedObservationCollection)
            {
                AddObservation(item);
            }
        }  
       
    }
}

