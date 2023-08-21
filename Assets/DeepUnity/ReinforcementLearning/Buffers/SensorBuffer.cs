
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class SensorBuffer
    {
        public int Capacity => Observations.Count();

        public Tensor Observations;
        private int position_index;
       
 	    public SensorBuffer(int capacity)
        {
            Observations = Tensor.Zeros(capacity).Select(x => float.NaN);
            position_index = 0;
        }
        public void Clear()
        {
            Observations = Observations.Select(x => float.NaN);
            position_index = 0;
        }
        public override string ToString()
        {
            return $"[Observations [{Observations.ToArray().ToCommaSeparatedString()}]]";

        }



        // Overloads
        /// <summary>
        /// Adds a one dimensional observation. Base method for all other AddObservation() methods.
        /// </summary>
        /// <param name="observation"></param>
        public void AddObservation(float observation)
        {
            if (Capacity - position_index < 1)
                throw new System.InsufficientMemoryException($"SensorBuffer overflow. Please add observations considering a capacity of {Capacity}.");

            Observations[position_index++] = observation;
        }
        /// <summary>
        /// Adds a one dimensional observation. The bool value is converted to float, 0 for false and 1 for true respectively.
        /// </summary>
        /// <param name="observation"></param>
        public void AddObservation(bool observation)
        {
            AddObservation(observation ? 1f : 0f);
        }
        /// <summary>
        /// Adds a one dimensional observation.
        /// </summary>
        /// <param name="observation"></param>
        public void AddObservation(int observation)
        {
            AddObservation(observation);
        }
        /// <summary>
        /// Adds an observation vector of length 2.
        /// </summary>
        /// <param name="observation2"></param>
        public void AddObservation(Vector2 observation2)
        {
            AddObservation(observation2.x);
            AddObservation(observation2.y);
        }
        /// <summary>
        /// Adds an observation vector of length 3.
        /// </summary>
        /// <param name="observation3"></param>
        public void AddObservation(Vector3 observation3)
        {
            AddObservation(observation3.x);
            AddObservation(observation3.y);
            AddObservation(observation3.z);
        }
        /// <summary>
        /// Adds an observation vector of length 4.
        /// </summary>
        /// <param name="observation4"></param>
        public void AddObservation(Quaternion observation4)
        {
            AddObservation(observation4.x);
            AddObservation(observation4.y);
            AddObservation(observation4.z);
            AddObservation(observation4.w);
        }
        /// <summary>
        /// Adds an observation vector of length <b>N</b>.
        /// </summary>
        /// <param name="observationsN"></param>
        /// <exception cref="System.InsufficientMemoryException"></exception>
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
        /// <summary>
        /// Adds a one hot encoded observation in a vector of length <paramref name="typesNumber"/>.
        /// </summary>
        /// <param name="index"> Must be in range [0, typesNumber)</param>
        /// <param name="typesNumber">Must be in range [2, n]</param>
        public void AddOneHotObservation(int index, int typesNumber)
        {
            if(index >= typesNumber || index < 0)
                throw new System.IndexOutOfRangeException($"Index ({index}) is out of range. Must be greater than 0 and less than typesNumber ({typesNumber})!");

            if (typesNumber < 2)
                throw new System.ArgumentException($"Types number ({typesNumber}) cannot be less than 2!");
            
            for (int i = 0; i < typesNumber; i++)
            {
                if (i == index)
                    AddObservation(1);
                else
                    AddObservation(0);
            }
        }
       
    }
}

