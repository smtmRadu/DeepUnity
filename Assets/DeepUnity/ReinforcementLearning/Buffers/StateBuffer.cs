
using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// All values added to the state buffer are clipped in range [-5, 5]
    /// </summary>
    public class StateBuffer
    {
        public readonly int Capacity;
        public Tensor State { get; set; }
        private int position_index { get; set; }
       
 	    public StateBuffer(int capacity)
        {
            State = Tensor.Zeros(capacity).Select(x => 0f);
            this.Capacity = capacity;
            position_index = 0;
        }
        public void Clear()
        {
            for (int i = 0; i < Capacity; i++)
            {
                State[i] = 0;
            }
            position_index = 0;
        }
        public bool IsFulfilled(out int missingObs)
        {
            missingObs = Capacity - position_index;
            return position_index == Capacity;
        }
        public override string ToString()
        {
            return $"[Observations [{State.ToArray().ToCommaSeparatedString()}]]";

        }



        // Overloads
        /// <summary>
        /// Adds a one dimensional observation, clipped between [-10, 10] for stability. Base method for all other AddObservation() methods.
        /// </summary>
        /// <param name="observation"></param>
        public void AddObservation(float observation)
        {
            if (Capacity - position_index < 1)
                throw new InsufficientMemoryException($"SensorBuffer overflow. Consider the capacity is {Capacity}.");

            if (float.IsNaN(observation))
            {
                ConsoleMessage.Warning("float.NaN value observation added to the SensorBuffer replaced with 0");
                observation = 0f;
            }
            State[position_index++] = Utils.Clip(observation, -5f, 5f);
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
            AddObservation((float)observation);
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
        public void AddObservationRange(IEnumerable observationsN)
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

