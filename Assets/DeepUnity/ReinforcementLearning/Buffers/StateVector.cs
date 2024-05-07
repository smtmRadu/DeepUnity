using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public class StateVector
    {

        public readonly int Capacity;
        public readonly int StateSize;
        public readonly int StackedInputs;

        private LinkedList<float> StateSequenceVector;
        public Tensor State => Tensor.Constant(StateSequenceVector);

        public StateVector(int state_size, int stacked_states)
        {
            Capacity = state_size * stacked_states;
            StateSize = state_size;
            StackedInputs = stacked_states;
            ResetToZero();
        }
        public void ResetToZero()
        {
            StateSequenceVector = new LinkedList<float>();

            for (int i = 0; i < StateSize * StackedInputs; i++)
            {
                StateSequenceVector.AddLast(0f);
            }
        }
        /// <summary>
        /// Returns 0 if enough observations were added and no overflow.
        /// More than 0 otherwise.
        /// </summary>
        /// <returns></returns>
        public int GetOverflow()
        {
            return StateSequenceVector.Count % StateSize;
        }
        public override string ToString()
        {
            if(StackedInputs > 1)
                return $"[Observations ({StateSize}x{StackedInputs}) [{State.ToArray().ToCommaSeparatedString()}]]";
            else
                return $"[Observations ({StateSize}) [{State.ToArray().ToCommaSeparatedString()}]]";

        }



        // Overloads
        /// <summary>
        /// Adds a one dimensional observation, clipped between [-10, 10] for stability. Base method for all other AddObservation() methods.
        /// </summary>
        /// <param name="observation"></param>
        public void AddObservation(float observation)
        {
            if (StateSequenceVector.Count > Capacity)
                throw new InsufficientMemoryException($"StateBuffer overflow. Consider the state size is {StateSize}.");

            if (float.IsNaN(observation))
            {
                ConsoleMessage.Warning("Float.NaN observation replaced with 0");
                observation = 0f;
            }
            StateSequenceVector.RemoveFirst();
            StateSequenceVector.AddLast(observation);

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
        /// <exception cref="InsufficientMemoryException"></exception>
        public void AddObservation(IEnumerable<float> observationsN)
        {
            foreach (var item in observationsN)
            {
                AddObservation(item);
            }
        }
        /// <summary>
        /// Adds an observation vector of length <b>N</b>.
        /// </summary>
        /// <param name="observationsN"></param>
        /// <exception cref="InsufficientMemoryException"></exception>
        public void AddObservation(IEnumerable<int> observationsN)
        {
            foreach (var item in observationsN)
            {
                AddObservation(item);
            }
        }
        /// <summary>
        /// Adds an observation vector of length <b>N</b>.
        /// </summary>
        /// <param name="observationsN"></param>
        /// <exception cref="InsufficientMemoryException"></exception>
        public void AddObservation(IEnumerable<bool> observationsN)
        {
            foreach (var item in observationsN)
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
            if (index >= typesNumber || index < 0)
                throw new IndexOutOfRangeException($"Index ({index}) is out of range. Must be greater than 0 and less than typesNumber {typesNumber}!");

            if (typesNumber < 2)
                throw new ArgumentException($"Types number ({typesNumber}) cannot be less than 2!");

            for (int i = 0; i < typesNumber; i++)
            {
                if (i == index)
                    AddObservation(1f);
                else
                    AddObservation(0f);
            }
        }
    }
}

