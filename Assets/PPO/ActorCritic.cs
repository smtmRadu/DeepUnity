using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    public class ActorCritic : ScriptableObject
    {
 	    public new string name;
        public int observationSize;
        public int actionSize;

        public HyperParameters hp;
        public NeuralNetwork critic;
        public NeuralNetwork muHead;

        public List<NeuralNetwork> discreteHeads;

    }
    public enum ActionType
    {
        Continuous,
        Discrete
    }
}

