using System;
using System.Linq;
using Unity.VisualScripting;

namespace DeepUnity
{
    public class ActionBuffer
    {
        /// <summary>
        /// The index of the discrete action of value in range [0, <em>Discrete Actions - 1]</em>. If no Discrete Actions are used, this will be equal to -1.
        /// </summary>
        public int DiscreteAction { get; set; }
        /// <summary>
        /// A vector of Length <em>Continuous Actions</em> containing values in range [-1, 1]. If no Continuous Actions are used, this array is null.
        /// </summary>
        public float[] ContinuousActions { get; set; }

        public ActionBuffer(int continuousDim, int discreteDim)
        {
            if (continuousDim < 0)
                throw new ArgumentException("Cannot have a negative number of continuous actions");

            if (discreteDim < 0)
                throw new ArgumentException("Cannot have a negative number of discrete actions");

            ContinuousActions = new float[continuousDim];
            DiscreteAction = -1;          
        }
        public void Clear()
        {
            DiscreteAction = -1;
            ContinuousActions = ContinuousActions?.Select(x => 0f).ToArray();
        }
        public override string ToString()
        {
            return $"[Continuous Actions [{ContinuousActions?.ToCommaSeparatedString()}] | Discrete Action [{DiscreteAction}]]";
        }
    }
}

