using System;
using System.Linq;
using Unity.VisualScripting;

namespace DeepUnity
{
    public class ActionBuffer
    {
        /// <summary>
        /// A vector of length <em>Discrete Branches</em> containing values in range [0, <em>branch value - 1]</em>
        /// </summary>
        public int[] DiscreteActions { get; set; }
        /// <summary>
        /// A vector of Length <em>Continuous Actions</em> containing values in range [-1, 1]
        /// </summary>
        public float[] ContinuousActions { get; set; }

        public ActionBuffer(int continuousDim, int[] discreteBranches)
        {
            if (continuousDim < 0)
                throw new ArgumentException("Cannot have a negative number of continuous actions");

            if (discreteBranches != null && discreteBranches.Any(x => x < 2))
                throw new ArgumentException("Cannot have a discrete branch that have less than 2 different actions.");

            ContinuousActions = new float[continuousDim];
            DiscreteActions = discreteBranches == null? new int[0] : new int[discreteBranches.Length];           
        }
        public void Clear()
        {
            DiscreteActions = DiscreteActions?.Select(x => -1).ToArray();
            ContinuousActions = ContinuousActions?.Select(x => float.NaN).ToArray();
        }
        public override string ToString()
        {
            return $"[ContinuousActions [{ContinuousActions?.ToCommaSeparatedString()}] | DiscreteActions [{DiscreteActions?.ToCommaSeparatedString()}]]";
        }
    }
}

