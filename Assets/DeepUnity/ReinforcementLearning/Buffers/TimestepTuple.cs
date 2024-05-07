using System;

namespace DeepUnity.ReinforcementLearning
{
    public class TimestepTuple : ICloneable
    {
        /// <summary>
        /// The index of the timestep in the episode
        /// </summary>
        public int index;

        /// <summary>
        /// (normalized) state
        /// </summary>
        public Tensor state;
        /// <summary>
        /// (normalized) next state
        /// </summary>
        public Tensor nextState;

        /// <summary>
        /// raw continuous actions, unsquashed by Tanh
        /// </summary>
        public Tensor action_continuous;
        public Tensor prob_continuous;
        /// <summary>
        /// one hot embedded of discrete action
        /// </summary>
        public Tensor action_discrete;
        public Tensor prob_discrete;

        /// <summary>
        /// Tensor(1) - positive or negative
        /// </summary>
        public Tensor reward;
        /// <summary>
        /// Tensor(1) - 1 if nextState is terminal, 0 otherwise
        /// </summary>
        public Tensor done { get; set; }

        /// <summary>
        /// Tensor(1) - positive or negative
        /// </summary>
        public Tensor advantage;
        /// <summary>
        /// Tensor(1) - The target for the Value function
        /// </summary>
        public Tensor v_target;
        /// <summary>
        /// Tensor(1) - The target for the Q function
        /// </summary>
        public Tensor q_target;


        public TimestepTuple(int index)
        {
            this.index = index;
            done = Tensor.Zeros(1);
            reward = Tensor.Zeros(1);
        }
        private TimestepTuple() { }
        public object Clone()
        {
            TimestepTuple clone = new TimestepTuple();

            clone.index = index;
            clone.state = state;
            clone.nextState = nextState;
            clone.action_continuous = action_continuous;
            clone.action_discrete = action_discrete;
     
            clone.prob_continuous = prob_continuous;
            clone.prob_discrete = prob_discrete;

            clone.reward = reward;
            clone.done = done;

            clone.advantage = advantage;
            clone.v_target = v_target;
            clone.q_target = q_target;         

            return clone;
        }
    }
}




