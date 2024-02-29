using System;

namespace DeepUnity.ReinforcementLearning
{
    public class TimestepTuple : ICloneable
    {
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

        public Tensor reward;
        public Tensor done { get; set; }

        public Tensor advantage;
        public Tensor value_target;
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
            clone.reward = reward;
            clone.prob_continuous = prob_continuous;
            clone.prob_discrete = prob_discrete;
            clone.advantage = advantage;
            clone.value_target = value_target;
            clone.done = done;

            return clone;
        }
    }
}




