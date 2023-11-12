using System;

namespace DeepUnity
{
    public class TimestepBuffer : ICloneable
    {
        public int index;

        public Tensor state;
        public Tensor nextState;

        public Tensor action_continuous;
        public Tensor prob_continuous;      
        public Tensor action_discrete;
        public Tensor prob_discrete;

        public Tensor reward;
        public Tensor done;

        public Tensor advantage;
        public Tensor value_target;
        public Tensor q_target;
       
      
        public TimestepBuffer(int index)
        {
            this.index = index;
            done = Tensor.Zeros(1);
            reward = Tensor.Zeros(1);
        }
        private TimestepBuffer() { }
        public object Clone()
        {
            TimestepBuffer clone = new TimestepBuffer();

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




