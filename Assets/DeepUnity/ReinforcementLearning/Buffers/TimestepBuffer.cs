using System;

namespace DeepUnity
{
    public class TimestepBuffer : ICloneable
    {
        public int index;
        public Tensor state;
        public Tensor action_continuous;      
        public Tensor action_discrete;    
        public Tensor reward;
        public Tensor log_probs_continuous;
        public Tensor log_probs_discrete;
        public Tensor advantage;
        public Tensor value_target;
        public Tensor done;
      
        public object Clone()
        {
            TimestepBuffer clone = new TimestepBuffer();


            clone.index = index;
            clone.state = state;
            clone.action_continuous = action_continuous;
            clone.action_discrete = action_discrete;
            clone.reward = reward;
            clone.log_probs_continuous = log_probs_continuous;
            clone.log_probs_discrete = log_probs_discrete;
            clone.advantage = advantage;
            clone.value_target = value_target;
            clone.done = done;

            return clone;
        }
    }
}



