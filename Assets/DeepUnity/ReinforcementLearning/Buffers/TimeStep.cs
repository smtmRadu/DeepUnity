

using System;

namespace DeepUnity
{
    public class TimeStep : ICloneable
    {
        public Tensor state;
        public Tensor action_continuous;      
        public Tensor action_discrete;    
        public Tensor reward;
        public Tensor piold_continuous;
        public Tensor piold_discrete;
        public Tensor advantage;
        public Tensor value_target;
      
        public object Clone()
        {
            TimeStep clone = new TimeStep();
            clone.state = state;
            clone.action_continuous = action_continuous;
            clone.action_discrete = action_discrete;
            clone.reward = reward;

            clone.piold_continuous = piold_continuous;
            clone.piold_discrete = piold_discrete;
            clone.advantage = advantage;
            clone.value_target = value_target;

            return clone;
        }
    }
}




