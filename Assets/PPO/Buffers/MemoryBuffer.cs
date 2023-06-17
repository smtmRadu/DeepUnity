namespace DeepUnity
{
    public class MemoryBuffer
    {
        Tensor[] states;
        Tensor[] actions;
        Tensor[] continuous_log_probs;
        Tensor[] discrete_log_probs;
        Tensor[] values;
        Tensor[] rewards;
        Tensor[] dones;

        int index;

        public MemoryBuffer(int capacity)
        {
            states = new Tensor[capacity];
            actions = new Tensor[capacity];
            continuous_log_probs = new Tensor[capacity];
            discrete_log_probs= new Tensor[capacity];
            values = new Tensor[capacity];
            rewards = new Tensor[capacity];
            dones = new Tensor[capacity];

            index = 0;
        }
        public void StoreContinuous(Tensor state, Tensor action, Tensor log_prob, Tensor value, Tensor reward, Tensor done)
        {
            if (index == states.Length)
                throw new System.Exception("MemoryBuffer is full.");

            states[index] = state;
            actions[index] = action;
            continuous_log_probs[index] = log_prob;
            values[index] = value;
            rewards[index] = reward;
            dones[index] = done;
            
            index++;
        }
        public void Clear()
        {
            for (int i = 0; i < index; i++)
            {
                states[i] = null;
                actions[i] = null;
                continuous_log_probs[i] = null;
                discrete_log_probs[i] = null;
                values[i] = null;
                rewards[i] = null;
                dones[i] = null;
            }

            index = 0;
        }
        public bool IsFull() => index == states.Length;
        public override string ToString()
        {
            return base.ToString();
        }
    }
}

