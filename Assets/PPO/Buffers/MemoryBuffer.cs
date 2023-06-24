namespace DeepUnity
{
    public class MemoryBuffer
    {
        public Tensor[] states;
        public Tensor[] actions;
        public Tensor[] continuous_log_probs;
        public Tensor[] discrete_log_probs;
        public Tensor[] values;
        public Tensor[] rewards;
        public Tensor[] dones;

        public int index;

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
        public void StoreContinuous(float[] state, float[] action, float log_prob, float value, float reward, bool done)
        {
            if (index == states.Length)
                throw new System.Exception("MemoryBuffer is full.");

            states[index] = Tensor.Constant(state);
            actions[index] = Tensor.Constant(action);
            continuous_log_probs[index] = Tensor.Constant(log_prob);
            values[index] = Tensor.Constant(value);
            rewards[index] = Tensor.Constant(reward);
            dones[index] = Tensor.Constant(done == true? 1 : 0);
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

