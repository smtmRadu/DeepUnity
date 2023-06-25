using System.ComponentModel.Design;
using System.Text;

namespace DeepUnity
{
    public class MemoryBuffer
    {
        public Tensor[] states;
        public Tensor[] values;
        public Tensor[] rewards;
        public Tensor[] dones;

        public Tensor[] continuous_actions;
        public Tensor[] continuous_log_probs;
        public Tensor[] discrete_actions;     
        public Tensor[] discrete_log_probs;

        public int indexPosition;

        public MemoryBuffer(int capacity)
        {
            states = new Tensor[capacity];
            continuous_actions = new Tensor[capacity];
            continuous_log_probs = new Tensor[capacity];
            discrete_actions = new Tensor[capacity];
            discrete_log_probs= new Tensor[capacity];
            values = new Tensor[capacity];
            rewards = new Tensor[capacity];
            dones = new Tensor[capacity];

            indexPosition = 0;
        }
        public void Store(Tensor state, Tensor cont_action, Tensor disc_action, Tensor cont_log_probs, Tensor disc_log_probs, Tensor value, Tensor reward, Tensor done)
        {
            if (indexPosition == states.Length)
                throw new System.Exception("MemoryBuffer is full.");

            states[indexPosition] = state;
            values[indexPosition] = value;
            rewards[indexPosition] = reward;
            dones[indexPosition] = done;

            continuous_actions[indexPosition] = cont_action;
            continuous_log_probs[indexPosition] = cont_log_probs;
            discrete_actions[indexPosition] = disc_action;
            discrete_log_probs[indexPosition] = disc_log_probs;

            indexPosition++;
        }
        public void Clear()
        {
            for (int i = 0; i < indexPosition; i++)
            {
                states[i] = null;
                continuous_actions[i] = null;
                continuous_log_probs[i] = null;
                discrete_log_probs[i] = null;
                values[i] = null;
                rewards[i] = null;
                dones[i] = null;
            }

            indexPosition = 0;
        }
        public bool IsFull() => indexPosition == states.Length;
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < indexPosition; i++)
            {
                sb.AppendLine($"\nFrame {i}\n(State: {states[i]}\nAction: {continuous_actions[i]}\nValue: {values[i]}\nReward: {rewards[i]}\nDone: {dones[i]}");
            }
            return sb.ToString();
           
        }
        public string ToShortString()
        {
            StringBuilder sb = new StringBuilder();
            for (int i = 0; i < indexPosition; i++)
            {
                sb.AppendLine($"Frame {i} [{states.GetHashCode()}]");
            }
            return sb.ToString();
        }
    }
}

