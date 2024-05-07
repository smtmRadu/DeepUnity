using System.Linq;
using System.Collections.Generic;
using System.Text;
using Unity.VisualScripting;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// The common buffer of the agents that holds all experiences across all parallel environments
    /// </summary>
    public class ExperienceBuffer
    {
        public readonly List<TimestepTuple> frames;
        public int Count { get => frames.Count; }
        public Tensor[] States { get => frames.Select(x => x.state).ToArray(); }
        public Tensor[] NextStates { get => frames.Select(x => x.nextState).ToArray(); }
        public Tensor[] ContinuousActions { get => frames.Select(x => x.action_continuous).ToArray(); }
        public Tensor[] ContinuousProbabilities { get => frames.Select(x => x.prob_continuous).ToArray(); }
        public Tensor[] DiscreteActions { get => frames.Select(x => x.action_discrete).ToArray(); }
        public Tensor[] DiscreteProbabilities { get => frames.Select(x => x.prob_discrete).ToArray(); }
        public Tensor[] ValueTargets { get => frames.Select(x => x.v_target).ToArray(); }
        public Tensor[] Advantages { get => frames.Select(x => x.advantage).ToArray(); }



        public ExperienceBuffer(int alloc_size)
        {
            frames = new List<TimestepTuple>(alloc_size);
        }
        public void Shuffle()
        {
            for (int i = Count - 1; i > 0; i--)
            {
                int r = Utils.Random.Range(0, Count);
                Tensor temp;

                temp = frames[i].state;
                frames[i].state = frames[r].state;
                frames[r].state = temp;

                temp = frames[i].nextState;
                frames[i].nextState = frames[r].nextState;
                frames[r].nextState = temp;

                if (frames[i].action_continuous != null)
                {
                    temp = frames[i].action_continuous;
                    frames[i].action_continuous = frames[r].action_continuous;
                    frames[r].action_continuous = temp;

                    temp = frames[i].prob_continuous;
                    frames[i].prob_continuous = frames[r].prob_continuous;
                    frames[r].prob_continuous = temp;
                }

                if (frames[i].action_discrete != null)
                {
                    temp = frames[i].action_discrete;
                    frames[i].action_discrete = frames[r].action_discrete;
                    frames[r].action_discrete = temp;

                    temp = frames[i].prob_discrete;
                    frames[i].prob_discrete = frames[r].prob_discrete;
                    frames[r].prob_discrete = temp;
                }

                temp = frames[i].reward;
                frames[i].reward = frames[r].reward;
                frames[r].reward = temp;

                temp = frames[i].advantage;
                frames[i].advantage = frames[r].advantage;
                frames[r].advantage = temp;

                temp = frames[i].v_target;
                frames[i].v_target = frames[r].v_target;
                frames[r].v_target = temp;

                temp = frames[i].done;
                frames[i].done = frames[r].done;
                frames[r].done = temp;
            }
        }
        /// <summary>
        /// Checks if the buffer is full. There can be no overflow.
        /// </summary>
        /// <param name="buffer_size"></param>
        /// <returns></returns>
        public bool IsFull(int buffer_size)
        {
            return Count == buffer_size;
        }
        /// <summary>
        /// Try to add agent's memory to the training data by the limit of the buffer_size. If is already full, the append fails.
        /// </summary>
        /// <param name="agentMemory"></param>
        /// <param name="buffer_size"></param>
        public void TryAppend(MemoryBuffer agentMemory, int buffer_size)
        {
            foreach (var frm in agentMemory.frames)
            {
                if (Count == buffer_size)
                    return;

                frames.Add(frm.Clone() as TimestepTuple);
            }
        }
        public void Clear()
        {
            frames.Clear();
        }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Trajectory ({Count})");
            sb.AppendLine("{");
            for (int i = 0; i < Count; i++)
            {
                sb.Append($" \t| {i.ToString("000")}");
                sb.Append($" | t: {frames[i].index.ToString("000")}");
                sb.Append($" | s[t]: [{frames[i].state.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | s'[t]: [{frames[i].nextState.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | a_cont[t]: [{frames[i].action_continuous?.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | a_disc[t]: [{frames[i].action_discrete?.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | r[t]: {frames[i].reward[0].ToString("0.000")}");

                if (frames[i].v_target != null)
                    sb.Append($" | V[t]: {frames[i].v_target[0].ToString("0.000")}");

                if (frames[i].advantage != null)
                    sb.Append($" | A[t]: {frames[i].advantage[0].ToString("0.000")}");

                if (frames[i].q_target != null)
                    sb.Append($" | Q[t]: {frames[i].q_target[0].ToString("0.000")}");

                sb.Append("\n");

                if (frames[i].done[0] == 1)
                    sb.Append("\n");
            }
            sb.AppendLine("}");
            return sb.ToString();
        }
    }

}


