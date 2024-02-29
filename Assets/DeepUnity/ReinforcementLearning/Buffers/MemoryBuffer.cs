using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Personal experience buffer of a single agent.
    /// </summary>
    public class MemoryBuffer
    {
        public int Count { get => frames.Count; }
        public List<TimestepTuple> frames { get; private set; } = new();

        public void Add(TimestepTuple timestep)
        {
            frames.Add(timestep);
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

                if (frames[i].value_target != null)
                    sb.Append($" | V[t]: {frames[i].value_target[0].ToString("0.000")}");

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
