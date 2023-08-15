using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class Trajectory
    {
        public int Count { get => frames.Count; }
        public float CumulativeReward { get => frames.Sum(x => x.reward[0]); }
        public Tensor[] States { get => frames.Select(x => x.state).ToArray(); }
        public Tensor[] ActionsContinuous { get => frames.Select(x => x.action_continuous).ToArray(); }
        public Tensor[] PIoldContinuous { get => frames.Select(x => x.piold_continuous).ToArray(); }
        public Tensor[] ActionsDiscrete { get => frames.Select(x => x.action_discrete).ToArray(); }
        public Tensor[] PIoldDiscrete { get => frames.Select(x => x.piold_discrete).ToArray(); }
        public Tensor[] Rewards { get => frames.Select(x => x.reward).ToArray(); }
        public Tensor[] ValueTargets { get => frames.Select(x => x.value_target).ToArray(); }
        public Tensor[] Advantages { get => frames.Select(x => x.value_target).ToArray(); }


        public bool reachedTerminalState = true;
        List<TimeStep> frames;

        public Trajectory()
        {
            frames = new List<TimeStep>();
        }

        public void Add(TimeStep timestep)
        {
            frames.Add(timestep.Clone() as TimeStep);
        }
        public void Reset()
        {
            reachedTerminalState = true;
            frames.Clear();
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

                if (frames[i].action_continuous != null)
                {
                    temp = frames[i].action_continuous;
                    frames[i].action_continuous = frames[r].action_continuous;
                    frames[r].action_continuous = temp;

                    temp = frames[i].piold_continuous;
                    frames[i].piold_continuous = frames[r].piold_continuous;
                    frames[r].piold_continuous = temp;
                }
                                  
                if (frames[i].action_discrete != null)
                {
                    temp = frames[i].action_discrete;
                    frames[i].action_discrete = frames[r].action_discrete;
                    frames[r].action_discrete = temp;

                    temp = frames[i].piold_discrete;
                    frames[i].piold_discrete = frames[r].piold_discrete;
                    frames[r].piold_discrete = temp;
                }

                temp = frames[i].reward;
                frames[i].reward = frames[r].reward;
                frames[r].reward = temp;

                temp = frames[i].advantage;
                frames[i].advantage = frames[r].advantage;
                frames[r].advantage = temp;

                temp = frames[i].value_target;
                frames[i].value_target = frames[r].value_target;
                frames[r].value_target = temp;
            }
        }




        public void ComputeAdvantagesAndVTargets(float gamma, float lambda, Sequential valueNetwork)
        {
            int T = Count;

            for (int timestep = 0; timestep < T; timestep++)
            {
                float discount = 1f;
                Tensor v_t = Tensor.Constant(0);

                for (int t = timestep; t < T; t++)
                {
                    v_t += discount * frames[t].reward;
                    discount *= gamma;
                }

                // V(s[t]) = 0 if the agent reached the terminal state (because there are no future rewards after this)
                if (!reachedTerminalState)
                {
                    v_t += discount * valueNetwork.Predict(frames[T - 1].state);
                }

                Tensor a_t = v_t - valueNetwork.Predict(frames[timestep].state);

                frames[timestep].value_target = v_t;
                frames[timestep].advantage = a_t;
            }
        }
        public void NormAdvantages()
        {
            float mean = frames.Average(x => x.advantage[0]);
            float var = frames.Sum(x => (x.advantage[0] - mean) * (x.advantage[0] - mean)) / (Count - 1);
            float std = MathF.Sqrt(var);
            for (int i = 0; i < Count; i++)
            {
                frames[i].advantage = (frames[i].advantage - mean) / (std + Utils.EPSILON);
            }
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Trajectory | Steps {Count} | Total Cumulated Reward {CumulativeReward} | Reached Terminal State {reachedTerminalState}");
            sb.AppendLine("{");
            for (int i = 0; i < Count; i++)
            {

                sb.Append($"\tTimestep {i.ToString("000")}");
                sb.Append($" | s[t]: [{frames[i].state.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | a[t]: [{frames[i].action_continuous.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | r[t]: {frames[i].reward[0].ToString("0.000")}");
                sb.Append($" | PIold(a[t],s[t]): [{frames[i].piold_continuous.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | A[t]: {frames[i].advantage[0].ToString("0.000")}");
                sb.Append($" | VTarget[t]: {frames[i].value_target[0].ToString("0.000")}");
                
                
                sb.Append("\n");
            }
            sb.AppendLine("}");
            return sb.ToString();
        }
    }
}
