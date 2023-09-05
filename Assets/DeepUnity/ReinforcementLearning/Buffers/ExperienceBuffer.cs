using DeepUnity;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class ExperienceBuffer
    {
        public readonly int Capacity;
        public int Count { get; private set; }
        public float CumulativeReward { get => frames.Sum(x => x.reward[0]); }
        public Tensor[] States { get => frames.Select(x => x.state).ToArray(); }
        public Tensor[] ContinuousActions { get => frames.Select(x => x.action_continuous).ToArray(); }
        public Tensor[] ContinuousProbabilities { get => frames.Select(x => x.prob_continuous).ToArray(); }
        public Tensor[] DiscreteActions { get => frames.Select(x => x.action_discrete).ToArray(); }
        public Tensor[] DiscreteProbabilities { get => frames.Select(x => x.prob_discrete).ToArray(); }
        public Tensor[] ValueTargets { get => frames.Select(x => x.value_target).ToArray(); }
        public Tensor[] Advantages { get => frames.Select(x => x.advantage).ToArray(); }


        public TimestepBuffer[] frames { get; private set; }

        public ExperienceBuffer(int capacity)
        {
            this.Capacity = capacity;
            frames = new TimestepBuffer[capacity];
            Count = 0;
        }

        public void Add(TimestepBuffer timestep)
        {
            frames[Count] = timestep.Clone() as TimestepBuffer;
            Count++;
        }
        public void Reset()
        {
            for (int i = 0; i < frames.Length; i++)
                frames[i] = null;

            Count = 0;
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

                temp = frames[i].value_target;
                frames[i].value_target = frames[r].value_target;
                frames[r].value_target = temp;

                // There are not really neccesarry to be shuffled
                temp = frames[i].done;
                frames[i].done = frames[r].done;
                frames[r].done = temp;
            }
        }
        public bool IsFull() => Count == Capacity;
        public void GAE(in float gamma, in float lambda, in int horizon, NeuralNetwork crticNetwork)
        {
            int T = Count;
            Tensor[] Vw_s = new Tensor[T];
            for (int i = 0; i < T; i++)
                Vw_s[i] = crticNetwork.Predict(frames[i].state);

            for (int t = 0; t < T; t++)
            {
                float discount = 1f;
                Tensor v_t = Tensor.Constant(0);

                for (int t_i = t; t_i < T; t_i++)
                {
                    v_t += discount * frames[t_i].reward;
                    discount *= gamma;

                    if (frames[t_i].done[0] == 1)
                        break;

                    if(t_i - t == horizon || t_i == T - 1)
                    {
                        v_t += discount * Vw_s[T - 1];
                        break;
                    }
                }

                Tensor a_t = v_t - Vw_s[t];

                frames[t].value_target = v_t;
                frames[t].advantage = a_t;
            }

            // for (int timestep = 0; timestep < T; timestep++)
            // {
            //     float discount = 1f;
            //     Tensor a_t = Tensor.Constant(0);
            // 
            //     for (int t = timestep; t < T - 1; t++)
            //     {
            //         float mask = 1f - frames[t].done[0];
            //         Tensor d_t = frames[t].reward + gamma * Vw_s[t + 1] * mask - Vw_s[t];
            //         a_t += discount * d_t;
            //         discount *= gamma * lambda;
            //
            //         if (frames[t].done[0] == 1)
            //             break;
            //     }
            // 
            //     Tensor v_t = a_t + V_s[timestep];
            // 
            //     frames[timestep].value_target = v_t;
            //     frames[timestep].advantage = a_t;
            // }     

        }
        public void NormalizeAdvantages()
        {
            float mean = frames.Average(x => x.advantage[0]);
            float var = frames.Sum(x => (x.advantage[0] - mean) * (x.advantage[0] - mean)) / (Count - 1);
            float std = MathF.Sqrt(var) + Utils.EPSILON;

            for (int i = 0; i < Count; i++)
                frames[i].advantage = (frames[i].advantage - mean) / std;
        }
        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Trajectory ({Count})");
            sb.AppendLine("{");
            for (int i = 0; i < Count; i++)
            {
                sb.Append($" \t| {i.ToString("000")}");
                sb.Append($" | t: {(frames[i].index).ToString("000")}");
                sb.Append($" | s[t]: [{frames[i].state.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | a[t]: [{frames[i].action_continuous.ToArray().ToCommaSeparatedString()}]");
                sb.Append($" | r[t]: {frames[i].reward[0].ToString("0.000")}");
                sb.Append($" | VTarget[t]: {frames[i].value_target[0].ToString("0.000")}");
                sb.Append($" | A[t]: {frames[i].advantage[0].ToString("0.000")}");
                
                           
                sb.Append("\n");

                if (frames[i].done[0] == 1)
                    sb.Append("\n");
            }
            sb.AppendLine("}");
            return sb.ToString();
        }
    }
}
