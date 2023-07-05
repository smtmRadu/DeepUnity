using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace DeepUnity
{
    public class TrajectoryBuffer
    {
        public bool reachedTerminalState = true;
        public List<Tensor> states;
        public List<Tensor> values;
        public List<Tensor> rewards;

        public List<Tensor> continuous_actions;
        public List<Tensor> continuous_log_probs;
        public List<Tensor> discrete_actions;
        public List<Tensor> discrete_log_probs;

        public List<Tensor> advantages;
        public List<Tensor> returns;

        public TrajectoryBuffer()
        {
            states = new List<Tensor>();
            values = new List<Tensor>();
            rewards = new List<Tensor>();
            continuous_actions = new List<Tensor>();
            continuous_log_probs = new List<Tensor>();
            discrete_actions = new List<Tensor>();
            discrete_log_probs= new List<Tensor>();

            advantages = new List<Tensor>();
            returns = new List<Tensor>();
        }
        
        public void Remember(Tensor state,  Tensor value, Tensor reward, Tensor continuos_action, Tensor cont_log_prob, Tensor discrete_action, Tensor disc_log_prob)
        {
            states.Add(state);
            values.Add(value);
            rewards.Add(reward);
            continuous_actions.Add(continuos_action);
            continuous_log_probs.Add(cont_log_prob);
            discrete_actions.Add(discrete_action);
            discrete_log_probs.Add(disc_log_prob);
        }

       
        public void Reset()
        {
            reachedTerminalState = true;
            states.Clear();
            values.Clear();
            rewards.Clear();
            continuous_actions.Clear();
            continuous_log_probs.Clear();
            discrete_actions.Clear();
            discrete_log_probs.Clear();
            returns.Clear();
            advantages.Clear();

        }

        public int Count { get => states.Count; }
        public float CumulativeReward { get => rewards.Sum(x => x[0]); }

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Trajectory ({Count} steps) | Total Reward {CumulativeReward} | Reached Terminal State {reachedTerminalState}");
            sb.AppendLine("{");
            for (int i = 0; i < Count; i++)
            {
                sb.AppendLine($"\tFrame {i} | Value: {values[i][0]} | Reward: {rewards[i][0]} | Advantage: {advantages[i][0]} | Return: {returns[i][0]}");
            }
            sb.AppendLine("}");
            return sb.ToString();

        }
    }
}

