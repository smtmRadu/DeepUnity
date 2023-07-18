using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    public class TrajectoryBuffer
    {
        public int Count { get => states.Count; }
        public float CumulativeReward { get => rewards.Sum(x => x[0]); }

        public List<Tensor> states;
        public List<Tensor> values;
        public List<Tensor> rewards;
        public List<Tensor> continuous_actions;
        public List<Tensor> continuous_log_probs;
        public List<Tensor> discrete_actions;
        public List<Tensor> discrete_log_probs;

        public List<Tensor> advantages;
        public List<Tensor> returns;

        public bool reachedTerminalState = true;

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
        
        public void Remember(TimeStep t)
        {
            states.Add(t.state);
            values.Add(t.value);
            rewards.Add(t.reward);
            continuous_actions.Add(t.continuous_action);
            continuous_log_probs.Add(t.continuous_log_prob);
            discrete_actions.Add(t.discrete_action);
            discrete_log_probs.Add(t.discrete_log_prob);
            t.Reset();
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

        public override string ToString()
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"Trajectory ({Count} steps) | Total Cumulated Reward {CumulativeReward} | Reached Terminal State {reachedTerminalState}");
            sb.AppendLine("{");
            for (int i = 0; i < Count; i++)
            {
                sb.AppendLine(
                    $"\tFrame {i} " +
                    $"| Reward: {rewards[i][0]} " +
                    $"| Return: {returns[i][0]} " +
                    $"| Advantage: {advantages[i][0]} " +                
                    $"| Value: {values[i][0]}");
            }
            sb.AppendLine("}");
            return sb.ToString();

        }
    }

    public class TimeStep
    {
        public Tensor state;
        public Tensor continuous_action;
        public Tensor continuous_log_prob;
        public Tensor discrete_action;
        public Tensor discrete_log_prob;
        public Tensor reward;
        public Tensor value;


        public void Reset()
        {
            state = null;
            continuous_action = null;
            continuous_log_prob = null;
            discrete_action = null;
            discrete_log_prob = null;
            reward = null;
            value = null;
        }
    }
}

