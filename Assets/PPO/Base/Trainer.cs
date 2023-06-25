using System.Collections.Generic;
using System.Linq;
using System.Text;
using Unity.VisualScripting;
using UnityEngine;

namespace DeepUnity
{
    public class Trainer : MonoBehaviour
    {
        private static Trainer Instance { get; set; }

        private List<Agent> agents;
        private int readyAgents;
        private HyperParameters hp;
        private ActorCritic net;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(this);
            }
            else
            {
                Instance = this;
            }
        }
        private void LateUpdate()
        {
            if (readyAgents == agents.Count)
                Train();

        }
        public static void Subscribe(Agent agent)
        {
            if(Instance == null)
            {
                GameObject go = new GameObject("Trainer");
                go.AddComponent<Trainer>();
                Instance.agents = new List<Agent>();
                Instance.readyAgents = 0;
                Instance.net = agent.network;
                Instance.hp = agent.Hp;
            }

            Instance.agents.Add(agent);
        }
        public static void Ready()
        {
            Instance.readyAgents++;
        }






        private void Train()
        {
            Debug.Log("TRAIN");

            foreach (var agent in agents)
            {
                // Unzip the memory
                Tensor[] states = agent.Memory.states;
                Tensor[] continuousActions = agent.Memory.continuous_actions;
                Tensor[] discreteActions = agent.Memory.discrete_actions;
                Tensor[] continuousLogProbs = agent.Memory.continuous_log_probs;
                Tensor[] discreteLogProbs = agent.Memory.discrete_log_probs;
                Tensor[] values = agent.Memory.values;
                Tensor[] rewards = agent.Memory.rewards;
                Tensor[] dones = agent.Memory.dones;

                // Compute advantages
                Tensor[] advantages = GeneralizedAdvantageEstimation(rewards, values, dones);


                // Debug only
                // StringBuilder sb = new StringBuilder();
                // for (int i = 0; i < advantages.Length; i++)
                // {
                //     sb.AppendLine($"Frame {i} Advantage: {advantages[i][0]}");
                // }
                // Utils.DebugInFile(sb.ToString());
                // Utils.DebugInFile(agent.Memory.ToString());

                // STEP Norm advantages
                // if(hp.normalize)
                // {
                //     int n = advantages.Length;
                //     float mean = advantages.Average(x => x[0]);
                //     float std = advantages.Sum(x => (x[0] - mean) / (n - 1));
                //     advantages = advantages.Select(x => (x[0] - mean) / std).Select(x => Tensor.Constant(x)).ToArray();
                // }

                // STEP Shuffle the data
                // System.Random rng = new System.Random(System.DateTime.Now.Millisecond);
                // 
                // for (int i = hp.bufferSize - 1; i > 0; i--)
                // {
                //     int r = rng.Next(i + 1);
                //     Utils.Swap(ref advantages[i], ref advantages[r]);
                //     Utils.Swap(ref states[i], ref states[r]);
                //     Utils.Swap(ref actions[i], ref actions[r]);
                //     Utils.Swap(ref logProbs[i], ref logProbs[r]);
                // }

                int noBatches = (int)(hp.bufferSize / (float)hp.batchSize);
                for (int e = 0; e < hp.numEpoch; e++)
                {                 
                    for (int n = 0; n < noBatches; n++)
                    {
                        int start = n * hp.batchSize;
                        
                        Tensor[] states_batch_arr = (Tensor[]) Utils.GetRange(states, start, hp.batchSize);
                        Tensor[] cont_action_batch_arr = (Tensor[])Utils.GetRange(continuousActions, start, hp.batchSize);
                        Tensor[] disc_action_batch_arr = (Tensor[])Utils.GetRange(discreteActions, start, hp.batchSize);
                        Tensor[] cont_logProbs_batch_arr = (Tensor[]) Utils.GetRange(continuousLogProbs, start, hp.batchSize);
                        Tensor[] disc_logProbs_batch_arr = (Tensor[])Utils.GetRange(discreteLogProbs, start, hp.batchSize);
                        Tensor[] advantages_batch_arr = (Tensor[])Utils.GetRange(advantages, start, hp.batchSize);


                        Tensor states_b = Tensor.Join(TDim.height, states_batch_arr);
                        Tensor adv_b = Tensor.Join(TDim.height, advantages_batch_arr);
                        
                        UpdateCriticNetwork(states_b, adv_b);
                        UpdateContinuousNetwork(states_b, cont_action_batch_arr, cont_logProbs_batch_arr, adv_b);
                        UpdateDiscreteNetwork(states_b, disc_action_batch_arr, disc_logProbs_batch_arr, adv_b);
                    
                    }
                }

                agent.Memory.Clear();
            }

            // net.Save();
            readyAgents = 0;
        }      


        private void UpdateCriticNetwork(Tensor states, Tensor advantages)
        {
            // [batch, values] shape
            Tensor values = net.critic.Forward(states);

            Tensor returns = values + advantages;
            Tensor critic_loss = Tensor.Pow(returns - values, 2f);

            net.criticOptimizer.ZeroGrad();
            net.critic.Backward(critic_loss);
            net.criticOptimizer.Step();
        }
        private void UpdateContinuousNetwork(Tensor states, Tensor[] actions, Tensor[] logProbs, Tensor advantages)
        {
            // Check if cont_actions are not NaN
            if (actions[0] == null)
                return;
        }
        private void UpdateDiscreteNetwork(Tensor states, Tensor[] actions, Tensor[] logProbs, Tensor advantages)
        {
            if (actions[0] == null)
                return;
        }



        private Tensor[] GeneralizedAdvantageEstimation(Tensor[] rewards, Tensor[] values, Tensor[] dones)
        {
            int count = rewards.Length;

            Tensor[] advantages = new Tensor[count];

            for (int t = 0; t < count - 1; t++)
            {
                float discount = 1f;
                float a_T = 0;

                for (int k = t; k < count - 1; k++)
                {
                    a_T += discount *
                           (rewards[k][0] + hp.gamma * values[k + 1][0] * (1f - dones[k][0]) -
                           values[k][0]);

                    discount *= hp.gamma * hp.lambda;

                    if (dones[k][0] == 1)
                        break;

                    if (k - t >= hp.timeHorizon)
                        break;
                }

                advantages[t] = Tensor.Constant(a_T);
            }

            advantages[advantages.Length - 1] = Tensor.Constant(0);

            return advantages;
        }
    }
}

