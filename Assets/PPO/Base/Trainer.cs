using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity
{
    public class Trainer : MonoBehaviour
    {
        private static Trainer Instance { get; set; }

        private List<Agent> agents;
        private int readyAgents;
        private HyperParameters hp;
        private AgentBehaviour ac;
        private int updatesCount;

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
                Instance.updatesCount = 0;
                Instance.ac = agent.model;
                Instance.hp = agent.Hp;

                Instance.ac.InitOptimisers(Instance.hp);
                Instance.ac.InitSchedulers(Instance.hp);
            }

            Instance.agents.Add(agent);
        }
        public static void Ready()
        {
            Instance.readyAgents++;
        }






        private void Train()
        {
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
                if(hp.normalize)
                {
                    int n = advantages.Length;
                    float mean = advantages.Average(x => x[0]);
                    float var = advantages.Sum(x => (x[0] - mean) * (x[0] - mean) / (n - 1));
                    float std = MathF.Sqrt(var);
                    advantages = advantages.Select(x => (x[0] - mean) / (std + Utils.EPSILON)).Select(x => Tensor.Constant(x)).ToArray();
                }

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
                    
                    List<float> criticAccs = new List<float>();

                    for (int n = 0; n < noBatches; n++)
                    {
                        int start = n * hp.batchSize;
                        
                        // Generate a batch
                        Tensor[] states_batch_arr = (Tensor[]) Utils.GetRange(states, start, hp.batchSize);
                        Tensor[] cont_action_batch_arr = (Tensor[])Utils.GetRange(continuousActions, start, hp.batchSize);
                        Tensor[] disc_action_batch_arr = (Tensor[])Utils.GetRange(discreteActions, start, hp.batchSize);
                        Tensor[] cont_logProbs_batch_arr = (Tensor[]) Utils.GetRange(continuousLogProbs, start, hp.batchSize);
                        Tensor[] disc_logProbs_batch_arr = (Tensor[])Utils.GetRange(discreteLogProbs, start, hp.batchSize);
                        Tensor[] advantages_batch_arr = (Tensor[])Utils.GetRange(advantages, start, hp.batchSize);


                        Tensor states_batch = Tensor.Join(TDim.height, states_batch_arr);
                        Tensor advantages_batch = Tensor.Join(TDim.height, advantages_batch_arr);
 
                        // Update Critic and Heads
                        float criticAcc = UpdateCriticNetwork(states_batch, advantages_batch);
                        try
                        {
                            // Because cont_XXX can have null elements, it may fail on Joining. This happen due to missing continuous actions.
                            UpdateContinuousNetwork(
                                states_batch, 
                                advantages_batch, 
                                Tensor.Join(TDim.height, cont_action_batch_arr), 
                                Tensor.Join(TDim.height, cont_logProbs_batch_arr));

                        }
                        catch { }
                        try
                        {
                            // Because cont_XXX can have null elements, it may fail on Joining. This happen due to missing discrete actions.
                            UpdateDiscreteNetwork(
                                states_batch,
                                advantages_batch, 
                                null, 
                                null);
                        }
                        catch { }

                        criticAccs.Add(criticAcc);
                    }

                    // Step schedulers after each epoch
                    ac.criticScheduler.Step();
                    ac.muHeadScheduler.Step();
                    ac.sigmaHeadScheduler.Step();
                    for (int i = 0; i < ac.discreteHeadsSchedulers.Length; i++)
                    {
                        ac.discreteHeadsSchedulers[i].Step();
                    }


                    print($"Epoch: {updatesCount++} | Critic Accuracy: {criticAccs.Average()}");
                }

              
                agent.Memory.Clear();
            }

            // ac.Save();
            readyAgents = 0;
        }      


        private float UpdateCriticNetwork(Tensor states, Tensor advantages)
        {
            // [batch, values] shape
            Tensor values = ac.critic.Forward(states);

            Tensor returns = values + advantages; //targets

            Tensor errors = Tensor.Abs(values - returns);
            Tensor dLdV = 2f * (values - returns); // dMSE

            ac.criticOptimizer.ZeroGrad();
            ac.critic.Backward(dLdV);
            ac.criticOptimizer.ClipGradNorm(0.5f);
            ac.criticOptimizer.Step();

            return 1f - Tensor.Mean(errors, TDim.height)[0];
        }
        private void UpdateContinuousNetwork(Tensor states, Tensor advantages, Tensor oldActions, Tensor oldLogProbs)
        {
            int batch = states.Shape.Height;
            int actions_num = oldActions.Shape.Width;
            advantages = Tensor.Expand(advantages, TDim.width, actions_num);


            // Unpack what we need
            Tensor mu;
            Tensor sigma;
            Instance.ac.ContinuousForward(states, out mu, out sigma);

            Tensor newLogProbs = Tensor.LogDensity(oldActions, mu, sigma);

            // Computing d Loss
            Tensor ratio = Tensor.Exp(newLogProbs - oldLogProbs);
            Tensor clipped_ratio = Tensor.Clip(ratio, 1 - hp.epsilon, 1 + hp.epsilon);
            Tensor PIold = Tensor.Exp(oldLogProbs);

            Tensor dMindX = Tensor.Zeros(batch, actions_num);
            Tensor dMindY = Tensor.Zeros(batch, actions_num);
            Tensor dClipdX = Tensor.Zeros(batch, actions_num);
            
            for (int b = 0; b < batch; b++)
            {
                for (int a = 0; a < actions_num; a++)
                {
                    float pt = ratio[b, a];
                    float eps = hp.epsilon;

                    float clip_p = clipped_ratio[b, a];
                    float At = advantages[b, a];

                    // dMin(x,y)/dx
                    dMindX[b,a] = (pt * At <= clip_p * At ? 1 : 0);

                    // dMin(x,y)/dy
                    dMindY[b,a] = (clip_p * At < pt * At ? 1 : 0);

                    // dClip(x,a,b)/dx
                    dClipdX[b,a] = (1.0 - eps <= pt && pt <= 1.0 + eps ? 1 : 0);
                }
            }

            // d-LClip / dPi[a,s]
            Tensor dmLdPI = -1f * (dMindX * advantages + dMindY * advantages * dClipdX) * 1f / PIold;

            // Entropy (no need for continuous actions space)
            // Tensor entropy = Tensor.Log(MathF.Sqrt(2f * MathF.PI * MathF.E) * sigma);
            // lClipLoss -= entropy * beta;

            // d PI[a,t] / d Mu
            Tensor dPidMu = Tensor.Exp(newLogProbs) * (oldActions - mu) / (sigma * sigma);
            Tensor dLdMu = dmLdPI * dPidMu;

            ac.muHeadOptimizer.ZeroGrad();
            ac.muHead.Backward(dLdMu);
            ac.muHeadOptimizer.ClipGradNorm(0.5f);
            ac.muHeadOptimizer.Step();


            // d Pi[a,t] / d Sigma
            // Tensor dPidSigma = ((actions - mu) * (actions - mu) - sigma * sigma) / Tensor.Pow(sigma, 3f);
            // Tensor sigmaLoss = dmLdPi * dPidSigma;
            // 
            // net.sigmaHeadOptimizer.ZeroGrad();
            // net.sigmaHead.Backward(sigmaLoss);
            // net.sigmaHeadOptimizer.Step();

            // Test KL
        }
        private void UpdateDiscreteNetwork(Tensor states, Tensor advantages, Tensor actions, Tensor oldLogProbs)
        {
            
        }



        private Tensor[] GeneralizedAdvantageEstimation(Tensor[] rewards, Tensor[] values, Tensor[] dones)
        {
            int count = rewards.Length;

            Tensor[] advantages = new Tensor[count];

            for (int t = 0; t < count; t++)
            {
                float discount = 1f;
                float a_T = 0;

                for (int k = t; k < Math.Min(count - 1, hp.timeHorizon); k++)
                {
                    a_T += discount *
                           (rewards[k][0] + hp.gamma * values[k + 1][0] * (1f - dones[k][0]) -
                           values[k][0]);

                    discount *= hp.gamma * hp.lambda;
                }

                advantages[t] = Tensor.Constant(a_T);
            }

            return advantages;
        }
    }
}

