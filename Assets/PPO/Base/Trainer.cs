using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEngine;


namespace DeepUnity
{
    public class Trainer : MonoBehaviour
    {
        private static Trainer Instance { get; set; }

        private Dictionary<Agent, bool> agents;
        private HyperParameters hp;
        private AgentBehaviour ac;

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
            // check if there are any ready agents
            if(agents.Values.Contains(true))
                Train();

        }
        public static void Subscribe(Agent agent)
        {
            if(Instance == null)
            {
                GameObject go = new GameObject("Trainer");
                go.AddComponent<Trainer>();
                Instance.agents = new();
                Instance.ac = agent.model;
                Instance.hp = agent.Hp;

                Instance.ac.InitOptimisers(Instance.hp);
                Instance.ac.InitSchedulers(Instance.hp);
            }

            Instance.agents.Add(agent, false);
        }
        public static void Ready(Agent agent)
        {
            Instance.agents[agent] = true;
        }





        private void Train()
        {

            foreach (var kv in agents)
            {
                if (kv.Value == false)
                    return;

                Agent agent = kv.Key;
                GeneralizedAdvantageEstimate(agent.Trajectory);


                // Unzip the memory
                List<Tensor> states = agent.Trajectory.states;
                List<Tensor> continuousActions = agent.Trajectory.continuous_actions;
                List<Tensor> discreteActions = agent.Trajectory.discrete_actions;
                List<Tensor> continuousLogProbs = agent.Trajectory.continuous_log_probs;
                List<Tensor> discreteLogProbs = agent.Trajectory.discrete_log_probs;
                List<Tensor> values = agent.Trajectory.values;
                List<Tensor> rewards = agent.Trajectory.rewards;
                List<Tensor> advantages = agent.Trajectory.advantages;
                List<Tensor> returns = agent.Trajectory.returns;


                Utils.DebugInFile(agent.Trajectory.ToString());

                // STEP Norm advantages
                int n = advantages.Count;
                float mean = advantages.Average(x => x[0]);
                float var = advantages.Sum(x => (x[0] - mean) * (x[0] - mean) / (n - 1));
                float std = MathF.Sqrt(var);
                advantages = advantages.Select(x => (x[0] - mean) / (std + Utils.EPSILON)).Select(x => Tensor.Constant(x)).ToList();
                
                

                int noBatches = (int)(hp.bufferSize / (float)hp.batchSize);
                for (int e = 0; e < hp.numEpoch; e++)
                {
                    // shuffle the training data lists together

                    // split traindata to minibatches
                    List<Tensor[]> states_batches = Utils.Split(states, hp.batchSize);
                    List<Tensor[]> cont_act_batches = Utils.Split(continuousActions, hp.batchSize);
                    List<Tensor[]> cont_log_probs_batches = Utils.Split(continuousLogProbs, hp.batchSize);
                    List<Tensor[]> advantages_batches = Utils.Split(advantages, hp.batchSize);
                    List<Tensor[]> returns_batches = Utils.Split(returns, hp.batchSize);

                    int M = states_batches.Count;

                    for (int b = 0; b < M; b++)
                    { 
                        Tensor states_batch = Tensor.Join(null, states_batches[b]);
                        Tensor advantages_batch = Tensor.Join(null, advantages_batches[b]);
                        Tensor returns_batch = Tensor.Join(null, returns_batches[b]);

                        UpdateCritic(states_batch, returns_batch);
                        UpdateContinuousNetwork(
                                states_batch,
                                advantages_batch,
                                Tensor.Join(null, cont_act_batches[b]),
                                Tensor.Join(null, cont_log_probs_batches[b]));

                    }
                   

                    // Step schedulers after each epoch
                    ac.criticScheduler.Step();
                    ac.muHeadScheduler.Step();
                    ac.sigmaHeadScheduler.Step();
                    for (int i = 0; i < ac.discreteHeadsSchedulers.Length; i++)
                    {
                        ac.discreteHeadsSchedulers[i].Step();
                    }
                }


                agent.Trajectory.Reset();
            }

            var keys = agents.Keys.ToList();
            foreach (var key in keys)
            {
                agents[key] = false;
            }

            // ac.Save();
        }

        private void UpdateCritic(Tensor states_batch, Tensor returns_batch)
        {
            Tensor values = ac.critic.Forward(states_batch);
            Tensor dLdV = 2f * (values - returns_batch); // derivative of MSE

            ac.criticOptimizer.ZeroGrad();
            ac.critic.Backward(dLdV);
            //ac.criticOptimizer.ClipGradNorm(0.5f);
            ac.criticOptimizer.Step();


            float error = Metrics.Accuracy(values, returns_batch);
            print($"Critic Accuracy {error * 100f}%");
        }

        private void UpdateContinuousNetwork(Tensor states, Tensor advantages, Tensor oldActions, Tensor oldLogProbs)
        {
            int batch = states.Height;
            int actions_num = oldActions.Width;
            advantages = Tensor.Expand(advantages, 1, actions_num);


            // Unpack what we need
            Tensor mu;
            Tensor sigma;
            Instance.ac.ContinuousForward(states, out mu, out sigma);

            Tensor newLogProbs = Tensor.LogDensity(oldActions, mu, sigma);

            // Computing d Loss
            Tensor ratio = Tensor.Exp(newLogProbs - oldLogProbs);
            Tensor clipped_ratio = Tensor.Clip(ratio, 1 - hp.epsilon, 1 + hp.epsilon);
            Tensor PIold = Tensor.Exp(oldLogProbs);

            float[,] dmindx = new float[batch, actions_num];
            float[,] dmindy = new float[batch, actions_num];
            float[,] dclipdx = new float[batch, actions_num];

            for (int b = 0; b < batch; b++)
            {
                for (int a = 0; a < actions_num; a++)
                {
                    float pt = ratio[b, a];
                    float eps = hp.epsilon;

                    float clip_p = clipped_ratio[b, a];
                    float At = advantages[b, a];

                    // dMin(x,y)/dx
                    dmindx[b,a] = (pt * At <= clip_p * At) ? 1 : 0;

                    // dMin(x,y)/dy
                    dmindy[b,a] = (clip_p * At < pt * At) ? 1 : 0;

                    // dClip(x,a,b)/dx
                    dclipdx[b,a] = (1.0 - eps <= pt && pt <= 1.0 + eps) ? 1 : 0;
                }
            }


            Tensor dMindX = Tensor.Constant(dmindx);
            Tensor dMindY = Tensor.Constant(dmindy);
            Tensor dClipdX = Tensor.Constant(dclipdx);


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
            //ac.muHeadOptimizer.ClipGradNorm(0.5f);
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

        public void GeneralizedAdvantageEstimate(TrajectoryBuffer trajectory)
        {
            // as described in the paper
            int T = trajectory.Count;
            for (int timestep = 0; timestep < T; timestep++)
            {
                float discount = 1f;
                Tensor v_t = Tensor.Constant(0);

                for(int t = timestep; t < T; t++)
                {       
                    v_t += discount * trajectory.rewards[t];
                    discount *= hp.gamma;

                }
                // horizon ommited for now

                if(trajectory.reachedTerminalState == false)
                {
                    v_t += discount * trajectory.values[T - 1];

                    // else Vwold(state[t + n]) = 0
                }

                Tensor a_t = v_t - trajectory.values[timestep];

                trajectory.returns.Add(v_t);
                trajectory.advantages.Add(a_t);

            }
        }
        

    }
}

