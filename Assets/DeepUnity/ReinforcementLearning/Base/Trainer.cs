using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;


namespace DeepUnity
{
    public class Trainer : MonoBehaviour
    {
        private static Trainer Instance { get; set; }

        private Dictionary<Agent, bool> agents;
        private HyperParameters hp;
        private AgentPerformanceTracker performanceTracker;
        private AgentBehaviour ac;
        [SerializeField] private int StepCount;

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
            if (Instance.agents.Values.Contains(true))
                Train();

        }

        // Methods use to interact with agents
        public static void Ready(Agent agent)
        {
            Instance.agents[agent] = true;
        }
        public static void Subscribe(Agent agent)
        {
            if(Instance == null)
            {
                EditorApplication.playModeStateChanged += SaveAC;
                EditorApplication.pauseStateChanged += SaveAC2;
                GameObject go = new GameObject("Trainer");
                go.AddComponent<Trainer>();
                Instance.agents = new();
                Instance.ac = agent.model;
                Instance.hp = agent.Hp;
                Instance.performanceTracker = agent.PerformanceTracker;

                Instance.ac.InitOptimisers(Instance.hp);
                Instance.ac.InitSchedulers(Instance.hp);

                Instance.StepCount = 0;
            }

            Instance.agents.Add(agent, false);
        }      



        // Methods used to save the Actor Critic network when editor state changes.
        private static void SaveAC(PlayModeStateChange state) => Instance.ac.Save();
        private static void SaveAC2(PauseState state) => Instance.ac.Save();



        // PPO algorithm here
        private void Train()
        {

            foreach (var kv in agents)
            {
                if (kv.Value == false)
                    continue;

                Agent agent = kv.Key;

                // if(agent.Trajectory.Count == 1)
                // {
                //     agent.Trajectory.Reset();
                //     continue;
                // }
               
                ComputeAdvantageEstimatesAndQValues(agent.Trajectory);

                if(hp.debug)
                    agent.Trajectory.DebugInFile();

                performanceTracker.cumulativeRewards.Append(agent.Trajectory.rewards.Sum(x => x[0]));


                // Unzip the trajectory
                List<Tensor> states = agent.Trajectory.states;
                List<Tensor> continuousActions = agent.Trajectory.continuous_actions;
                List<Tensor> discreteActions = agent.Trajectory.discrete_actions;
                List<Tensor> continuousLogProbs = agent.Trajectory.continuous_log_probs;
                List<Tensor> discreteLogProbs = agent.Trajectory.discrete_log_probs;
                List<Tensor> values = agent.Trajectory.values;
                List<Tensor> rewards = agent.Trajectory.rewards;
                List<Tensor> advantages = agent.Trajectory.advantages;
                List<Tensor> returns = agent.Trajectory.returns;

               
    
                // STEP Norm advantages
                int n = advantages.Count;
                float mean = advantages.Average(x => x[0]);
                float var = advantages.Sum(x => (x[0] - mean) * (x[0] - mean) / (n - 1));
                float std = MathF.Sqrt(var);
                advantages = advantages.Select(x => (x[0] - mean) / (std + Utils.EPSILON)).Select(x => Tensor.Constant(x)).ToList();
                
                for (int e = 0; e < hp.numEpoch; e++)
                {
                    // shuffle the trajectory lists together

                    // split traindata to minibatches
                    List<Tensor[]> states_batches = Utils.Split(states, hp.batchSize);
                    List<Tensor[]> cont_act_batches = Utils.Split(continuousActions, hp.batchSize);
                    List<Tensor[]> cont_log_probs_batches = Utils.Split(continuousLogProbs, hp.batchSize);
                    List<Tensor[]> advantages_batches = Utils.Split(advantages, hp.batchSize);
                    List<Tensor[]> returns_batches = Utils.Split(returns, hp.batchSize);

                    int M = states_batches.Count;

                    for (int b = 0; b < M; b++)
                    { 
                        Tensor states_batch = Tensor.Cat(null, states_batches[b]);
                        Tensor advantages_batch = Tensor.Cat(null, advantages_batches[b]);
                        Tensor returns_batch = Tensor.Cat(null, returns_batches[b]);
                        Tensor cont_act_batch = Tensor.Cat(null, cont_act_batches[b]);
                        Tensor cont_log_probs_batch = Tensor.Cat(null, cont_log_probs_batches[b]);

                        UpdateCritic(states_batch, returns_batch);
                        UpdateContinuousNetwork(
                                states_batch,
                                advantages_batch,
                                cont_act_batch,
                                cont_log_probs_batch);

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


            // Set agents states to unready (this remains like this due to foreach problems when modifing the dict)
            var keys = agents.Keys.ToList();
            foreach (var key in keys)
            {
                agents[key] = false;
            }
        }
        private void UpdateCritic(Tensor states_batch, Tensor returns_batch)
        {
            Tensor values = ac.critic.Forward(states_batch);
            Loss dLdV = Loss.MSE(values, returns_batch);

            ac.criticOptimizer.ZeroGrad();
            ac.critic.Backward(dLdV.Derivative);
            ac.criticOptimizer.Step();

            if(performanceTracker != null)
            {
                float criticLoss = dLdV.Value.Mean(0)[0];
                performanceTracker?.criticLoss.Append(criticLoss);
            }
        }
        private void UpdateContinuousNetwork(Tensor states, Tensor advantages, Tensor oldActions, Tensor oldLogProbs)
        {
            int batch = states.Rank == 2 ? states.Size(-2) : 1;
            int actions_num = oldActions.Size(-1);
            advantages = Tensor.Expand(advantages, 1, actions_num);


            // Unpack what we need
            Tensor mu;
            Tensor sigma;
            Instance.ac.ContinuousForward(states, out mu, out sigma);
            Tensor sigmaSqr = sigma * sigma;

            Tensor newLogProbs = Tensor.LogDensity(oldActions, mu, sigma);

            // Computing d Loss
            Tensor ratio = Tensor.Exp(newLogProbs - oldLogProbs);
            Tensor clipped_ratio = Tensor.Clip(ratio, 1f - hp.epsilon, 1f + hp.epsilon);
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
                    dmindx[b,a] = (pt * At <= clip_p * At) ? 1f : 0f;

                    // dMin(x,y)/dy
                    dmindy[b,a] = (clip_p * At < pt * At) ? 1f : 0f;

                    // dClip(x,a,b)/dx
                    dclipdx[b,a] = (1.0f - eps <= pt && pt <= 1.0f + eps) ? 1f : 0f;
                }
            }


            Tensor dMindX = Tensor.Constant(dmindx);
            Tensor dMindY = Tensor.Constant(dmindy);
            Tensor dClipdX = Tensor.Constant(dclipdx);


            // d-LClip / dPi[a,s]
            Tensor dmLdPi = -1f * (dMindX * advantages + dMindY * advantages * dClipdX) * 1f / PIold;

            // Entropy 
            Tensor entropy = Tensor.Log(MathF.Sqrt(2f * MathF.PI * MathF.E) * sigma);
            dmLdPi -= entropy * hp.beta;

            // d PI[a,t] / d Mu
            Tensor dPidMu = Tensor.Exp(newLogProbs) * (oldActions - mu) / (sigmaSqr);
            Tensor dLdMu = dmLdPi * dPidMu;

            ac.muHeadOptimizer.ZeroGrad();
            ac.muHead.Backward(dLdMu);
            ac.muHeadOptimizer.Step();


            // d Pi[a,t] / d Sigma
            // Tensor dPidSigma = ((oldActions - mu) * (oldActions - mu) - sigmaSqr) / (sigmaSqr * sigma);
            // Tensor dLdSigma = dmLdPi * dPidSigma;
            // 
            // ac.sigmaHeadOptimizer.ZeroGrad();
            // ac.sigmaHead.Backward(dLdSigma);
            // ac.sigmaHeadOptimizer.Step();

            // Test KL
        }
        private void UpdateDiscreteNetworks(Tensor states, Tensor advantages, Tensor oldActions, Tensor oldLogProbs)
        {

        }
        public void ComputeAdvantageEstimatesAndQValues(TrajectoryBuffer trajectory)
        {
            int T = trajectory.Count;
            
            for (int timestep = 0; timestep < T; timestep++)
            {
                float discount = 1f;
                Tensor v_t = Tensor.Constant(0);

                
                for (int t = timestep; t < T; t++)
                {       
                    v_t += discount * trajectory.rewards[t];
                    discount *= hp.gamma;

                    // if(t - timestep == hp.horizon) goto save v_t and a_t
                }

                // If the trajectory terminated due to the maximal trajectory length T being reached,
                // Vwold(st + n) denotes the state value associated with state st+n as predicted by the state value network

                // Otherwise, Vwold(st + n) is set to 0, since this condition indicates that the agent reached a terminal
                // state within its environment from where onward no future rewards could be accumulated any longer.
                if (trajectory.reachedTerminalState == false)
                {
                    v_t += discount * trajectory.values[T - 1];

                    
                }
                // else Vwold(state[t + n]) = 0


                Tensor a_t = v_t - trajectory.values[timestep];

                trajectory.returns.Add(v_t);
                trajectory.advantages.Add(a_t);

            }
        }
        

    }
}

