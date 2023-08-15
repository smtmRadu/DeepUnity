using System;
using System.Collections.Generic;
using System.Drawing;
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
        private Hyperparameters hp;
        private TrainingStatistics performanceTracker;
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
                EditorApplication.playModeStateChanged += Autosave1;
                EditorApplication.pauseStateChanged += Autosave2;
                GameObject go = new GameObject("Trainer");
                go.AddComponent<Trainer>();
                Instance.agents = new();
                Instance.ac = agent.model;
                Instance.hp = agent.hp;
                Instance.performanceTracker = agent.PerformanceTrack;
                Instance.ac.InitOptimisers(Instance.hp);
                Instance.ac.InitSchedulers(Instance.hp);
            }

            Instance.agents.Add(agent, false);
        }
        // Methods used to save the Actor Critic network when editor state changes.
        private static void Autosave1(PlayModeStateChange state) => Instance.ac.Save();
        private static void Autosave2(PauseState state) => Instance.ac.Save();



        // PPO algorithm here
        private void Train()
        {
            foreach (var kv in agents)
            {
                if (!kv.Value)
                    continue;

                Agent agent = kv.Key;
                Trajectory trajectory = agent.Trajectory;

                if (hp.debug) Utils.DebugInFile(trajectory.ToString());

                performanceTracker?.cumulativeReward.Append(trajectory.CumulativeReward);

                for (int epoch = 0; epoch < hp.numEpoch; epoch++)
                {
                    // randomizeOrder(train_data)
                    trajectory.Shuffle();

                    // unpack
                    Tensor[] states = trajectory.States;
                    Tensor[] actions_continuous = trajectory.ActionsContinuous;
                    Tensor[] actions_discrete = trajectory.ActionsDiscrete;
                    Tensor[] probs_continous = trajectory.PIoldContinuous;
                    Tensor[] probs_discrete = trajectory.PIoldDiscrete;
                    Tensor[] value_targets = trajectory.ValueTargets;
                    Tensor[] advantages = trajectory.Advantages;

                    // split traindata into minibatches
                    List<Tensor[]> states_batches = Utils.Split(states, hp.batchSize);
                    List<Tensor[]> cont_act_batches = Utils.Split(actions_continuous, hp.batchSize);
                    List<Tensor[]> cont_probs_batches = Utils.Split(probs_continous, hp.batchSize);
                    List<Tensor[]> advantages_batches = Utils.Split(advantages, hp.batchSize);
                    List<Tensor[]> value_targets_batches = Utils.Split(value_targets, hp.batchSize);

                    int M = states_batches.Count;

                    for (int b = 0; b < M; b++)
                    { 
                        Tensor states_batch = Tensor.Cat(null, states_batches[b]);
                        Tensor advantages_batch = Tensor.Cat(null, advantages_batches[b]);
                        Tensor value_targets_batch = Tensor.Cat(null, value_targets_batches[b]);
                        Tensor cont_act_batch = Tensor.Cat(null, cont_act_batches[b]);
                        Tensor cont_probs_batch = Tensor.Cat(null, cont_probs_batches[b]);

                        UpdateCritic(states_batch, value_targets_batch);
                        UpdateContinuousNetwork(
                                states_batch,
                                advantages_batch,
                                cont_act_batch,
                                cont_probs_batch);
                        // UpdateDiscreteNetworks(
                        //     states_batch,
                        //     advantages_batch,
                        //     discreteActions,
                        //     discreteLogProbs);
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

            performanceTracker?.learningRate.Append(ac.criticScheduler.CurrentLR);
        }
        /// <summary>
        /// <paramref name="targets_batch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// <paramref name="targets_batch"/> - <em>V target</em> | Tensor(<em>Batch Size, 1</em>)
        /// </summary>
        private void UpdateCritic(Tensor states_batch, Tensor targets_batch)
        {
            Tensor values_batch = ac.critic.Forward(states_batch);
            Loss criticLoss = Loss.MSE(values_batch, targets_batch);

            ac.criticOptimizer.ZeroGrad();
            ac.critic.Backward(criticLoss.Derivative);
            ac.criticOptimizer.ClipGradNorm(0.5f);
            ac.criticOptimizer.Step();

            performanceTracker?.valueLoss.Append(criticLoss.Item);
        }
        /// <summary>
        /// <paramref name="states_batch"/> - <em>s</em> | Tensor (<em>Batch Size, Observations</em>) <br></br>
        /// <paramref name="advantages_batch"/> - <em>A</em> | Tensor(<em>Batch Size, 1</em>) <br></br>
        /// <paramref name="actions_batch"/> - <em>a</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// <paramref name="PIold_batch"/> - <em>log πθₒₗ(a|s) </em>| Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        private void UpdateContinuousNetwork(Tensor states_batch, Tensor advantages_batch, Tensor actions_batch, Tensor PIold_batch)
        {
            int batch_size = states_batch.Rank == 2 ? states_batch.Size(0) : 1;
            int continuous_actions_num = actions_batch.Size(-1);

            // Forwards pass
            Tensor mu_batch;
            Tensor sigma_batch;
            Instance.ac.ContinuousForward(states_batch, out mu_batch, out sigma_batch);
            Tensor PI_batch = Tensor.PDF(actions_batch, mu_batch, sigma_batch);

            Tensor ratio = PI_batch / PIold_batch; // a.k.a pₜ(θ)

            // Compute Lᶜˡⁱᵖ
            Tensor A = advantages_batch.Expand(1, continuous_actions_num);
            Tensor LClip = Tensor.Minimum(
                                ratio * A, 
                                Tensor.Clip(ratio, 1 - hp.epsilon, 1 + hp.epsilon) * A);
            performanceTracker?.objectiveFunction.Append(LClip.Mean(0).Mean(0)[0]);

            if(LClip.Contains(float.NaN))
            {
                Debug.Log($"------NaN LCLIP removed [{LClip.ToArray().ToCommaSeparatedString()}]-------\n");
                return;   
            }

            // Computing δLClip
            Tensor dmindx_At = Tensor.Zeros(batch_size, continuous_actions_num);
            Tensor dmindy_At = Tensor.Zeros(batch_size, continuous_actions_num);
            Tensor dclipdx = Tensor.Zeros(batch_size, continuous_actions_num);

            for (int b = 0; b < batch_size; b++)
            {
                for (int a = 0; a < continuous_actions_num; a++)
                {
                    float pt = ratio[b, a];
                    float e = hp.epsilon;
                    float At = advantages_batch[b, 0];

                    // δMin(x,y)/δx
                    dmindx_At[b,a] = (pt * At <= Utils.Clip(pt, 1f - e, 1f + e) * At) ? 1f : 0f * At;

                    // δMin(x,y)/δy
                    dmindy_At[b,a] = (Utils.Clip(pt, 1f - e, 1f + e) * At < pt * At) ? 1f : 0f * At;

                    // δClip(x,a,b)/δx
                    dclipdx[b,a] = (1f - e <= pt && pt <= 1f + e) ? 1f : 0f;
                }
            }

            // δ-LClip / δpi(a|s)  (20) --------------------------------------------------------------------
            Tensor dmLClip_dPi = -1f * (dmindx_At + dmindy_At * dclipdx) / PIold_batch;
            // -----------------------------------------------------------------------------------------------



            // Entropy --------------------------------------------------------------------------------------- 
            // Tensor entropy = Tensor.Log(MathF.Sqrt(2f * MathF.PI * MathF.E) * sigma);
            // dmLdPi -= entropy * hp.beta;
            // -----------------------------------------------------------------------------------------------



            // δpi(a|s) / δmu  (26) ------------------------------------------------------------------------
            Tensor dPi_dMu = PI_batch * (actions_batch - mu_batch) / (sigma_batch.Pow(2));
            // -----------------------------------------------------------------------------------------------

            // δ-LClip / δmu = (δ-LClip / δpi(a|t)) * (δpi(a|s) / δmu);
            Tensor dmLClip_dMu = dmLClip_dPi * dPi_dMu;
            ac.muHeadOptimizer.ZeroGrad();
            ac.muHead.Backward(dmLClip_dMu);
            ac.muHeadOptimizer.ClipGradNorm(0.5f);
            ac.muHeadOptimizer.Step();


            // // δpi(a|s) / δsigma (XX) -----------------------------------------------------------------------
            // Tensor dPi_dSigma = ((actions_batch - mu_batch).Pow(2) - sigma_batch.Pow(2)) / sigma_batch.Pow(3);
            // // ------------------------------------------------------------------------------------------------
            // 
            // // δ-LClip / δmu = (δ-LClip / δpi(a|t)) * (δpi(a|s) / δsigma)
            // Tensor dmLClip_dSigma = dmLClip_dPi * dPi_dSigma;
            // ac.sigmaHeadOptimizer.ZeroGrad();
            // ac.sigmaHead.Backward(dmLClip_dSigma);
            // ac.sigmaHeadOptimizer.ClipGradNorm(0.5f);
            // ac.sigmaHeadOptimizer.Step();

            // Test KL
        }
        private void UpdateDiscreteNetworks(Tensor states, Tensor advantages, Tensor oldActions, Tensor oldLogProbs)
        {

        }

    }
}

