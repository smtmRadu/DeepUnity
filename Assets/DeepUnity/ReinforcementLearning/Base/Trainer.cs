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

        private List<Agent> agents;
        private Hyperparameters hp;
        private TrainingStatistics trainingStatisticsTrack;
        private AgentBehaviour ac;
        private bool train = false;

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
            if (Instance.train)
                Train();
        }

        // Methods use to interact with the agents
        public static void ReadyToTrain() => Instance.train = true;
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
                Instance.trainingStatisticsTrack = agent.PerformanceTrack;
                Instance.ac.InitOptimisers(Instance.hp);
                Instance.ac.InitSchedulers(Instance.hp);
            }
            agent.PerformanceTrack = Instance.trainingStatisticsTrack;
            Instance.agents.Add(agent);
        }

        // Methods used to save the Actor Critic network when editor state changes.
        private static void Autosave1(PlayModeStateChange state) => Instance.ac.Save();
        private static void Autosave2(PauseState state) => Instance.ac.Save();



        // PPO algorithm
        private void Train()
        {
            if(Instance.trainingStatisticsTrack != null)
                Instance.trainingStatisticsTrack.iterations++;

            foreach (var train_data in Instance.agents.Select(x => x.Memory))
            {
                train_data.GAE(hp.gamma, hp.lambda, hp.horizon, ac.critic);

                if(hp.normalizeAdvantages)
                    train_data.NormAdvantages();

                if (hp.debug) 
                    Utils.DebugInFile(train_data.ToString());

                for (int epoch = 0; epoch < hp.numEpoch; epoch++)
                {
                    // randomizeOrder(train_data)
                    train_data.Shuffle();

                    // unpack & split traindata into minibatches
                    List<Tensor[]> states_batches = Utils.Split(train_data.States, hp.batchSize);
                    List<Tensor[]> cont_act_batches = Utils.Split(train_data.ContinuousActions, hp.batchSize);
                    List<Tensor[]> cont_log_probs_batches = Utils.Split(train_data.ContinuousLogProbs, hp.batchSize);
                    List<Tensor[]> advantages_batches = Utils.Split(train_data.Advantages, hp.batchSize);
                    List<Tensor[]> value_targets_batches = Utils.Split(train_data.ValueTargets, hp.batchSize);

                    for (int b = 0; b < states_batches.Count; b++)
                    { 
                        Tensor states_batch = Tensor.Cat(null, states_batches[b]);
                        Tensor advantages_batch = Tensor.Cat(null, advantages_batches[b]);
                        Tensor value_targets_batch = Tensor.Cat(null, value_targets_batches[b]);
                        Tensor cont_act_batch = Tensor.Cat(null, cont_act_batches[b]);
                        Tensor cont_log_probs_batch = Tensor.Cat(null, cont_log_probs_batches[b]);

                        UpdateCritic(states_batch, value_targets_batch);
                        UpdateContinuousNetwork(
                                states_batch,
                                advantages_batch,
                                cont_act_batch,
                                cont_log_probs_batch);
                        // UpdateDiscreteNetworks(
                        //     states_batch,
                        //     advantages_batch,
                        //     discreteActions,
                        //     discreteProbs);
                    }
                   
                    // Step schedulers after each epoch
                    ac.criticScheduler.Step();
                    ac.actorMuScheduler.Step();
                    ac.actorSigmaScheduler.Step();
                    for (int i = 0; i < ac.actorDiscretesScheduler.Length; i++)
                    {
                        ac.actorDiscretesScheduler[i].Step();
                    }
                }

                train_data.Reset();
            }

            Instance.train = false;
            trainingStatisticsTrack?.learningRate.Append(ac.criticScheduler.CurrentLR);
            trainingStatisticsTrack?.epsilon.Append(hp.epsilon);
        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>) where * = <em>Observations Shape</em><br></br>
        /// <paramref name="targets"/> - <em>Vtarget</em> | Tensor(<em>Batch Size, 1</em>)
        /// </summary>
        private void UpdateCritic(Tensor states, Tensor targets)
        {
            Tensor values = ac.critic.Forward(states);
            Loss criticLoss = Loss.MSE(values, targets);

            ac.criticOptimizer.ZeroGrad();
            ac.critic.Backward(criticLoss.Derivative);
            ac.criticOptimizer.ClipGradNorm(0.5f);
            ac.criticOptimizer.Step();

            trainingStatisticsTrack?.valueLoss.Append(criticLoss.Item);
        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>)  where * = <em>Observations Shape</em><br></br>
        /// <paramref name="advantages"/> - <em>A</em> | Tensor(<em>Batch Size, 1</em>) <br></br>
        /// <paramref name="actions"/> - <em>a</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// <paramref name="logPiOld"/> - <em>log πθold(a|s) </em>| Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        private void UpdateContinuousNetwork(Tensor states, Tensor advantages, Tensor actions, Tensor logPiOld)
        {
            int batch_size = states.Rank == 2 ? states.Size(0) : 1;
            int continuous_actions_num = actions.Size(-1);

            // Forwards pass
            Tensor mu;
            Tensor sigma;
            Instance.ac.ContinuousForward(states, out mu, out sigma);
            Tensor logPi = Tensor.LogProbability(actions, mu, sigma);

            Tensor ratio = (logPi - logPiOld).Exp(); // a.k.a pₜ(θ) = πθ(a|s) / πθold(a|s)

            // Compute L_CLIP
            advantages = advantages.Expand(1, continuous_actions_num);
            Tensor LClip = Tensor.Minimum(
                                ratio * advantages, 
                                Tensor.Clip(ratio, 1 - hp.epsilon, 1 + hp.epsilon) * advantages);

            if(LClip.Contains(float.NaN))
            {
                Debug.Log($"<color=#fcba03>Warning! L_CLIP containing NaN values removed [{LClip.ToArray().ToCommaSeparatedString()}]! </color>");
                return;   
            }
            trainingStatisticsTrack?.policyLoss.Append(-LClip.Mean(0).Mean(0)[0]);

            // Computing δLClip
            Tensor dmindx = Tensor.Zeros(batch_size, continuous_actions_num);
            Tensor dmindy = Tensor.Zeros(batch_size, continuous_actions_num);
            Tensor dclipdx = Tensor.Zeros(batch_size, continuous_actions_num);

            for (int b = 0; b < batch_size; b++)
            {
                for (int a = 0; a < continuous_actions_num; a++)
                {
                    float pt = ratio[b, a];
                    float e = hp.epsilon;
                    float At = advantages[b, a];
                    float clip_pt = Math.Clamp(pt, 1f - e, 1f + e);

                    // δMin(x,y)/δx
                    dmindx[b, a] = (pt * At <= clip_pt * At) ? 1f : 0f;

                    // δMin(x,y)/δy
                    dmindy[b, a] = (clip_pt * At < pt * At) ? 1f : 0f;

                    // δClip(x,a,b)/δx
                    dclipdx[b, a] = (1f - e <= pt && pt <= 1f + e) ? 1f : 0f;
                }
            }

            // δ-LClip / δpi(a|s)  (20)
            Tensor dmLClip_dPi = -1f * (dmindx * advantages + dmindy * advantages * dclipdx) / logPiOld.Exp();
           
            if(ac.standardDeviation == StandardDeviationType.Trainable)
            {
                // Entropy
                Tensor entropy = Tensor.Log(MathF.Sqrt(2f * MathF.PI * MathF.E) * sigma);
                dmLClip_dPi -= entropy * hp.beta;
            }

            // δpi(a|s) / δmu  (26) 
            Tensor dPi_dMu = logPi.Exp() * (actions - mu) / sigma.Pow(2);


            // δ-LClip / δmu = (δ-LClip / δpi(a|t)) * (δpi(a|s) / δmu)
            Tensor dmLClip_dMu = dmLClip_dPi * dPi_dMu;
            ac.actorMuOptimizer.ZeroGrad();
            ac.actorMu.Backward(dmLClip_dMu);
            ac.actorMuOptimizer.ClipGradNorm(0.5f);
            ac.actorMuOptimizer.Step();

            if(ac.standardDeviation == StandardDeviationType.Trainable)
            {
                // δpi(a|s) / δsigma
                Tensor dPi_dSigma = ((actions - mu).Pow(2) - sigma.Pow(2)) / sigma.Pow(3);

                // δ-LClip / δmu = (δ-LClip / δpi(a|t)) * (δpi(a|s) / δsigma)
                Tensor dmLClip_dSigma = dmLClip_dPi * dPi_dSigma;
                ac.actorSigmaOptimizer.ZeroGrad();
                ac.actorSigma.Backward(dmLClip_dSigma);
                ac.actorSigmaOptimizer.ClipGradNorm(0.5f);
                ac.actorSigmaOptimizer.Step();
            }
           
        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>)  where * = <em>Observations Shape</em><br></br>
        /// <paramref name="advantages"/> - <em>A</em> | Tensor(<em>Batch Size, 1</em>) <br></br>
        /// <paramref name="actions"/> - <em>a</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// <paramref name="logPiOld"/> - <em>log πθold(a|s) </em>| Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        private void UpdateDiscreteNetworks(Tensor states, Tensor advantages, Tensor actions, Tensor logPiOld)
        {

        }

    }
}

