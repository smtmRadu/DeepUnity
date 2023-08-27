using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;


namespace DeepUnity
{
    /// <summary>
    /// https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
    /// https://link.springer.com/article/10.1007/BF00992696
    /// </summary>
    public class Trainer : MonoBehaviour
    {
        private static Trainer Instance { get; set; }

        [ReadOnly, SerializeField] private List<Agent> parallelAgents;
        [ReadOnly, SerializeField] private Hyperparameters hp;
        [ReadOnly, SerializeField] private TrainingStatistics trainingStatisticsTrack;
        [ReadOnly, SerializeField] private AgentBehaviour ac;
        [SerializeField] private int autosave = 5;

        private bool trainFlag = false;
        private float autosaveSecondsElapsed = 0f;
        private float meanPolicyLoss = 0f;
        private float meanValueLoss = 0f;
        private DateTime timeWhenTheTrainingStarted = DateTime.Now;


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
        private void Update()
        {
            if (Instance.trainingStatisticsTrack != null)
            {
                Instance.trainingStatisticsTrack.trainingTime += Time.deltaTime;
                Instance.trainingStatisticsTrack.realTrainingTime =
                    $"{(int)(DateTime.Now - Instance.timeWhenTheTrainingStarted).TotalHours} hrs : " +
                    $"{(int)(DateTime.Now - Instance.timeWhenTheTrainingStarted).TotalMinutes % 60} min : " +
                    $"{(int)(DateTime.Now - Instance.timeWhenTheTrainingStarted).TotalSeconds % 60} sec";
            }

            // Autosave process 
            Instance.autosaveSecondsElapsed += Time.deltaTime;
        }
        private void LateUpdate()
        {
            if (Instance.trainFlag)
            {
                Train();
                Instance.trainFlag = false;

                if (Instance.trainingStatisticsTrack != null)
                {
                    Instance.trainingStatisticsTrack.parallelAgents = Instance.parallelAgents.Count;
                    Instance.trainingStatisticsTrack.totalSteps += Instance.hp.bufferSize * parallelAgents.Count;
                    Instance.trainingStatisticsTrack.iterations++;
                }
            }

            // Autosaves the ac
            if (Instance.autosaveSecondsElapsed >= Instance.autosave * 60f)
            {
                Instance.autosaveSecondsElapsed = 0f;
                Instance.ac.Save();
            }

        }

        // Methods use to interact with the agents
        public static void ReadyToTrain() => Instance.trainFlag = true;
        public static void Subscribe(Agent agent)
        {
            if(Instance == null)
            {
                EditorApplication.playModeStateChanged += Autosave1;
                EditorApplication.pauseStateChanged += Autosave2;
                GameObject go = new GameObject("[Deep Unity] Trainer");
                go.AddComponent<Trainer>();
                Instance.parallelAgents = new();
                Instance.ac = agent.model;
                Instance.hp = agent.hp;
                Instance.ac.InitOptimisers(Instance.hp);
                Instance.ac.InitSchedulers(Instance.hp);
            }

            // Assign a common TrainingStatistics for all agents (and also the trainer)
            if(agent.PerformanceTrack != null)
            {
                Instance.trainingStatisticsTrack = agent.PerformanceTrack;
                Instance.parallelAgents.ForEach(x => x.PerformanceTrack = agent.PerformanceTrack);
            }

            Instance.parallelAgents.Add(agent);
        }

        // Methods used to save the Actor Critic network when editor state changes.
        private static void Autosave1(PlayModeStateChange state)
        {
            Instance.ac.Save();
            if (state == PlayModeStateChange.ExitingPlayMode && Instance.trainingStatisticsTrack != null)
            {
                Instance.trainingStatisticsTrack.start = Instance.timeWhenTheTrainingStarted.ToLongTimeString() + ", " + Instance.timeWhenTheTrainingStarted.ToLongDateString();
                Instance.trainingStatisticsTrack.end = DateTime.Now.ToLongTimeString() + ", " + DateTime.Now.ToLongDateString();
                string pth = Instance.trainingStatisticsTrack.ExportAsSVG(Instance.ac.behaviourName);
                Debug.Log($"<color=#57f542>Training Session statistics log saved at <b><i>{pth}</i></b>.</color>");
                AssetDatabase.Refresh();
            }
                
        }
        private static void Autosave2(PauseState state) => Instance.ac.Save();
      

        // PPO algorithm
        private void Train()
        {
            ac.SetActorDevice(ac.trainingDevice);
            foreach (var train_data in Instance.parallelAgents.Select(x => x.Memory))
            {
                ac.SetCriticDevice(ac.inferenceDevice); // critic is used now to compute the values, so for time efficiency, inference device is used now
                train_data.GAE(hp.gamma, hp.lambda, hp.horizon, ac.critic);
                ac.SetCriticDevice(ac.trainingDevice); // and then on training, training device is used.

                if(hp.normalizeAdvantages)
                    train_data.NormalizeAdvantages();

                if (hp.debug) 
                    Utils.DebugInFile(train_data.ToString());

                for (int epoch = 0; epoch < hp.numEpoch; epoch++)
                {
                    meanPolicyLoss = 0f;
                    meanValueLoss = 0f;


                    // randomizeOrder(train_data)
                    train_data.Shuffle();

                    // unpack & split traindata into minibatches
                    List<Tensor[]> states_batches = Utils.Split(train_data.States, hp.batchSize);                 
                    List<Tensor[]> advantages_batches = Utils.Split(train_data.Advantages, hp.batchSize);
                    List<Tensor[]> value_targets_batches = Utils.Split(train_data.ValueTargets, hp.batchSize);
                    List<Tensor[]> cont_act_batches = Utils.Split(train_data.ContinuousActions, hp.batchSize);
                    List<Tensor[]> cont_probs_batches = Utils.Split(train_data.ContinuousProbabilities, hp.batchSize);
                    List<Tensor[]> disc_act_batches = Utils.Split(train_data.DiscreteActions, hp.batchSize); // the list contains the batches, the first [] represents the batch, the second[] represents the branch
                    List<Tensor[]> disc_probs_batches = Utils.Split(train_data.DiscreteProbabilities, hp.batchSize);

                    for (int b = 0; b < states_batches.Count; b++)
                    { 
                        Tensor states_batch = Tensor.Cat(null, states_batches[b]);
                        Tensor advantages_batch = Tensor.Cat(null, advantages_batches[b]);
                        Tensor value_targets_batch = Tensor.Cat(null, value_targets_batches[b]);
                        
                        UpdateCritic(states_batch, value_targets_batch);

                        if(Instance.ac.IsUsingContinuousActions)
                        {
                            Tensor cont_act_batch = Tensor.Cat(null, cont_act_batches[b]);
                            Tensor cont_probs_batch = Tensor.Cat(null, cont_probs_batches[b]);
                            UpdateContinuousNetwork(
                                states_batch,
                                advantages_batch,
                                cont_act_batch,
                                cont_probs_batch);
                        }
                        if(Instance.ac.IsUsingDiscreteActions)
                        {
                            int num_branches = Instance.ac.discreteBranches.Length;
                            for (int branch_id = 0; branch_id < num_branches; branch_id++)
                            {
                                Tensor disc_act_batch_branch_i = Tensor.Cat(null, cont_act_batches[b][branch_id]);
                                Tensor disc_prob_batch_branch_i = Tensor.Cat(null, cont_probs_batches[b][branch_id]);
                                UpdateDiscreteBranchNetwork(
                                    states_batch,
                                    advantages_batch,
                                    disc_act_batch_branch_i,
                                    disc_prob_batch_branch_i,
                                    branch_id);

                            }
                          
                        }


                    }
                   
                    // Step schedulers after each epoch
                    ac.criticScheduler.Step();
                    if(ac.IsUsingContinuousActions)
                    {
                        ac.actorMuScheduler.Step();
                        ac.actorSigmaScheduler.Step();
                    }                   
                    if(ac.IsUsingDiscreteActions)                  
                        for (int i = 0; i < ac.actorDiscretesSchedulers.Length; i++)
                        {
                            ac.actorDiscretesSchedulers[i].Step();
                        }

                    // Save statistics info
                    trainingStatisticsTrack?.learningRate.Append(ac.criticScheduler.CurrentLR);
                    trainingStatisticsTrack?.epsilon.Append(hp.epsilon);
                    trainingStatisticsTrack?.policyLoss.Append(meanPolicyLoss / hp.batchSize);
                    trainingStatisticsTrack?.valueLoss.Append(meanValueLoss / hp.batchSize);

                }
                train_data.Reset();
            }
            ac.SetActorDevice(ac.inferenceDevice);
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

            meanValueLoss += criticLoss.Item;
        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>)  where * = <em>Observations Shape</em><br></br>
        /// <paramref name="advantages"/> - <em>A</em> | Tensor(<em>Batch Size, 1</em>) <br></br>
        /// <paramref name="actions"/> - <em>a</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// <paramref name="piOld"/> - <em>πθold(a|s) </em>| Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        private void UpdateContinuousNetwork(Tensor states, Tensor advantages, Tensor actions, Tensor piOld)
        {
            int batch_size = states.Rank == 2 ? states.Size(0) : 1;
            int continuous_actions_num = actions.Size(-1);

            // Forwards pass
            Tensor mu;
            Tensor sigma;
            Instance.ac.ContinuousForward(states, out mu, out sigma);
            Tensor pi = Tensor.Probability(actions, mu, sigma);

            Tensor ratio = pi / piOld; // a.k.a pₜ(θ) = πθ(a|s) / πθold(a|s)

            // Compute L CLIP
            advantages = advantages.Expand(1, continuous_actions_num);
            Tensor LClip = Tensor.Minimum(
                                ratio * advantages, 
                                Tensor.Clip(ratio, 1 - hp.epsilon, 1 + hp.epsilon) * advantages);

            if(LClip.Contains(float.NaN))
            {
                ConsoleMessage.Warning($"L_CLIP containing NaN values removed [{ LClip.ToArray().ToCommaSeparatedString()}]");
                return;   
            }
            meanPolicyLoss += Mathf.Abs(LClip.Mean(0).Mean(0)[0]);

            // Computing δ-LClip / δπθ(a|s)
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

            // δ-LClip / δπθ(a|s)  (20) Bick.D
            Tensor dmLClip_dPi = -1f * (dmindx * advantages + dmindy * advantages * dclipdx) / piOld;


            // Entropy bonus added if σ is trainable
            if (ac.standardDeviation == StandardDeviationType.Trainable)
            {
                Tensor entropy = Tensor.Log(MathF.Sqrt(2f * MathF.PI * MathF.E) * sigma);
                dmLClip_dPi -= entropy * hp.beta;
            }

            // δπθ(a|s) / δμ = πθ(a|s) * (x - μ) / σ^2 /  (26) Bick.D
            Tensor dPi_dMu = pi * (actions - mu) / sigma.Pow(2);


            // δ-LClip / δμ = (δ-LClip / δπθ(a|s)) * (δπθ(a|s) / δμ)
            Tensor dmLClip_dMu = dmLClip_dPi * dPi_dMu;
            ac.actorMuOptimizer.ZeroGrad();
            ac.actorMu.Backward(dmLClip_dMu);
            ac.actorMuOptimizer.ClipGradNorm(0.5f);
            ac.actorMuOptimizer.Step();

            if(ac.standardDeviation == StandardDeviationType.Trainable)
            {
                // δπθ(a|s) / δσ = πθ(a|s) * (x - μ)^2 / σ^3    (Simple statistical gradient-following for connectionst Reinforcement Learning (pag 14)
                Tensor dPi_dSigma = pi * ((actions - mu).Pow(2) - sigma.Pow(2)) / sigma.Pow(3);

                // δ-LClip / δμ = (δ-LClip / δπθ(a|s)) * (δπθ(a|s) / δσ)
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
        /// <paramref name="actions"/> - <em>a</em> | Tensor (<em>Batch Size, Discrete Branchᵢ</em>) <br></br>
        /// <paramref name="piOld"/> - <em>πθold(a|s) </em>| Tensor (<em>Batch Size, Discrete Branchᵢ</em>) <br></br>
        /// </summary>
        private void UpdateDiscreteBranchNetwork(Tensor states, Tensor advantages, Tensor actions, Tensor piOld, int discreteBranchId)
        {
            throw new NotImplementedException();

            int batch_size = states.Rank == 2 ? states.Size(0) : 1;
            int discrete_actions_num = actions.Size(-1); // on this branch


        }

    }

    [CustomEditor(typeof(Trainer), true), CanEditMultipleObjects]
    sealed class CustomTrainerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            List<string> dontDrawMe = new() { "m_Script" };

            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}

