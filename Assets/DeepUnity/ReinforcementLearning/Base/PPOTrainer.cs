using System;
using System.Collections.Generic;
using UnityEditor;
using UnityEngine;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using DeepUnity.Models;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// [1] https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
    /// [2] https://link.springer.com/article/10.1007/BF00992696
    /// [3] https://ieeexplore.ieee.org/document/9520424
    /// </summary>
    /// What may cause NaN values:
    /// 1. Softmax activation on discrete head may "explode"
    internal sealed class PPOTrainer : DeepUnityTrainer
    {
        protected override void Initialize() { }
        protected override void OnBeforeFixedUpdate()
        {
            // If agents cumulativelly collected enough data to fill up the buffer (it can surpass for multiple agents)
            if (MemoriesCount >= hp.bufferSize)
            {
                foreach (var agent_memory in parallelAgents.Select(x => x.Memory))
                {
                    if (agent_memory.Count == 0)
                        continue;

                    ComputeGAE_andVtargets(agent_memory, hp.gamma, hp.lambda, hp.horizon, model.vNetwork);
                    train_data.TryAppend(agent_memory, hp.bufferSize);
                    if (hp.debug) Utils.DebugInFile(agent_memory.ToString());
                    agent_memory.Clear();
                }

                updateClock = Stopwatch.StartNew();
                updateIterations++;
                Train();
                updateClock.Stop();

                actorLoss = actorLoss / (hp.bufferSize / hp.batchSize * hp.numEpoch);
                criticLoss = criticLoss / (hp.bufferSize / hp.batchSize * hp.numEpoch);
                entropy = entropy / (hp.bufferSize / hp.batchSize * hp.numEpoch);
                learningRate = model.vScheduler.CurrentLR;
                currentSteps += hp.bufferSize;
            }
        }



        // PPO Algorithm
        private void Train()
        {
            // https://openreview.net/forum?id=nIAxjsniDzg shows that the advantages must be normalized over the mini-batch

            model.vNetwork.Device = model.trainingDevice; // This is always on training device because it is used to compute the values of the entire train_data of states once..
            
            if(model.muNetwork != null)
                model.muNetwork.Device = model.trainingDevice;

            if (model.sigmaNetwork != null)
                model.sigmaNetwork.Device = model.trainingDevice;

            if (model.discreteNetwork != null)
                model.discreteNetwork.Device = model.trainingDevice;

            for (int epoch_index = 0; epoch_index < hp.numEpoch; epoch_index++)
            {
                // shuffle the dataset
                if (hp.batchSize != hp.bufferSize) // epoch_index > 0 && => because we do advantages normalization
                    train_data.Shuffle();

                // unpack & split train_data into minibatches
                List<Tensor[]> states_batches = Utils.Split(train_data.States, hp.batchSize);
                List<Tensor[]> advantages_batches = Utils.Split(train_data.Advantages, hp.batchSize);
                List<Tensor[]> value_targets_batches = Utils.Split(train_data.ValueTargets, hp.batchSize);
                List<Tensor[]> cont_act_batches = Utils.Split(train_data.ContinuousActions, hp.batchSize);
                List<Tensor[]> cont_probs_batches = Utils.Split(train_data.ContinuousProbabilities, hp.batchSize);
                List<Tensor[]> disc_act_batches = Utils.Split(train_data.DiscreteActions, hp.batchSize);
                List<Tensor[]> disc_probs_batches = Utils.Split(train_data.DiscreteProbabilities, hp.batchSize);


                // θ new. New probabilities of the policy used for early stopping/rollback
                Tensor cont_probs_new = null;
                Tensor cont_probs_old = null;
                Tensor disc_probs_new = null;
                Tensor disc_probs_old = null;

                for (int b = 0; b < states_batches.Count; b++)
                {
                    Tensor states_batch = Tensor.Concat(null, states_batches[b]);
                    Tensor advantages_batch = Tensor.Concat(null, advantages_batches[b]);
                    Tensor value_targets_batch = Tensor.Concat(null, value_targets_batches[b]);

                    if (hp.normalizeAdvantages)
                        advantages_batch = NormalizeAdvantages(advantages_batch); // Advantage normalization is shit when applied on small batches

                    UpdateValueNetwork(states_batch, value_targets_batch);

                    if (model.IsUsingContinuousActions)
                    {
                        Tensor cont_act_batch = Tensor.Concat(null, cont_act_batches[b]);
                        cont_probs_old = Tensor.Concat(null, cont_probs_batches[b]);
                        UpdateContinuousNetwork(
                            states_batch,
                            advantages_batch,
                            cont_act_batch,
                            cont_probs_old,
                            out cont_probs_new);
                    }
                    if (model.IsUsingDiscreteActions)
                    {
                        Tensor disc_act_batch = Tensor.Concat(null, disc_act_batches[b]);
                        disc_probs_old = Tensor.Concat(null, disc_probs_batches[b]);
                        UpdateDiscreteNetwork(
                            states_batch,
                            advantages_batch,
                            disc_act_batch,
                            disc_probs_old,
                            out disc_probs_new);
                    }

                }

                // Check KL Divergence based on the last Minibatch (see [3])
                if (hp.KLDivergence != KLType.Off)
                {
                    // Though even if i should stop for them separatelly (i mean i can let for one to continue the training if kl is small) i will let it simple..
                    float kldiv_cont = model.IsUsingContinuousActions ? ComputeKLDivergence(cont_probs_new, cont_probs_old) : 0;
                    float kldiv_disc = model.IsUsingDiscreteActions ? ComputeKLDivergence(disc_probs_new, disc_probs_old) : 0;

                    if (kldiv_cont > hp.targetKL || kldiv_disc > hp.targetKL)
                    {
                        // ConsoleMessage.Info($"<b>Early Stopping</b> triggered (KL_continuous: {kldiv_cont} | KL_discrete{kldiv_disc}) ({hp.KLDivergence})");
                        break;
                    }
                    // for rollback is the same but we need to cache the old state of the network and it is costly....
                }


                // Step LR schedulers after each epoch (this allows selection at runtime)
                if (hp.LRSchedule)
                {
                    model.vScheduler.Step();
                    model.muScheduler?.Step();
                    model.sigmaScheduler?.Step();
                    model.discreteScheduler?.Step();
                }
            }

            if (model.muNetwork != null)
                model.muNetwork.Device = model.inferenceDevice;
            if(model.sigmaNetwork != null)
                model.sigmaNetwork.Device = model.inferenceDevice; 
            if(model.discreteNetwork != null)
                model.discreteNetwork.Device = model.inferenceDevice;


            train_data.Clear();
        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>) where * = <em>Observations Shape</em><br></br>
        /// <paramref name="value_targets"/> - <em>Vtarget</em> | Tensor(<em>Batch Size, 1</em>)
        /// </summary>
        private void UpdateValueNetwork(Tensor states, Tensor value_targets)
        {
            Tensor values = model.vNetwork.Forward(states);
            Loss mse = Loss.MSE(values, value_targets);
            float lossItem = mse.Item;

            if (float.IsNaN(lossItem))
                return;
            else
                criticLoss += lossItem;

            model.vOptimizer.ZeroGrad();
            model.vNetwork.Backward(mse.Gradient * 0.5f);
            model.vOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.vOptimizer.Step();


        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>)  where * = <em>Observations Shape (default: Space Size)</em><br></br>
        /// <paramref name="advantages"/> - <em>A</em> | Tensor(<em>Batch Size, 1</em>) <br></br>
        /// <paramref name="actions"/> - <em>a</em> | Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// <paramref name="piOld"/> - <em>πθold(a|s) </em>| Tensor (<em>Batch Size, Continuous Actions</em>) <br></br>
        /// </summary>
        private void UpdateContinuousNetwork(Tensor states, Tensor advantages, Tensor actions, Tensor piOld, out Tensor pi)
        {
            int batch_size = states.Rank >= 2 ? states.Size(0) : 1;
            int continuous_actions_num = actions.Size(-1);

            // Forwards pass
            Tensor mu, sigma;
            model.ContinuousForward(states, out mu, out sigma);
            pi = Tensor.Probability(actions, mu, sigma);
            Tensor sigmaSquared = sigma.Square();

            Tensor ratio = pi / piOld; // a.k.a pₜ(θ) = πθ(a|s) / πθold(a|s)

            // Compute L CLIP
            advantages = advantages.Expand(1, continuous_actions_num);
            Tensor LClip = Tensor.Minimum(
                                ratio * advantages,
                                Tensor.Clip(ratio, 1f - hp.epsilon, 1f + hp.epsilon) * advantages);

            float surrogateItem = LClip.Abs().ToArray().Average();
            if (float.IsNaN(surrogateItem))
            {
                ConsoleMessage.Warning($"PPO LCLIP batch containing NaN values was skipped. Consider clipping the observations strongly");
                return;
            }
            actorLoss += surrogateItem;


            // Computing ∂-LClip / ∂πθ(a|s)
            Tensor dmindx = Tensor.Zeros(batch_size, continuous_actions_num);
            Tensor dmindy = Tensor.Zeros(batch_size, continuous_actions_num);
            Tensor dclipdx = Tensor.Zeros(batch_size, continuous_actions_num);

            Parallel.For(0, batch_size, b =>
            {
                for (int a = 0; a < continuous_actions_num; a++)
                {
                    float pt = ratio[b, a];
                    float e = hp.epsilon;
                    float At = advantages[b, a];
                    float clip_pt = Math.Clamp(pt, 1f - e, 1f + e);

                    // ∂Min(x,y)/∂x
                    dmindx[b, a] = pt * At <= clip_pt * At ? 1f : 0f;

                    // ∂Min(x,y)/∂y
                    dmindy[b, a] = clip_pt * At < pt * At ? 1f : 0f;

                    // ∂Clip(x,a,b)/∂x
                    dclipdx[b, a] = 1f - e <= pt && pt <= 1f + e ? 1f : 0f;
                }

            });


            // ∂-LClip / ∂πθ(a|s)  (20) Bick.D
            Tensor dmLClip_dPi = -1f * (dmindx * advantages + dmindy * advantages * dclipdx) / piOld;

            // Entropy bonus added if σ is trainable
            // H(πθ(a|s)) = - integral( πθ(a|s) log πθ(a|s) ) = 1/2 * log(2πeσ^2) // https://en.wikipedia.org/wiki/Differential_entropy
            // Tensor H = 0.5f * Tensor.Log(2f * MathF.PI * MathF.E * sigma.Pow(2)); 
            entropy += sigma.ToArray().Average(); // H.ToArray().Average(); // i modified it because is simply to understand since it is sigma         

            // if (dmLClip_dPi.Contains(float.NaN)) return;


            // ∂πθ(a|s) / ∂μ = πθ(a|s) * (x - μ) / σ^2   (26) Bick.D
            Tensor dPi_dMu = pi * (actions - mu) / sigmaSquared;


            // ∂-LClip / ∂μ = (∂-LClip / ∂πθ(a|s)) * (∂πθ(a|s) / ∂μ)
            Tensor dmLClip_dMu = dmLClip_dPi * dPi_dMu;
            model.muOptimizer.ZeroGrad();
            model.muNetwork.Backward(dmLClip_dMu);
            model.muOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.muOptimizer.Step();

            if (model.standardDeviation == StandardDeviationType.Trainable)
            {
                // ∂πθ(a|s) / ∂σ = πθ(a|s) * ((x - μ)^2 - σ^2) / σ^3    (Simple statistical gradient-following for connectionst Reinforcement Learning (pag 14))
                Tensor dPi_dSigma = pi * ((actions - mu).Pow(2) - sigmaSquared) / (sigmaSquared * sigma);

                // ∂-H / ∂σ = -1/σ
                Tensor dmH_dSigma = sigma.Select(x => -1f / x);

                // ∂-LClip / ∂σ = (∂-LClip / ∂πθ(a|s)) * (∂πθ(a|s) / ∂σ) + β * (∂-H / ∂σ)
                Tensor dmLClip_dSigma = dmLClip_dPi * dPi_dSigma + hp.beta * dmH_dSigma;

                model.sigmaOptimizer.ZeroGrad();
                model.sigmaNetwork.Backward(dmLClip_dSigma);
                model.sigmaOptimizer.ClipGradNorm(hp.gradClipNorm);
                model.sigmaOptimizer.Step();
            }

        }
        /// <summary>
        /// <paramref name="states"/> - <em>s</em> | Tensor (<em>Batch Size, *</em>)  where * = <em>Observations Shape</em><br></br>
        /// <paramref name="advantages"/> - <em>A</em> | Tensor(<em>Batch Size, 1</em>) <br></br>
        /// <paramref name="actions"/> - <em>a</em> | Tensor (<em>Batch Size, Discrete Actions</em>) - One Hot Vectors<br></br> 
        /// <paramref name="piOld"/> - <em>πθold(a|s) </em>| Tensor (<em>Batch Size, Discrete Actions</em>) <br></br>
        /// </summary>
        private void UpdateDiscreteNetwork(Tensor states, Tensor advantages, Tensor actions, Tensor piOld, out Tensor pi)
        {
            // so pi is the probability of the specific action, but in my code pi is the same with phi it doesn't matter until differentiation
            // and phi is the vector of all actions probabilities
            // action is the one hot vector with actions sampled from phi
            // Somehow pi is phi here (because the probabilities of the actions are directly given by the network), a little confusion but you get it :)
            int batch_size = states.Rank >= 2 ? states.Size(0) : 1;
            int discrete_actions_num = piOld.Size(-1);

            model.DiscreteForward(states, out pi);

            Tensor ratio = pi / piOld;
            // Compute L CLIP
            advantages = advantages.Expand(1, discrete_actions_num);
            Tensor LClip = Tensor.Minimum(
                                ratio * advantages,
                                Tensor.Clip(ratio, 1 - hp.epsilon, 1 + hp.epsilon) * advantages);


            float surrogateItem = LClip.Abs().ToArray().Average();
            if (float.IsNaN(surrogateItem))
            {
                ConsoleMessage.Warning($"PPO LCLIP batch containing NaN values was skipped. Consider clipping the observations strongly");
                return;
            }
            actorLoss += surrogateItem;


            // Computing ∂-LClip / ∂πθ(a|s)
            Tensor dmindx = Tensor.Zeros(batch_size, discrete_actions_num);
            Tensor dmindy = Tensor.Zeros(batch_size, discrete_actions_num);
            Tensor dclipdx = Tensor.Zeros(batch_size, discrete_actions_num);

            Parallel.For(0, batch_size, b =>
            {
                for (int a = 0; a < discrete_actions_num; a++)
                {
                    float pt = ratio[b, a];
                    float e = hp.epsilon;
                    float At = advantages[b, a];
                    float clip_pt = Math.Clamp(pt, 1f - e, 1f + e);

                    // ∂Min(x,y)/∂x
                    dmindx[b, a] = pt * At <= clip_pt * At ? 1f : 0f;

                    // ∂Min(x,y)/∂y
                    dmindy[b, a] = clip_pt * At < pt * At ? 1f : 0f;

                    // ∂Clip(x,a,b)/∂x
                    dclipdx[b, a] = 1f - e <= pt && pt <= 1f + e ? 1f : 0f;
                }
            });

            // ∂-LClip / ∂πθ(a|s)  (20) Bick.D
            Tensor dmLClip_dPi = -1f * (dmindx * advantages + dmindy * advantages * dclipdx) / piOld;

            // ∂πθ(a|s) / ∂φ  (20) Bick.D, it's basically 1 if the same action was sampled and 0 if not :D
            Tensor dPi_dPhi = actions;


            // Entropy bonus for discrete actions
            // H = - φ * log(φ)
            Tensor H = -pi * pi.Log();
            entropy += H.ToArray().Average();

            // ∂-H / ∂φ = log(φ) + 1;
            Tensor dmH_dPhi = pi.Log() + 1;

            // ∂-L / ∂φ = (∂-L / ∂-πθ(a|s)) * (∂-πθ(a|s) / ∂φ) + β * (∂-H / ∂φ) 
            Tensor dmLClip_dPhi = dmLClip_dPi * dPi_dPhi + hp.beta * dmH_dPhi;

            model.discreteOptimizer.ZeroGrad();
            model.discreteNetwork.Backward(dmLClip_dPhi);
            model.discreteOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.discreteOptimizer.Step();
        }
        /// <summary>
        /// DKL(P(μ1,σ1) || Q(μ2, σ2)) = log(σ2,/σ1) + (σ2^2 + (μ1 - μ2)^2) / (2σ2^2) - 1/2 for norm distributions
        /// <br></br>
        /// Computes the KL Divergence between and the old and the new policy.
        /// </summary>
        /// <param name="probs_new">πθ(a|s)</param>
        /// <param name="probs_old">πθold(a|s)</param>
        private static float ComputeKLDivergence(Tensor probs_new, Tensor probs_old)
        {
            Tensor KL = probs_new * Tensor.Log(probs_new / (probs_old + Utils.EPSILON));
            KL = KL.Mean(0);
            KL = KL.Sum(0);
            return KL[0];
        }
        /// <summary>
        /// This method computes the generalized advantage estimation and value function targets.
        /// </summary>
        /// <param name="GAMMA"></param>
        /// <param name="LAMBDA"></param>
        /// <param name="HORIZON"></param>
        /// <param name="valueNetwork"></param>
        private static void ComputeGAE_andVtargets(in MemoryBuffer memory, float GAMMA, float LAMBDA, int HORIZON, Sequential valueNetwork)
        {
            // Well i think we can compute this separately for each agent in a multihreaded way.. but i m afraid predict will not work well so it's fine
            //4.43 max threads
            // 4.49 8 threads
            // 4.83 4 threads
            // 4.70 no parallel
            var frames = memory.frames;
            int T = memory.Count;
            Tensor[] all_states_plus_lastNextState = new Tensor[T + 1];
            for (int i = 0; i < T; i++)
                all_states_plus_lastNextState[i] = frames[i].state;
            all_states_plus_lastNextState[T] = frames[T - 1].nextState;

            // Vw_s has length of T + 1
            float[] Vw_s = valueNetwork.Predict(Tensor.Concat(null, all_states_plus_lastNextState)).ToArray();

            // Vw_s = Tensor.FilterNaN(Vw_s, 0); it happpen in some cases to get NaN in the first timestep and then all are gonna be NaN


            // Generalized Advantage Estimation
            Parallel.For(0, T, timestep =>
            {
                float discount = 1f;
                float Ahat_t = 0f;
                for (int t = timestep; t < MathF.Min(t + HORIZON, T); t++)
                {
                    float r_t = frames[t].reward[0];
                    float V_st = Vw_s[t];
                    float V_next_st = frames[t].done[0] == 1 ? 0 : Vw_s[t + 1];  // if the state is terminal, next value is set to 0.

                    float delta_t = r_t + GAMMA * V_next_st - V_st;
                    Ahat_t += discount * delta_t;
                    discount *= GAMMA * LAMBDA;

                    if (frames[t].done[0] == 1)
                        break;
                }

                // Vtarg[t] = GAE(gamma, lambda, t) + V[t]
                float Vtarget_t = Ahat_t + Vw_s[timestep];

                frames[timestep].value_target = Tensor.Constant(Vtarget_t);
                frames[timestep].advantage = Tensor.Constant(Ahat_t);
            });
        }
        /// <summary>
        /// This method normalizes the advantages for a minibatch
        /// </summary>
        private static Tensor NormalizeAdvantages(Tensor advantages)
        {
            float std = advantages.Std(0, 1)[0]; // note that we use biased estimator
            float mean = advantages.Mean(0)[0];
            return (advantages - mean) / (std + Utils.EPSILON);

        }
    }
#if UNITY_EDITOR
    [CustomEditor(typeof(PPOTrainer), true), CanEditMultipleObjects]
    sealed class CustomPPOTrainerEditor : Editor
    {
        static string[] dontDrawMe = new string[] { "m_Script" };
        public override void OnInspectorGUI()
        {
            DrawPropertiesExcluding(serializedObject, dontDrawMe);
            serializedObject.ApplyModifiedProperties();
        }
    }
#endif
}

