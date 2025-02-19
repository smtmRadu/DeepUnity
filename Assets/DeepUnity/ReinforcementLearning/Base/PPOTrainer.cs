using System;
using System.Collections.Generic;
using UnityEditor;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using DeepUnity.Models;
using DeepUnity.Activations;
using DeepUnity.Optimizers;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// [1] https://fse.studenttheses.ub.rug.nl/25709/1/mAI_2021_BickD.pdf
    /// [2] https://link.springer.com/article/10.1007/BF00992696
    /// [3] https://ieeexplore.ieee.org/document/9520424
    /// </summary>

    internal sealed class PPOTrainer : DeepUnityTrainer, IOnPolicy
    {
        //Internal PPO Config
        //Note that AdamW+Clipping was changed to StableAdamW. On tests it is way better than what was previously used.
        const float epsilon = 1e-5F; // PPO openAI eps they use :D, but in Andrychowicz et al. (2021) they use TF default 1e-7
        const float valueWD = 0.0F; // Value net weight decay (AdamW)
        const bool amsGrad = false;

        public Optimizer optim_v { get; set; }   
        public Optimizer optim_mu { get; set; }
        public Optimizer optim_sigma { get; set; }
        public Optimizer optim_discrete { get; set; }

        // Cache networks for KLE-Rollback. In case it is used.
        private Tensor[] v_kle_cache { get; set; }
        private Tensor[] mu_kle_cache { get; set; }
        private Tensor[] sigma_kle_cache { get; set; }
        private Tensor[] disc_kle_cache { get; set; }
       
        protected override void Initialize()
        {    
            // MEAN Tanh module is removed in PPO
            if (model.IsUsingContinuousActions && model.muNetwork.Modules.Last().GetType() == typeof(Tanh))
                model.muNetwork.Modules = model.muNetwork.Modules.Take(model.muNetwork.Modules.Length - 1).ToArray();


            // Initialize Optimizers & Schedulers
            int total_epochs = (int)hp.maxSteps / hp.bufferSize * hp.numEpoch; // THIS IS FOR PPO, but for now i will let it for SAC as well       
            // optim_v = new AdamW(model.vNetwork.Parameters(), hp.criticLearningRate, eps: epsilon, weight_decay: valueWD, amsgrad:amsGrad, fused:true);
            optim_v = new StableAdamW(model.vNetwork.Parameters(), hp.criticLearningRate, eps: epsilon, weight_decay: valueWD, fused: true);
            optim_v.Scheduler = new LinearAnnealing(optim_v, start_factor: 1f, end_factor: 0f, total_iters: total_epochs);

            if (model.IsUsingContinuousActions)
            {
                //optim_mu = new AdamW(model.muNetwork.Parameters(), hp.actorLearningRate, eps: epsilon, amsgrad: amsGrad, weight_decay:0F, fused:true);
                optim_mu = new StableAdamW(model.muNetwork.Parameters(), hp.actorLearningRate, eps: epsilon,weight_decay: 0F, fused:true);
                optim_mu.Scheduler = new LinearAnnealing(optim_mu, start_factor: 1f, end_factor: 0f, total_iters: total_epochs);

                // optim_sigma = new AdamW(model.sigmaNetwork.Parameters(), hp.actorLearningRate, eps: epsilon, amsgrad: amsGrad, weight_decay: 0F, fused: true);
                optim_sigma = new StableAdamW(model.sigmaNetwork.Parameters(), hp.actorLearningRate, eps: epsilon, weight_decay: 0F, fused: true);
                optim_sigma.Scheduler = new LinearAnnealing(optim_sigma, start_factor: 1f, end_factor: 0f, total_iters: total_epochs);
            }

            if (model.IsUsingDiscreteActions)
            {
                // optim_discrete = new AdamW(model.discreteNetwork.Parameters(), hp.actorLearningRate, eps: epsilon, amsgrad: amsGrad, weight_decay: 0F, fused: true);
                optim_discrete = new StableAdamW(model.muNetwork.Parameters(), hp.actorLearningRate, eps: epsilon, weight_decay: 0F, fused: true);
                optim_discrete.Scheduler = new LinearAnnealing(optim_discrete, start_factor: 1f, end_factor: 0f, total_iters: total_epochs);
            }


            // Initialize inference device
            if (model.muNetwork != null)
                model.muNetwork.Device = model.inferenceDevice;

            if (model.sigmaNetwork != null)
                model.sigmaNetwork.Device = model.inferenceDevice;

            if (model.discreteNetwork != null)
                model.discreteNetwork.Device = model.inferenceDevice;

            if (model.stochasticity != Stochasticity.FixedStandardDeviation || model.stochasticity != Stochasticity.TrainebleStandardDeviation)           
                model.stochasticity = Stochasticity.FixedStandardDeviation;
            
        }
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
                    train_data.TryAppend(agent_memory.frames, hp.bufferSize);
                    // if (hp.debug) Utils.DebugInFile(agent_memory.ToString());
                    agent_memory.Clear();
                }

                actorLoss = 0;
                criticLoss = 0;

                updateBenchmarkClock = Stopwatch.StartNew();       
                Train();
                updateBenchmarkClock.Stop();

                updateIterations++;
                actorLoss = actorLoss / (hp.bufferSize / hp.batchSize * hp.numEpoch);
                criticLoss = criticLoss / (hp.bufferSize / hp.batchSize * hp.numEpoch);
                entropy = entropy / (hp.bufferSize / hp.batchSize * hp.numEpoch);
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

                // unpack & split train_data into minibatches (parallel is faster)
                LinkedList<Task<Tensor[]>> tasks = new LinkedList<Task<Tensor[]>>();

                Task<Tensor[]> statesTask = Task.Run(() =>
                    Utils.Split(train_data.States, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                );
                tasks.AddLast(statesTask);
                Task<Tensor[]> advantagesTask = Task.Run(() =>
                    Utils.Split(train_data.Advantages, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                );
                tasks.AddLast(advantagesTask);
                Task<Tensor[]> valueTargetsTask = Task.Run(() =>
                    Utils.Split(train_data.ValueTargets, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                );
                tasks.AddLast(valueTargetsTask);
                Task<Tensor[]> contActTask   = null;
                Task<Tensor[]> contProbsTask = null;
                Task<Tensor[]> discActTask   = null;
                Task<Tensor[]> discProbsTask = null;
                if (model.IsUsingContinuousActions)
                {
                    contActTask = Task.Run(() =>
                        Utils.Split(train_data.ContinuousActions, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                    );

                    contProbsTask = Task.Run(() =>
                        Utils.Split(train_data.ContinuousProbabilities, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                    );
                    tasks.AddLast(contActTask);
                    tasks.AddLast(contProbsTask);
                }
                

                if(model.IsUsingDiscreteActions)
                {
                    discActTask = Task.Run(() =>
                                        Utils.Split(train_data.DiscreteActions, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                                    );

                    discProbsTask = Task.Run(() =>
                        Utils.Split(train_data.DiscreteProbabilities, hp.batchSize).Select(x => Tensor.Concat(null, x)).ToArray()
                    );
                    tasks.AddLast(discActTask);
                    tasks.AddLast(discProbsTask);
                }

                Task.WaitAll(tasks.ToArray());
                Tensor[] states_batches = statesTask.Result;
                Tensor[] advantages_batches = advantagesTask.Result;
                Tensor[] value_targets_batches = valueTargetsTask.Result;
                Tensor[] cont_act_batches = model.IsUsingContinuousActions ? contActTask.Result : null;
                Tensor[] cont_probs_batches = model.IsUsingContinuousActions ? contProbsTask.Result : null;
                Tensor[] disc_act_batches = model.IsUsingDiscreteActions ? discActTask.Result : null;
                Tensor[] disc_probs_batches = model.IsUsingDiscreteActions ? discProbsTask.Result : null;


                // θ new. New probabilities of the policy used for early stopping/rollback
                Tensor cont_probs_new_kle = null;
                Tensor cont_probs_old_kle = null;
                Tensor disc_probs_new_kle = null;
                Tensor disc_probs_old_kle = null;

                // Cache params[t-1] in case kl_div > d_targ
                if(hp.KLDivergence == KLEType.Rollback)
                {
                    LinkedList<Task> tasks_kle = new();

                    tasks_kle.AddLast(Task.Run(() => v_kle_cache = model.vNetwork.Parameters().Select(x => x.param.Clone() as Tensor).ToArray()));
                    if(model.IsUsingContinuousActions)
                    {
                        tasks_kle.AddLast(Task.Run(() => mu_kle_cache = model.muNetwork.Parameters().Select(x => x.param.Clone() as Tensor).ToArray()));

                        if(model.stochasticity == Stochasticity.TrainebleStandardDeviation)
                            tasks_kle.AddLast(Task.Run(() => sigma_kle_cache = model.sigmaNetwork.Parameters().Select(x => x.param.Clone() as Tensor).ToArray()));
         
                    }
                    if(model.IsUsingDiscreteActions)
                    {
                        tasks_kle.AddLast(Task.Run(() => disc_kle_cache = model.sigmaNetwork.Parameters().Select(x => x.param.Clone() as Tensor).ToArray()));
                    }

                    Task.WaitAll(tasks_kle.ToArray());
                }

                for (int b = 0; b < states_batches.Length; b++)
                {
                    Tensor normalized_advantages = hp.normalizeAdvantages ? NormalizeAdvantages(advantages_batches[b]) : advantages_batches[b];// Advantage normalization is shit when applied on small batches

                    UpdateValueNetwork(states_batches[b], value_targets_batches[b]);

                    if (model.IsUsingContinuousActions)
                    {
                        cont_probs_old_kle = cont_probs_batches[b];
                        UpdateContinuousNetwork(
                            states_batches[b],
                            normalized_advantages,
                            cont_act_batches[b],
                            cont_probs_batches[b],
                            out cont_probs_new_kle);
                    }
                    if (model.IsUsingDiscreteActions)
                    {
                        disc_probs_old_kle = disc_probs_batches[b];
                        UpdateDiscreteNetwork(
                            states_batches[b],
                            normalized_advantages,
                            disc_act_batches[b],
                            disc_probs_batches[b],
                            out disc_probs_new_kle);
                    }

                }

                // Check KL Divergence based on the last Minibatch (see [3])
                if (hp.KLDivergence != KLEType.Off)
                {
                    // Though even if i should stop for them separatelly (i mean i can let for one to continue the training if kl is small) i will let it simple..
                    float kldiv_cont = model.IsUsingContinuousActions ? ComputeKLDivergence(cont_probs_new_kle, cont_probs_old_kle) : 0;
                    float kldiv_disc = model.IsUsingDiscreteActions ? ComputeKLDivergence(disc_probs_new_kle, disc_probs_old_kle) : 0;

                    if (kldiv_cont > hp.targetKL || kldiv_disc > hp.targetKL)
                    {
                        // ConsoleMessage.Info($"<b>KLE-{hp.KLDivergence}</b> triggered in epoch {epoch_index + 1}/{hp.numEpoch}\n [KL_continuous: {kldiv_cont} | KL_discrete: {kldiv_disc}) | KL_target: {hp.targetKL}]");
                        
                        if (hp.KLDivergence == KLEType.Stop)
                            break;
                        else if (hp.KLDivergence == KLEType.Rollback)
                        {
                            LinkedList<Task> tasks_kle = new();

                            tasks_kle.AddLast(Task.Run(() =>
                            {
                                foreach (var (cache, current) in v_kle_cache.Zip(model.vNetwork.Parameters(), (x, y) => (x, y.param)))
                                {
                                    Tensor.CopyTo(cache, current);
                                }
                            }));
                           
                            if (model.IsUsingContinuousActions)
                            {
                                tasks_kle.AddLast(Task.Run(() =>
                                {
                                    foreach (var (cache, current) in mu_kle_cache.Zip(model.muNetwork.Parameters(), (x, y) => (x, y.param)))
                                    {
                                        Tensor.CopyTo(cache, current);
                                    }
                                }));
                                if (model.stochasticity == Stochasticity.TrainebleStandardDeviation)
                                    tasks_kle.AddLast(Task.Run(() =>
                                    {
                                        foreach (var (cache, current) in sigma_kle_cache.Zip(model.sigmaNetwork.Parameters(), (x, y) => (x, y.param)))
                                        {
                                            Tensor.CopyTo(cache, current);
                                        }
                                    }));
                                                        
                            }
                            if (model.IsUsingDiscreteActions)
                            {
                                tasks_kle.AddLast(Task.Run(() =>
                                {
                                    foreach (var (cache, current) in disc_kle_cache.Zip(model.discreteNetwork.Parameters(), (x, y) => (x, y.param)))
                                    {
                                        Tensor.CopyTo(cache, current);
                                    }
                                }));
                            }

                            Task.WaitAll(tasks_kle.ToArray());
                            break;
                        }
                        else
                            throw new NotImplementedException("Unhandles KLE type");
                    }
                    // for rollback is the same but we need to cache the old state of the network and it is costly....
                }

                // Step LR schedulers after each epoch (this allows selection at runtime)
                if (hp.LRSchedule)
                {
                    optim_v.Scheduler.Step();
                    optim_mu?.Scheduler.Step();
                    optim_sigma?.Scheduler.Step();
                    optim_discrete?.Scheduler.Step();
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
            float lossItem = mse.Item * hp.valueCoeff;

            if (float.IsNaN(lossItem))
                return;
            else
                criticLoss += lossItem;

            optim_v.ZeroGrad();
            model.vNetwork.Backward(mse.Grad * hp.valueCoeff);
            // optim_v.ClipGradNorm(hp.maxNorm);
            optim_v.Step();


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

            float surrogateItem = LClip.Abs().Average();
            if (float.IsNaN(surrogateItem))
            {
                ConsoleMessage.Warning($"PPO LCLIP batch containing NaN values was skipped. Consider clipping the observations");
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
            entropy += sigma.Average(); // H.Average(); // i modified it because is simply to understand since it is sigma         

            // if (dmLClip_dPi.Contains(float.NaN)) return;


            // ∂πθ(a|s) / ∂μ = πθ(a|s) * (x - μ) / σ^2   (26) Bick.D
            Tensor dPi_dMu = pi * (actions - mu) / sigmaSquared;


            // ∂-LClip / ∂μ = (∂-LClip / ∂πθ(a|s)) * (∂πθ(a|s) / ∂μ)
            Tensor dmLClip_dMu = dmLClip_dPi * dPi_dMu;
            optim_mu.ZeroGrad();
            model.muNetwork.Backward(dmLClip_dMu);
            // optim_mu.ClipGradNorm(hp.maxNorm);
            optim_mu.Step();

            if (model.stochasticity == Stochasticity.TrainebleStandardDeviation)
            {
                // ∂πθ(a|s) / ∂σ = πθ(a|s) * ((x - μ)^2 - σ^2) / σ^3    (Simple statistical gradient-following for connectionst Reinforcement Learning (pag 14))
                Tensor dPi_dSigma = pi * ((actions - mu).Square() - sigmaSquared) / (sigmaSquared * sigma);

                // ∂-H / ∂σ = -1/σ
                Tensor dmH_dSigma = sigma.Select(x => -1f / x);

                // ∂-LClip / ∂σ = (∂-LClip / ∂πθ(a|s)) * (∂πθ(a|s) / ∂σ) + β * (∂-H / ∂σ)
                Tensor dmLClip_dSigma = dmLClip_dPi * dPi_dSigma + hp.beta * dmH_dSigma;

                optim_sigma.ZeroGrad();
                model.sigmaNetwork.Backward(dmLClip_dSigma);
                // optim_sigma.ClipGradNorm(hp.maxNorm);
                optim_sigma.Step();
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


            float surrogateItem = LClip.Abs().Average();
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
            entropy += H.Average();

            // ∂-H / ∂φ = log(φ) + 1;
            Tensor dmH_dPhi = pi.Log() + 1;

            // ∂-L / ∂φ = (∂-L / ∂-πθ(a|s)) * (∂-πθ(a|s) / ∂φ) + β * (∂-H / ∂φ) 
            Tensor dmLClip_dPhi = dmLClip_dPi * dPi_dPhi + hp.beta * dmH_dPhi;

            optim_discrete.ZeroGrad();
            model.discreteNetwork.Backward(dmLClip_dPhi);
            // optim_discrete.ClipGradNorm(hp.maxNorm);
            optim_discrete.Step();
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
            KL = KL.Mean(0).Sum(0);
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

                frames[timestep].v_target = Tensor.Constant(Vtarget_t);
                frames[timestep].advantage = Tensor.Constant(Ahat_t);
            });
        }
        /// <summary>
        /// This method normalizes the advantages for a minibatch
        /// </summary>
        private static Tensor NormalizeAdvantages(Tensor advantages)
        {
            float std = advantages.Std(0, correction: 0)[0]; // note that we use biased estimator
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

