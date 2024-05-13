using DeepUnity.Models;
using System;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using DeepUnity.Optimizers;
using DeepUnity.Modules;
using DeepUnity.Activations;
using UnityEditor.Tilemaps;

namespace DeepUnity.ReinforcementLearning

{
    internal class TD3Trainer : DeepUnityTrainer
    {
        private static Sequential Qtarg1;
        private static Sequential Qtarg2;
        private static Sequential Pitarg;

        public Optimizer optim_q1 { get; set; }
        public Optimizer optim_q2 { get; set; }
        public Optimizer optim_mu { get; set; }

        public LRScheduler scheduler_q1 { get; set; }
        public LRScheduler scheduler_q2 { get; set; }
        public LRScheduler scheduler_mu { get; set; }


        private int new_experiences_collected = 0;
        private int qFunctionUpdates = 0; // used to track num of phi steps to check for policy delay
        protected override void Initialize()
        {
            if (model.muNetwork.Modules.Last().GetType() != typeof(Tanh))
                model.muNetwork.Modules = model.muNetwork.Modules.Concat(new IModule[] { new Tanh() }).ToArray();


            // Initialize optimizers
            optim_q1 = new Adam(model.q1Network.Parameters(), hp.criticLearningRate);
            optim_q2 = new Adam(model.q2Network.Parameters(), hp.criticLearningRate);
            optim_mu = new Adam(model.muNetwork.Parameters(), hp.actorLearningRate);

            // Initialize schedulers
            scheduler_q1 = new LinearLR(optim_q1, start_factor: 1f, end_factor: 0f, epochs: (int)model.config.maxSteps);
            scheduler_q2 = new LinearLR(optim_q2, start_factor: 1f, end_factor: 0f, epochs: (int)model.config.maxSteps);
            scheduler_mu = new LinearLR(optim_mu, start_factor: 1f, end_factor: 0f, epochs: (int)model.config.maxSteps);

            // Initialize target networks
            Qtarg1 = model.q1Network.Clone() as Sequential;
            Qtarg2 = model.q2Network.Clone() as Sequential;
            Pitarg = model.muNetwork.Clone() as Sequential;

            // Set devices
            model.muNetwork.Device = model.inferenceDevice;

            Qtarg1.Device = model.trainingDevice;
            Qtarg2.Device = model.trainingDevice;
            Pitarg.Device = model.trainingDevice;
            model.q1Network.Device = model.trainingDevice;
            model.q2Network.Device = model.trainingDevice;

            // Set initial random actions
            model.stochasticity = Stochasticity.Random;

            if (hp.updateAfter < hp.minibatchSize * 5)
            {
                ConsoleMessage.Info("'Update After' was set higher than the 'batch size'");
                hp.updateAfter = hp.minibatchSize * 5;
            }
            if (model.normalize)
            {
                ConsoleMessage.Info("Online Normalization is not available for off-policy algorithms");
                model.normalize = false;
            }

        }

        protected override void OnBeforeFixedUpdate()
        {
            int decision_freq = parallelAgents[0].DecisionRequester.decisionPeriod;
            new_experiences_collected += parallelAgents.Count;
            if (new_experiences_collected >= hp.updateInterval * decision_freq)
            {
                // if the buffer will be full after this collection, let's clear the old values
                if (train_data.Count >= hp.replayBufferSize)
                    train_data.frames.RemoveRange(0, train_data.Count / 2); // If buffer is full, remove old half

                foreach (var agent_mem in parallelAgents.Select(x => x.Memory))
                {
                    if (agent_mem.Count == 0)
                        continue;

                    train_data.TryAppend(agent_mem, hp.replayBufferSize);
                    agent_mem.Clear();
                }

                if (train_data.Count >= hp.updateAfter)
                {
                    model.stochasticity = Stochasticity.ActiveNoise;

                    actorLoss = 0;
                    criticLoss = 0;

                    updateClock = Stopwatch.StartNew();
                    Train();
                    updateClock.Stop();

                    updateIterations++;
                    actorLoss /= hp.updatesNum;
                    criticLoss /= hp.updatesNum;
                    entropy = model.noiseValue;
                    currentSteps += new_experiences_collected / decision_freq;
                    new_experiences_collected = 0;
                }
            }
        }

        private void Train()
        {
            model.muNetwork.Device = model.trainingDevice;

            for (int epoch_index = 0; epoch_index < hp.updatesNum; epoch_index++)
            {
                // Sample a random batch of transitions from the replay buffer
                TimestepTuple[] batch = Utils.Random.Sample(hp.minibatchSize, train_data.frames);

                // Batchify
                Tensor states = Tensor.Concat(null, batch.Select(x => x.state).ToArray());
                Tensor states_prime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());
                Tensor actions = Tensor.Concat(null, batch.Select(x => x.action_continuous).ToArray());

               
                Tensor q_targets;
                Tensor a_prime_s_prime;
                ComputeTargetActions(states_prime, out a_prime_s_prime);
                ComputeQTargets(batch, states_prime, a_prime_s_prime, out q_targets);
                UpdateQFunctions(states, actions, q_targets);
                qFunctionUpdates++;

                if(qFunctionUpdates % hp.policyDelay == 0)
                {
                    UpdatePolicy(states);
                    UpdateTargetNetworks();
                }

                if (hp.LRSchedule)
                {
                    scheduler_q1.Step();
                    scheduler_q2.Step();
                    scheduler_mu.Step();
                }

            }
            model.muNetwork.Device = model.inferenceDevice;
        }

       
        public void ComputeTargetActions(Tensor sPrime, out Tensor aPrime_sPrime)
        {
            Tensor muTarg_sPrime = Pitarg.Predict(sPrime);
            Tensor xi = Tensor.RandomNormal((0, model.noiseValue), muTarg_sPrime.Shape).Clip(-hp.noiseClip, hp.noiseClip);

            aPrime_sPrime = (muTarg_sPrime + xi).Clip(-1f, 1f);
        }
        private void ComputeQTargets(TimestepTuple[] batch, Tensor sPrime, Tensor aPrime_sPrime, out Tensor y)
        {
            Tensor paired_sPrime_aPrime = Pairify(sPrime, aPrime_sPrime);
            Tensor Qtarg1_sPrime_aTildePrime = Qtarg1.Predict(paired_sPrime_aPrime); // (B, 1)
            Tensor Qtarg2_sPrime_aTildePrime = Qtarg2.Predict(paired_sPrime_aPrime); // (B, 1)

            for (int b = 0; b < batch.Length; b++)
            {
                // y(r,s',d) = r + Ɣ(1 - d)[min(Q1t(s',ã'), Q2t(s',ã'))]

                float r = batch[b].reward[0];
                float d = batch[b].done[0];
                float Qt1_sa = Qtarg1_sPrime_aTildePrime[b, 0];
                float Qt2_sa = Qtarg2_sPrime_aTildePrime[b, 0];

                float y_ = r + hp.gamma * (1f - d) * MathF.Min(Qt1_sa, Qt2_sa);

                batch[b].q_target = Tensor.Constant(y_);
            }

            y = Tensor.Concat(null, batch.Select(x => x.q_target).ToArray());
        }
        private void UpdateQFunctions(Tensor states, Tensor actions, Tensor y)
        {
            // Update Q functions          
            // ∇φ = (Qφ(s,a) - y(r,s',d)^2
            Tensor stateActionPair = Pairify(states, actions);
            Tensor Q1_s_a = model.q1Network.Forward(stateActionPair);
            Tensor Q2_s_a = model.q2Network.Forward(stateActionPair);

            Loss q1Loss = Loss.MSE(Q1_s_a, y);
            Loss q2Loss = Loss.MSE(Q2_s_a, y);
            criticLoss += (q1Loss.Item + q2Loss.Item) / 2;

            optim_q1.ZeroGrad();
            optim_q2.ZeroGrad();
            model.q1Network.Backward(q1Loss.Gradient);
            model.q2Network.Backward(q2Loss.Gradient);
            optim_q1.Step();
            optim_q2.Step();
        }
        private void UpdatePolicy(Tensor states)
        {
            // ObjectiveLoss = 1/|B| Σ Q1(s, mu(s))
            Tensor mu_s = model.muNetwork.Forward(states);
            Tensor q1_s_mu_s = model.q1Network.Forward(Pairify(states, mu_s));
            actorLoss += q1_s_mu_s.Average();

            Tensor objectiveLossGrad = -Tensor.Ones(q1_s_mu_s.Shape); // gradient ascent (grad of Mean()) -> mean is computed already on each layer (mean of the batch)
            Tensor s_mu_s_grad = model.q1Network.Backward(objectiveLossGrad);
            Tensor mu_grad = ExtractActionFromStateAction(s_mu_s_grad, states.Size(-1), mu_s.Size(-1));
            
            optim_mu.ZeroGrad();
            model.muNetwork.Backward(mu_grad);
            optim_mu.Step();
        }
        public void UpdateTargetNetworks()
        {
            Tensor[] theta = model.muNetwork.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi1 = model.q1Network.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi2 = model.q2Network.Parameters().Select(x => x.param).ToArray();

            Tensor[] theta_targ = Pitarg.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi_targ1 = Qtarg1.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi_targ2 = Qtarg2.Parameters().Select(x => x.param).ToArray();

            // We update the target q functions and theta softly...
            // OpenAI algorithm uses polyak = 0.995, the same thing with using τ = 0.005
            // φtarg,i <- (1 - τ)φtarg,i + τφi     for i = 1,2

            for (int i = 0; i < theta.Length; i++)
            {
                Tensor.CopyTo((1f - hp.tau) * theta_targ[i] + hp.tau * theta[i], theta_targ[i]);
            }

            for (int i = 0; i < phi1.Length; i++)
            {           
                Tensor.CopyTo((1f - hp.tau) * phi_targ1[i] + hp.tau * phi1[i], phi_targ1[i]);
                Tensor.CopyTo((1f - hp.tau) * phi_targ2[i] + hp.tau * phi2[i], phi_targ2[i]);
            }

            
        }




        /// <summary>
        /// Note that this must be changed if i plan to allow different input shape than vectorized. (for cnn for example)
        /// </summary>
        private static Tensor ExtractActionFromStateAction(Tensor stateActionBatch, int state_size, int action_size)
        {
            int batch_size = stateActionBatch.Size(0);
            Tensor actions = Tensor.Zeros(batch_size, action_size);
            Parallel.For(0, batch_size, b =>
            {
                for (int f = 0; f < action_size; f++)
                {
                    actions[b, f] = stateActionBatch[b, state_size + f];
                }
            });
            return actions;
        }
        private static Tensor Pairify(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = actionBatch.Size(1);
            Tensor pair = Tensor.Zeros(batch_size, state_size + action_size);
            Parallel.For(0, batch_size, b =>
            {
                for (int s = 0; s < state_size; s++)
                {
                    pair[b, s] = stateBatch[b, s];
                }
                for (int a = 0; a < action_size; a++)
                {
                    pair[b, state_size + a] = actionBatch[b, a];
                }
            });
            return pair;
        }
    }

}


