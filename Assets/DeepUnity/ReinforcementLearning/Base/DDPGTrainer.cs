using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using DeepUnity.Optimizers;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// DDPG paper hyperparam:
    /// Adam, lr -> 1e-4 & 1e-3 for actor and critic
    /// WD for Q = 1e-2
    /// gamma = .99
    /// tau = 1e-3
    /// actor ends with tanh (as we do)
    /// relu activations
    /// batch-size = 64
    /// buffer = 1e6
    /// Ornstein-Uhlenbeck process -> theta = 0.15, sigma= 0.2
    /// </summary>
    internal sealed class DDPGTrainer : DeepUnityTrainer, IOffPolicy
    {
        //Internal DDPG Config
        const float epsilon = 1e-7F; 
        const float valueWD = 0.01F; // Value net weight decay (AdamW)
        const bool amsGrad = false;


        private Sequential Q_targ;
        private Sequential Mu_targ;

        public Optimizer optim_q1 { get; set; }
        public Optimizer optim_mu { get; set; }

        private int new_experiences_collected = 0;

        protected override void Initialize()
        {
            if (model.muNetwork.Modules.Last().GetType() != typeof(Tanh))
            {
                model.muNetwork.Modules = model.muNetwork.Modules.Concat(new IModule[] { new Tanh() }).ToArray();
                model.Save();
            }             

            // Initialize optimizers
            optim_q1 = new AdamW(model.q1Network.Parameters(), hp.criticLearningRate, weight_decay: valueWD, amsgrad:amsGrad);
            optim_mu = new Adam(model.muNetwork.Parameters(), hp.actorLearningRate, amsgrad: amsGrad);

            // Initialize schedulers
            optim_q1.Scheduler = new LinearLR(optim_q1, start_factor: 1f, end_factor: 0f, total_iters: (int)hp.maxSteps * hp.updatesNum / hp.updateInterval);
            optim_mu.Scheduler = new LinearLR(optim_mu, start_factor: 1f, end_factor: 0f, total_iters: (int)hp.maxSteps * hp.updatesNum / hp.updateInterval);

            // Initialize target networks
            Q_targ = model.q1Network.Clone() as Sequential;
            Mu_targ = model.muNetwork.Clone() as Sequential;

            // Set devices
            model.muNetwork.Device = model.inferenceDevice;
            model.q1Network.Device = model.trainingDevice;

            Q_targ.Device = model.trainingDevice;
            Mu_targ.Device = model.trainingDevice;

            // Use random actions initialliy
            model.stochasticity = Stochasticity.Random; // random until reaching 'UpdateAfter' steps

            if (hp.updateAfter < hp.minibatchSize * 5)
            {
                ConsoleMessage.Info("'Update After' was set higher than the 'batch size'");
                hp.updateAfter = hp.minibatchSize * 5;
            }
            if(model.normalize)
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
                if (train_data.Count > hp.replayBufferSize)
                    train_data.frames.RemoveRange(0, train_data.Count / 2); // If buffer is full, remove old half

                
                foreach (var agent_mem in parallelAgents.Select(x => x.Memory))
                {
                    if (agent_mem.Count == 0)
                        continue;

                    train_data.TryAppend(agent_mem.frames, hp.replayBufferSize);
                    agent_mem.Clear();
                }
                
                if (train_data.Count >= hp.updateAfter)
                {
                    model.stochasticity = Stochasticity.ActiveNoise;

                    actorLoss = 0;
                    criticLoss = 0;

                    updateBenchmarkClock = Stopwatch.StartNew();        
                    Train();
                    updateBenchmarkClock.Stop();

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
                TimestepTuple[] minibatch = Utils.Random.Sample(hp.minibatchSize, train_data.frames);            

                // Batchify
                Tensor stateBatch = Tensor.Concat(null, minibatch.Select(x => x.state).ToArray());             
                Tensor actionBatch = Tensor.Concat(null, minibatch.Select(x => x.action_continuous).ToArray());

                Tensor yBatch;
                ComputeQTargets(minibatch, out yBatch);
                UpdateQFunctions(stateBatch, actionBatch, yBatch);
                UpdatePolicy(stateBatch);
                UpdateTargetNetworks();

                if (hp.LRSchedule)
                {
                    optim_q1.Scheduler.Step();
                    optim_mu.Scheduler.Step();
                }
            }

            model.muNetwork.Device = model.inferenceDevice;
        }

        private void ComputeQTargets(TimestepTuple[] batch, out Tensor criticTargets)
        {
            Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());
            Tensor muTarg_sPrime = Mu_targ.Predict(sPrime);
            Tensor qTarg_Prime = Q_targ.Predict(Pairify(sPrime, muTarg_sPrime)); // (B, 1)

            Parallel.For(0, batch.Length, b =>
            {
                // y(r,s',d) = r + Ɣ(1 - d) * Qφt(s',μθt(s'))

                float r = batch[b].reward[0];
                float d = batch[b].done[0];
                float qt_ = qTarg_Prime[b, 0];
                print(d);
                float y = r + hp.gamma * (1f - d) * qt_;

                batch[b].q_target = Tensor.Constant(y);
            });

            criticTargets = Tensor.Concat(null, batch.Select(x => x.q_target).ToArray());
        }
        private void UpdateQFunctions(Tensor states, Tensor actions, Tensor y)
        {
            // Update Q functions          
            // ∇φ = (Qφ(s,a) - y(r,s',d))^2
            Tensor Q_sa = model.q1Network.Forward(Pairify(states, actions));
            Loss loss = Loss.MSE(Q_sa, y);
            criticLoss += loss.Item;
           
            optim_q1.ZeroGrad();
            model.q1Network.Backward(loss.Grad);
            optim_q1.ClipGradNorm(0.5f);
            optim_q1.Step();
        }
        private void UpdatePolicy(Tensor states)
        {
            // ObjectiveLoss = 1/|B| Σb Q(s, μ(s))
            Tensor mu_s = model.muNetwork.Forward(states);
            Tensor q_s_mu_s = model.q1Network.Forward(Pairify(states, mu_s));
            actorLoss += -q_s_mu_s.Average();

            Tensor lossGrad = Tensor.Fill(-1f, q_s_mu_s.Shape); //  (grad of Mean() is 1/n but the framework already takes it as mean)
            model.q1Network.RequiresGrad = false;
            Tensor qinput_grad = model.q1Network.Backward(lossGrad); //phi params are 'fixed'
            model.q1Network.RequiresGrad = true;
            Tensor mu_grad = ExtractActionFromStateAction(qinput_grad, states.Size(-1), mu_s.Size(-1));

            optim_mu.ZeroGrad();
            model.muNetwork.Backward(mu_grad); //gradient ascent
            optim_mu.ClipGradNorm(0.5f);
            optim_mu.Step();

           
        }
        private void UpdateTargetNetworks()
        {
            // We update the target phi and target theta params... softly      

            Tensor[] theta = model.muNetwork.Parameters().Select(x => x.param).ToArray();
            Tensor[] theta_targ = Mu_targ.Parameters().Select(x => x.param).ToArray();

            for (int i = 0; i < theta.Length; i++)
            {
                Tensor.CopyTo((1f - hp.tau) * theta_targ[i] + hp.tau * theta[i], theta_targ[i]);
            }


            Tensor[] phi = model.q1Network.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi_targ = Q_targ.Parameters().Select(x => x.param).ToArray();
            
            for (int i = 0; i < phi.Length; i++)
            {
                Tensor.CopyTo((1f - hp.tau) * phi_targ[i] + hp.tau * phi[i], phi_targ[i]);
            }
        }

        /// <summary>
        /// Note that this must be changed if i plan to allow different input shape than vectorized. (e.g. cnns)
        /// </summary>
        public static Tensor ExtractActionFromStateAction(Tensor stateActionBatch, int state_size, int action_size)
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
        /// <summary>
        /// Concatenates s and a. \\\/
        /// </summary>
        /// <param name="stateBatch"></param>
        /// <param name="actionBatch"></param>
        /// <returns></returns>
        public static Tensor Pairify(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(-1);
            int action_size = actionBatch.Size(-1);
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


