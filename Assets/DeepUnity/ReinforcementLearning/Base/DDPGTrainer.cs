using DeepUnity.Models;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;

namespace DeepUnity.ReinforcementLearning

{
    internal class DDPGTrainer : DeepUnityTrainer
    {
        private static Sequential PhiTarg;
        private static Sequential ThetaTarg;

        private int new_experiences_collected = 0;

        protected override void Initialize()
        {
            PhiTarg = model.q1Network.Clone() as Sequential;
            ThetaTarg = model.muNetwork.Clone() as Sequential;

            model.muNetwork.Device = model.inferenceDevice;
            model.q1Network.Device = model.trainingDevice;

            PhiTarg.Device = model.trainingDevice;
            ThetaTarg.Device = model.trainingDevice;
            

            if (model.stochasticity != Stochasticity.ActiveNoise)
            {
                ConsoleMessage.Info("Behaviour's stochasticity is now given by active noise.");
                model.stochasticity = Stochasticity.ActiveNoise;
            }
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
                    actorLoss = 0;
                    criticLoss = 0;

                    updateClock = Stopwatch.StartNew();        
                    Train();
                    updateClock.Stop();

                    updateIterations++;
                    actorLoss /= hp.updatesNum;
                    criticLoss /= hp.updatesNum;
                    learningRate = model.muScheduler.CurrentLR;
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
                Tensor s = Tensor.Concat(null, batch.Select(x => x.state).ToArray());
                Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());
                Tensor a = Tensor.Concat(null, batch.Select(x => x.action_continuous).ToArray());


                Tensor y_r_sPrime_d;
                ComputeQTargets(batch, sPrime, out y_r_sPrime_d);
                UpdateQFunctions(s, a, y_r_sPrime_d);
                UpdatePolicy(s);
                UpdateTargetNetworks();

                if (hp.LRSchedule)
                {
                    model.q1Scheduler.Step();
                    model.muScheduler.Step();
                }
            }

            model.muNetwork.Device = model.inferenceDevice;
        }

        private void ComputeQTargets(TimestepTuple[] batch, Tensor sPrime, out Tensor y)
        {
            Tensor QPhiTarg_sPrime_ThetaTarg_sPrime = PhiTarg.Predict(Pairify(sPrime, ThetaTarg.Predict(sPrime))); // (B, 1)

            for (int b = 0; b < batch.Length; b++)
            {
                // y(r,s',d) = r + Ɣ(1 - d) * Q(s',ã')

                float r = batch[b].reward[0];
                float d = batch[b].done[0];
                float Qt_sa = QPhiTarg_sPrime_ThetaTarg_sPrime[b, 0];

                float y_r_sPrime_d = r + hp.gamma * (1f - d) * Qt_sa;

                batch[b].q_target = Tensor.Constant(y_r_sPrime_d);
            }

            y = Tensor.Concat(null, batch.Select(x => x.q_target).ToArray());
        }
        private void UpdateQFunctions(Tensor states, Tensor actions, Tensor Q_targets)
        {
            // Update Q functions          
            // ∇φ = (Qφ(s,a) - y(r,s',d))^2
            Tensor Q_s_a = model.q1Network.Forward(Pairify(states, actions));
            Loss qLoss = Loss.MSE(Q_s_a, Q_targets);
            float lossItem = qLoss.Item;

            if (float.IsNaN(lossItem))
                return;
            else
                criticLoss += lossItem;
           
            model.q1Optimizer.ZeroGrad();
            model.q1Network.Backward(qLoss.Gradient * 0.5f);
            model.q1Optimizer.Step();
        }
        private void UpdatePolicy(Tensor states)
        {
            // ObjectiveLoss = 1/|B| Σ Q(s, mu(s))
            Tensor mu_s = model.muNetwork.Forward(states);
            Tensor q_s_mu_s = model.q1Network.Forward(Pairify(states, mu_s));
            actorLoss += q_s_mu_s.Average();

            Tensor objectiveLossGrad = Tensor.Ones(q_s_mu_s.Shape); //  (grad of Mean())
            Tensor s_mu_s_grad = model.q1Network.Backward(objectiveLossGrad); //phi params are fixed
            Tensor mu_s_grad = ExtractActionFromStateAction(s_mu_s_grad, states.Size(-1), mu_s.Size(-1));

            model.muOptimizer.ZeroGrad();
            model.muNetwork.Backward(-mu_s_grad); //gradient ascent
            model.muOptimizer.Step();
        }
        public void UpdateTargetNetworks()
        {
            Tensor[] theta = model.muNetwork.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi = model.q1Network.Parameters().Select(x => x.param).ToArray();

            Tensor[] theta_targ = ThetaTarg.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi_targ = PhiTarg.Parameters().Select(x => x.param).ToArray();

            // We update the target q functions and target theta params...

            for (int i = 0; i < theta.Length; i++)
            {
                Tensor.CopyTo((1f - hp.tau) * theta_targ[i] + hp.tau * theta[i], theta_targ[i]);
            }

            for (int i = 0; i < phi.Length; i++)
            {
                Tensor.CopyTo((1f - hp.tau) * phi_targ[i] + hp.tau * phi[i], phi_targ[i]);
            }
        }




        /// <summary>
        /// Note that this must be changed if i plan to allow different input shape than vectorized. (e.g. cnns)
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


