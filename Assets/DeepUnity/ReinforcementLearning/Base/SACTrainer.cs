using System;
using System.Diagnostics;
using System.Linq;

namespace DeepUnity
{
    // https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f 
    // https://spinningup.openai.com/en/latest/algorithms/sac.html
  

    // So SAC trains Q1 and Q2 without an aditional value function
    public class SACTrainer : DeepUnityTrainer
    {
        private static NeuralNetwork Qtarg1;
        private static NeuralNetwork Qtarg2;

        protected override void Initialize()
        {
            Qtarg1 = model.q1Network.Clone() as NeuralNetwork;
            Qtarg2 = model.q2Network.Clone() as NeuralNetwork;
        }
        protected override void FixedUpdate()
        {
            
            if (currentSteps > hp.updateAfter && // Start to update the policy after some experiences were collected, note to force the actions to be random in this part..
                (currentSteps % hp.updateEvery == 0 || // Update every x steps
                train_data.Count == hp.bufferSize)) // If the buffer is full, do not collect experiences anymore, just train... Well actually i would rather remove the old experiences instead...
            {
                foreach (var agent_mem in parallelAgents.Select(x => x.Memory))
                {
                    if (agent_mem.Count == 0)
                        continue;

                    train_data.TryAppend(agent_mem, hp.bufferSize);
                    if (hp.debug) Utils.DebugInFile(agent_mem.ToString());
                    agent_mem.Clear();
                }
                Stopwatch clock = Stopwatch.StartNew();
                Train();
                clock.Stop();
            }

            base.FixedUpdate(); 
        }
        private void Train()
        {          
            for (int epoch_index = 0; epoch_index < hp.updatesNum; epoch_index++)
            {
                // Sample a random batch of transitions from the replay buffer
                TimestepTuple[] batch = Utils.Random.Sample(hp.batchSize, train_data.frames);

                // Batchify
                Tensor states = Tensor.Cat(null, batch.Select(x => x.state).ToArray());
                Tensor actions = Tensor.Cat(null, batch.Select(x => x.action_continuous).ToArray());
                Tensor y = Tensor.Cat(null, batch.Select(x => x.q_target).ToArray());

                ComputeQTargets(batch, hp.gamma);
                UpdateQFunctions(states, actions, y);
                UpdatePolicy(states);
                UpdateTargetNetworks();
            }
        }

       
        private void ComputeQTargets(TimestepTuple[] batch, in float GAMMA)
        {
            Tensor nextStates = Tensor.Cat(null, batch.Select(x => x.nextState).ToArray());

            Tensor new_actions;
            Tensor new_probs;
            model.ContinuousPredict(nextStates, out new_actions, out new_probs);

            Tensor[] Qtarg1 = QTarg1Predict(nextStates, new_actions).Split(0, 1); // [](CONT_ACT)
            Tensor[] Qtarg2 = QTarg2Predict(nextStates, new_actions).Split(0, 1); // [](CONT_ACT)

            Tensor[] pi = Tensor.Split(new_probs, 0, 1); // We split to get each element from the batch
            for (int t = 0; t < batch.Length; t++) // note that is random
            {
                // y(r,s',d) = r + Ɣ(1 - d)[ min(Q1t(s',ã'), Q2t(s',ã')) - απθ(ã'|s') ]

                // Extract values from any timestep i
                float r = batch[t].reward[0];
                float d = batch[t].done[0];

                Tensor y = r + GAMMA * (1 - d) * (Tensor.Minimum(Qtarg1[t], Qtarg2[t]) - hp.alpha * Tensor.Log(pi[t])); // a.k.a Qhat, shape: (CONT_ACT)
                batch[t].q_target = y;
            }
        }
        private void UpdateQFunctions(Tensor states, Tensor actions, Tensor y)
        {
            // Update Q functions          
            // ∇φ = (Qφ(s,a) - y(r,s',d)^2
            Tensor Q1_s_a = model.Q1Forward(states, actions);
            Tensor Q2_s_a = model.Q2Forward(states, actions);

            Loss q1Loss = Loss.MSE(Q1_s_a, y);
            Loss q2Loss = Loss.MSE(Q2_s_a, y);

            model.q1Optimizer.ZeroGrad();
            model.q2Optimizer.ZeroGrad();
            model.q1Network.Backward(q1Loss.Derivative * 0.5f);
            model.q2Network.Backward(q2Loss.Derivative * 0.5f);
            model.q1Optimizer.ClipGradNorm(hp.gradClipNorm);
            model.q2Optimizer.ClipGradNorm(hp.gradClipNorm);
            model.q1Optimizer.Step();
            model.q2Optimizer.Step();
        }
        private void UpdatePolicy(Tensor states)
        {
            Tensor tilde_a_s, mu, sigma;
            model.ContinuousReparametrizedForward(states, out tilde_a_s, out mu, out sigma);
            Tensor logPi_tildeas_S = Tensor.LogProbability(tilde_a_s, mu, sigma);

            // ⊙·····························································⊙
            // ãθ(s) is a sample from πθ(•|s), which is differentiable wrt θ via the reparametrization trick.
            // ãθ(s,ξ) = tanh(μθ(s) + σθ(s) ⊙ ξ), where ξ ~ N(0, 1)
            Tanh tanh = new Tanh();
            tilde_a_s = tanh.Forward(tilde_a_s);

            Tensor Q1_s_tildeas = model.Q1Forward(states, tilde_a_s);
            Tensor Q2_s_tildeas = model.Q2Forward(states, tilde_a_s);
            Tensor minQ1Q2 = Tensor.Minimum(Q1_s_tildeas, Q2_s_tildeas);

            // ∇θ = min[ Qφ1(s,ãθ(s)), Qφ2(s,ãθ(s))] - αlogπθ(ãθ(s)|s) ]
            Tensor piLoss = Tensor.Minimum(Q1_s_tildeas, Q2_s_tildeas) - logPi_tildeas_S;


            // Consider we do gradient *ascent*, so we do negative of the loss function

            // We cannot compute the derivative of the negative L with respect to the 
            // heads outputs, so we need to expand the differential equation...
            // ∂-Lθ / ∂μθ = ?
            // ∂-Lθ / ∂σθ = ?

            // ∂L / ∂πθ(a|s) = 1/πθ(a|s);
            Tensor dmL_dPi = logPi_tildeas_S.Pow(-1f);

            // [ ∂πθ(a|s) / ∂μ != πθ(a|s) * (x - μ) / σ^2 ] because the normal sampling is using reparametrization trick
            // [ ∂πθ(a|s) / ∂σ != πθ(a|s) * ((x - μ)^2 - σ^2) / σ^3 ] because the normal sampling is using reparametrization trick

            // ∂πθ(a|s) / ∂μ = ?
            Tensor dPi_dMu = null;
            // ∂πθ(a|s) / ∂σ = ?
            Tensor dPi_dSigma = null;


            // ∂-L / ∂μ = (∂-L / ∂πθ(a|s)) * (∂πθ(a|s) / ∂μ)
            Tensor dmL_dMu = dmL_dPi * dPi_dMu;

            // ∂-L / ∂σ = (∂-L / ∂πθ(a|s)) * (∂πθ(a|s) / ∂σ)
            Tensor dmL_dSigma = dmL_dPi * dPi_dSigma;
          

            model.muOptimizer.ZeroGrad();
            model.muNetwork.Backward(dmL_dMu);
            model.muOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.muOptimizer.Step();


            model.sigmaOptimizer.ZeroGrad();
            model.sigmaNetwork.Backward(dmL_dSigma);
            model.sigmaOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.sigmaOptimizer.Step();
            
           
        }
        public void UpdateTargetNetworks()
        {
            Tensor[] phi1 = model.q1Network.Parameters();
            Tensor[] phi2 = model.q2Network.Parameters();

            Tensor[] phi_targ1 = Qtarg1.Parameters();
            Tensor[] phi_targ2 = Qtarg2.Parameters();


            // We update the target q functions softly...
            // φtarg,i <- τ φtarg,i + (1 - τ)φi     for i = 1,2

            for (int i = 0; i < phi1.Length; i++)
            {
                phi_targ1[i].AssignAs(hp.tau * phi_targ1[i] + (1f - hp.tau) * phi1[i]);
                phi_targ2[i].AssignAs(hp.tau * phi_targ2[i] + (1f - hp.tau) * phi2[i]);
            }        
        }



        private Tensor QTarg1Predict(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = stateBatch.Size(1);
            Tensor input = Tensor.Zeros(batch_size, state_size + action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < state_size; f++)
                {
                    input[i, f] = stateBatch[i, f];
                }
                for (int f = state_size; f < state_size + action_size; f++)
                {
                    input[i, f] = actionBatch[i, f];
                }
            }
            return Qtarg1.Predict(input);
        }
        private Tensor QTarg2Predict(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = stateBatch.Size(1);
            Tensor input = Tensor.Zeros(batch_size, state_size + action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < state_size; f++)
                {
                    input[i, f] = stateBatch[i, f];
                }
                for (int f = state_size; f < state_size + action_size; f++)
                {
                    input[i, f] = actionBatch[i, f];
                }
            }
            return Qtarg2.Predict(input);
        }

    }
}



/* Recycle bin
 * private void UpdateValueNetwork(Tensor states)
        {
            // Update Value function Jv(φ) = 1/2 * ((Vφ(s) - E[Q(s,ã) - logπθ(ã|s)])^2
            Tensor actions;
            Tensor probs;
            model.ContinuousPredict(states, out actions, out probs);

            Tensor Q1_sa = model.Q1Forward(states, actions);
            Tensor Q2_sa = model.Q2Forward(states, actions);
            Tensor critic_values = Tensor.Minimum(Q1_sa, Q2_sa);

            Tensor V_targ = critic_values - probs.Log(); // Remember to retain the graph for Q networks, critic_values also backwards
            Tensor V_s = model.vNetwork.Forward(states);
            Loss ValueLoss = Loss.MSE(V_s, V_targ);

            model.vOptimizer.ZeroGrad();
            model.vNetwork.Backward(ValueLoss.Derivative * 0.5f);
            model.vOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.vOptimizer.Step();
        }
*/