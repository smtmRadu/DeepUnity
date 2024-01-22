using System;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using System.Threading.Tasks;

namespace DeepUnity
{
    // https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f 
    // https://spinningup.openai.com/en/latest/algorithms/sac.html
    // Actually q networks are receiving both squashed and unsquashed inputs

    // So SAC trains Q1 and Q2 without an aditional value function
    public class SACTrainer : DeepUnityTrainer
    {
        // Q target networks
        private static NeuralNetwork Qtarg1;
        private static NeuralNetwork Qtarg2;
        private int new_experiences_collected = 0;

        protected override void Initialize()
        {
            Qtarg1 = model.q1Network.Clone() as NeuralNetwork;
            Qtarg2 = model.q2Network.Clone() as NeuralNetwork;
            if(model.standardDeviation == StandardDeviationType.Fixed)
            {
                ConsoleMessage.Info("Behaviour's standard deviation is Trainable");
                model.standardDeviation = StandardDeviationType.Trainable;
            }
            if(hp.updateAfter <= hp.batchSize)
            {
                ConsoleMessage.Info("'Update After' was set higher than the 'batch size'.");
                hp.updateAfter = hp.batchSize * 5;
            }
            
        }
        protected override void OnBeforeFixedUpdate()
        {
            int decision_freq = parallelAgents[0].DecisionRequester.decisionPeriod;
            new_experiences_collected += parallelAgents.Count;
            if (new_experiences_collected > (hp.updateEvery * decision_freq)) 
            {
                // if the buffer will be full after this collection, let's clear the old values
                if(train_data.Count >= hp.bufferSize)             
                    train_data.frames.RemoveRange(0, train_data.Count / 2); // If buffer is full, remove old half
                
                foreach (var agent_mem in parallelAgents.Select(x => x.Memory))
                {
                    if (agent_mem.Count == 0)
                        continue;

                    train_data.TryAppend(agent_mem, hp.bufferSize);
                    agent_mem.Clear();
                }

                if (train_data.Count >= hp.updateAfter)
                {
                    updateClock = Stopwatch.StartNew();
                    updateIterations++;
                    Train();
                    updateClock.Stop();

                    learningRate = model.muScheduler.CurrentLR;
                    entropy = hp.alpha;
                    currentSteps += new_experiences_collected / decision_freq; 
                    new_experiences_collected = 0;
                }             
            }
        }
       
        
        private void Train()
        {          
            for (int epoch_index = 0; epoch_index < hp.updatesNum; epoch_index++)
            {            
                // Sample a random batch of transitions from the replay buffer
                TimestepTuple[] batch = Utils.Random.Sample(hp.batchSize, train_data.frames);

                // Batchify
                Tensor states = Tensor.Concat(null, batch.Select(x => x.state).ToArray());
                Tensor raw_continuous_actions = Tensor.Concat(null, batch.Select(x => x.action_continuous).ToArray());

                ComputeQTargets(batch);

                Tensor y = Tensor.Concat(0, batch.Select(x => x.q_target).ToArray());

                UpdateQFunctions(states, raw_continuous_actions, y);
                UpdatePolicy(states);
                UpdateTargetNetworks();
            }
        }
        private void ComputeQTargets(TimestepTuple[] batch)
        {           
            Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());
   
            Tensor aTildePrime;     // ã => actions newly sampled from πθ(•|s)
            Tensor piTildePrime;    // (B, CONTINUOUS_ACTIONS)
            model.ContinuousPredict(sPrime, out aTildePrime, out piTildePrime);

            Tensor pair_sPrime_aTildePrime = StateActionPair(sPrime, aTildePrime);
            Tensor[] Qtarg1_sPrime_aTildePrime = Qtarg1.Predict(pair_sPrime_aTildePrime).Split(0, 1); // [](1)
            Tensor[] Qtarg2_sPrime_aTildePrime = Qtarg2.Predict(pair_sPrime_aTildePrime).Split(0, 1); // [](1)

            Tensor logPi = piTildePrime.Prod(-1).Log(); // (B) -- the log prob scalar is the product over the probs vec
            for (int t = 0; t < batch.Length; t++)
            {
                // y(r,s',d) = r + Ɣ(1 - d)[min(Q1t(s',ã'), Q2t(s',ã')) - αlogπθ(ã'|s')]

                float r = batch[t].reward[0];
                float d = batch[t].done[0];

                batch[t].q_target = r + hp.gamma * (1f - d) * (Tensor.Minimum(Qtarg1_sPrime_aTildePrime[t], Qtarg2_sPrime_aTildePrime[t]) - hp.alpha * logPi[t]);
            }
        }
        private void UpdateQFunctions(Tensor states, Tensor continuous_actions, Tensor Q_targets)
        {
            // Update Q functions          
            // ∇φ = (Qφ(s,a) - y(r,s',d)^2
            Tensor stateActionPair = StateActionPair(states, continuous_actions);
            Tensor Q1_s_a = model.q1Network.Forward(stateActionPair);
            Tensor Q2_s_a = model.q2Network.Forward(stateActionPair);

            Loss q1Loss = Loss.MSE(Q1_s_a, Q_targets);
            Loss q2Loss = Loss.MSE(Q2_s_a, Q_targets);
            criticLoss = (q1Loss.Item + q2Loss.Item) / 2;

            model.q1Optimizer.ZeroGrad();
            model.q2Optimizer.ZeroGrad();
            model.q1Network.Backward(q1Loss.Gradient * 0.5f);
            model.q2Network.Backward(q2Loss.Gradient * 0.5f);
            model.q1Optimizer.Step();
            model.q2Optimizer.Step();          
        }
        private void UpdatePolicy(Tensor states)
        {
            // For Appendix C.
            // log πθ(a|s) = log μ(u|s) - log (1 - tanh^2(u))
            // ⊙·····························································⊙
            // ãθ(s) is a sample from πθ(•|s), which is differentiable wrt θ via the reparametrization trick.
            // ãθ(s,ξ) = tanh(μθ(s) + σθ(s) ⊙ ξ), where ξ ~ N(0, 1)

            // ã' => actions newly sampled from πθ(•|s)
            // ãθ(s,ξ) = tanh(μθ(s) + σθ(s) ⊙ ξ), where ξ ~ N(0, 1) => actions newly sampled from πθ(•|s) using the reparametrization trick, which is differentiable wrt θ via the reparametrization trick.

            // u = mu + sigma * ksi

            int batch_size = states.Size(0);
            Tensor aTildeS, u, mu, sigma, ksi;
            model.ContinuousForward(states, out mu, out sigma);
            ksi = Tensor.RandomNormal(sigma.Shape);

            // ãθ(s) = tanh(u)
            Tanh tanh = new Tanh();
            u = mu + sigma * ksi;
            aTildeS = tanh.Forward(u);
            int D = aTildeS.Size(-1);

            // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
            Tensor muDist = Tensor.Pow(
                                MathF.Sqrt(MathF.Pow(2f * MathF.PI, D)) *
                                sigma.Prod(-1, true),
                            -1)
                            * Tensor.Exp(-1f / 2f * Tensor.Sum(ksi.Pow(2f), -1, true)); // mu_dist represents a scalar value, obtained by passing the u value through the dist check appendix C 
            // shape (B, 1) // keep shape like this

            // Check appendinx C
            // log πθ(ãθ(s) |s) = log μ(u|s) - E [log (1 - tanh^2(u))]
            Tensor logPiaTildeS = Tensor.Log(muDist) - Tensor.Sum(Tensor.Log(- new Tanh().Predict(u).Pow(2f) + 1), -1, true); // (B, 1)

            Tensor pair_states_aTildeS = StateActionPair(states, aTildeS);
            Tensor Q1s_aTildeS = model.q1Network.Forward(pair_states_aTildeS);
            Tensor Q2s_aTildeS = model.q2Network.Forward(pair_states_aTildeS);

            // ∇θ min[ Qφ1(s,ãθ(s)), Qφ2(s,ãθ(s))] - αlogπθ(ãθ(s)|s) ]
            Tensor objectiveFunctionJ = Tensor.Minimum(Q1s_aTildeS, Q2s_aTildeS) - hp.alpha * logPiaTildeS;
            actorLoss = objectiveFunctionJ.ToArray().Average();

            // Firstly, we compute the derivative of minQ1Q2 wrt ãθ(s)
            Tensor dminQ1Q2_dQ1 = Tensor.Zeros(batch_size, 1);
            Tensor dminQ1Q2_dQ2 = Tensor.Zeros(batch_size, 1);
            for (int b = 0; b < batch_size; b++)
            {
                if (Q1s_aTildeS[b, 0] <= Q2s_aTildeS[b, 0])
                    dminQ1Q2_dQ1[b, 0] = 1f;
                else
                    dminQ1Q2_dQ2[b, 0] = 1f;
            }
            
            Tensor dminQ1Q2_ds_aTildeS = model.q1Network.Backward(dminQ1Q2_dQ1) + model.q2Network.Backward(dminQ1Q2_dQ2); // Take the gradients from both networks


            // By backwarding the min, we will receive a Tensor of shape (B, S + A), and because we compute the derivative wrt A,
            // we need only A from this tensor, so we extract it separately
            Tensor dminQ1Q2_daTildeS = ExtractActionFromStateAction(dminQ1Q2_ds_aTildeS, states.Size(-1), aTildeS.Size(-1));

            Tensor dminQ1Q2_du = tanh.Backward(dminQ1Q2_daTildeS);
            Tensor dminQ1Q2_dMu = dminQ1Q2_du * 1f;
            Tensor dminQ1Q2_dSigma = dminQ1Q2_du * ksi;

            // Secondly, we compute the derivative of logπθ(ãθ(s)|s) wrt ãθ(s). Note that we are using Gaussian Distribution squashed, check Appendix C in SAC paper
           
            Tensor sech_u = u.Select(x => Utils.Hyperbolics.Sech(x));
            Tensor tanh_u = u.Select(x => Utils.Hyperbolics.Tanh(x));

            // So we know that 
            // MuDist = Multivariate Normal Distribution(u, mu, sigma)
            // dLogMu/dx = CovMat^(-1)(x - mu)

            // log πθ(a|s) = log μ(u|s) - log (1 - tanh^2(u))
            // ∇ log πθ(a|s) = dlog μ(u|s)/du - dlog (1 - tanh^2(u))/du = (u - mu)/sigma^2 - (-2 * tanh(u) * sech^2(u) / (1 - tanh^2(u)))

            Tensor dLogMuDist_du = (u - mu) / sigma.Pow(2f); // https://observablehq.com/@herbps10/distributions-and-their-gradients
            Tensor dLog_1mTanh2u_du = -2f * tanh_u * sech_u.Pow(2f) / (- tanh_u.Pow(2f) + 1f); // Wolfram Alpha?:D

            Tensor dAlphaLogPi_aTildeS_s_du = hp.alpha * (dLogMuDist_du - dLog_1mTanh2u_du);
            Tensor dAlphaLogPiaTildeS_s_dMu = dAlphaLogPi_aTildeS_s_du * 1f;
            Tensor dAlphaLogPiaTildeS_s_dSigma = dAlphaLogPi_aTildeS_s_du * ksi;

            Tensor dJ_dMu = dminQ1Q2_dMu - dAlphaLogPiaTildeS_s_dMu;    
            model.muOptimizer.ZeroGrad();
            model.muNetwork.Backward(-dJ_dMu);
            model.muOptimizer.Step();



            Tensor dJ_dSigma = dminQ1Q2_dSigma - dAlphaLogPiaTildeS_s_dSigma;
            model.sigmaOptimizer.ZeroGrad();
            model.sigmaNetwork.Backward(-dJ_dSigma);
            model.sigmaOptimizer.Step();
                          
        }
        public void UpdateTargetNetworks()
        {
            Tensor[] phi1 = model.q1Network.Parameters().Select(x => x.theta).ToArray();
            Tensor[] phi2 = model.q2Network.Parameters().Select(x => x.theta).ToArray();

            Tensor[] phi_targ1 = Qtarg1.Parameters().Select(x => x.theta).ToArray();
            Tensor[] phi_targ2 = Qtarg2.Parameters().Select(x => x.theta).ToArray();

            // We update the target q functions softly...
            // OpenAI algorithm uses polyak = 0.995, the same thing with using τ = 0.005, inverse the logic duhh. 
            // φtarg,i <- (1 - τ)φtarg,i + τφi     for i = 1,2

            for (int i = 0; i < phi1.Length; i++)
            {
                Tensor.CopyTo((1f - hp.tau) * phi_targ1[i] + hp.tau * phi1[i], phi_targ1[i]);
                Tensor.CopyTo((1f - hp.tau) * phi_targ2[i] + hp.tau * phi2[i], phi_targ2[i]);
            }        
        }



        /// <summary>
        /// including the batch. Note that this must be changed, based on the shape of the input, depending either is using a convolutional layer or recurrent.
        /// </summary>
        private static Tensor ExtractActionFromStateAction(Tensor stateActionBatch, int state_size, int action_size)
        {
            int batch_size = stateActionBatch.Size(0);
            Tensor actions = Tensor.Zeros(batch_size, action_size);
            for (int b = 0; b < batch_size; b++)
            {
                for (int f = 0; f < action_size; f++)
                {
                    actions[b, f] = stateActionBatch[b, state_size + f];
                }
            }
            return actions;
        }
        private static Tensor StateActionPair(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = actionBatch.Size(1);
            Tensor pair = Tensor.Zeros(batch_size, state_size + action_size);
            Parallel.For(0, batch_size, i =>
            {
                for (int s = 0; s < state_size; s++)
                {
                    pair[i, s] = stateBatch[i, s];
                }
                for (int a = 0; a < action_size; a++)
                {
                    pair[i, state_size + a] = actionBatch[i, a];
                }
            });
            return pair;
        }

    }
}