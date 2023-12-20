using System;
using System.Diagnostics;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;

namespace DeepUnity
{
    // https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f 
    // https://spinningup.openai.com/en/latest/algorithms/sac.html


    // So SAC trains Q1 and Q2 without an aditional value function
    public class SACTrainer : DeepUnityTrainer
    {
        // Q target networks
        private static NeuralNetwork Qtarg1;
        private static NeuralNetwork Qtarg2;
        private Tensor QTarg1Predict(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = actionBatch.Size(1);
            Tensor input = Tensor.Zeros(batch_size, state_size + action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < state_size; f++)
                {
                    input[i, f] = stateBatch[i, f];
                }
                for (int f = 0; f < action_size; f++)
                {
                    input[i, state_size + f] = actionBatch[i, f];
                }
            }
            return Qtarg1.Predict(input);
        }
        private Tensor QTarg2Predict(Tensor stateBatch, Tensor actionBatch)
        {
            int batch_size = stateBatch.Size(0);
            int state_size = stateBatch.Size(1);
            int action_size = actionBatch.Size(1);
            Tensor input = Tensor.Zeros(batch_size, state_size + action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < state_size; f++)
                {
                    input[i, f] = stateBatch[i, f];
                }
                for (int f = 0; f < action_size; f++)
                {
                    input[i, state_size + f] = actionBatch[i, f];
                }
            }
            return Qtarg2.Predict(input);
        }





        protected override void Initialize()
        {
            Qtarg1 = model.q1Network.Clone() as NeuralNetwork;
            Qtarg2 = model.q2Network.Clone() as NeuralNetwork;

            if(hp.batchSize > hp.updateEvery)
            {
                ConsoleMessage.Warning("Batch Size must be less or equal to Update Every");
                EditorApplication.isPlaying = false;
            }
        }
        protected override void FixedUpdate()
        {
            int collected_steps = 0;
            foreach (var ag in parallelAgents)
            {
                collected_steps += ag.Memory.Count;
            }

            if (collected_steps > 0 && collected_steps % hp.updateEvery == 0) 
            {
                // if the buffer will be full after this collection, let's clear the old values
                if(train_data.Count >= hp.bufferSize - collected_steps)             
                    train_data.frames.RemoveRange(0, train_data.Count / 2);
                
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

                currentSteps += collected_steps;
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
                Tensor states = Tensor.Concat(null, batch.Select(x => x.state).ToArray());
                Tensor actions = Tensor.Concat(null, batch.Select(x => x.action_continuous).ToArray());
                
 
                UpdateQFunctions(states, actions, ComputeQTargets(batch, hp.gamma));
                UpdatePolicy(states);
                UpdateTargetNetworks();
            }
        }

        
        // Note ... consider that when computing Q targets, the newly sampled actions are sampled from that second distrbutions,
        // and we also obtain the probabilities out from that...........
        private Tensor ComputeQTargets(TimestepTuple[] batch, in float GAMMA)
        {
            
            Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());

       
            Tensor aTildePrime;     // ã => actions newly sampled from πθ(•|s)
            Tensor piTildePrime;    // (B, CONTINUOUS_ACTIONS)
            model.ContinuousPredict(sPrime, out aTildePrime, out piTildePrime);

            Tensor[] Qtarg1_sPrime_aTildePrime = QTarg1Predict(sPrime, aTildePrime).Split(0, 1); // [](CONT_ACT)
            Tensor[] Qtarg2_sPrime_aTildePrime = QTarg2Predict(sPrime, aTildePrime).Split(0, 1); // [](CONT_ACT)

            // We need to actually take a look here, it is posible that the probability is mean over the action space,
            // we need to check the Appendix C on how the probabilities are taken from the new distribution. 

            Tensor[] pi = Tensor.Split(piTildePrime, 0, 1); // We split to get each element from the batch
            float[] mean_pi = pi.Select(x => x.Mean(-1)[0]).ToArray();
            for (int t = 0; t < batch.Length; t++) // note that is random
            {
                // y(r,s',d) = r + Ɣ(1 - d)[ min(Q1t(s',ã'), Q2t(s',ã')) - απθ(ã'|s') ]

                float r = batch[t].reward[0];
                float d = batch[t].done[0];

                Tensor y = r + GAMMA * (1f - d) * (Tensor.Minimum(Qtarg1_sPrime_aTildePrime[t], Qtarg2_sPrime_aTildePrime[t]) - hp.alpha *  MathF.Log(mean_pi[t]));
                batch[t].q_target = y;
            }
            return Tensor.Concat(0, batch.Select(x => x.q_target).ToArray());
        }
        private void UpdateQFunctions(Tensor states, Tensor actions, Tensor y)
        {
            // Update Q functions          
            // ∇φ = (Qφ(s,a) - y(r,s',d)^2
            Tensor Q1_s_a = model.Q1Forward(states, actions);
            Tensor Q2_s_a = model.Q2Forward(states, actions);

            Loss q1Loss = Loss.MSE(Q1_s_a, y);
            Loss q2Loss = Loss.MSE(Q2_s_a, y);
            track.valueLoss.Append((q1Loss.Item + q2Loss.Item) / 2);

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
            // For Appendix C.
            // log πθ(a|s) = log μ(u|s) - log (1 - tanh^2(u))
            // ⊙·····························································⊙
            // ãθ(s) is a sample from πθ(•|s), which is differentiable wrt θ via the reparametrization trick.
            // ãθ(s,ξ) = tanh(μθ(s) + σθ(s) ⊙ ξ), where ξ ~ N(0, 1)

            // ã' => actions newly sampled from πθ(•|s)
            // ãθ(s,ξ) = tanh(μθ(s) + σθ(s) ⊙ ξ), where ξ ~ N(0, 1) => actions newly sampled from πθ(•|s) using the reparametrization trick, which is differentiable wrt θ via the reparametrization trick.

            // u = mu + sigma * ksi

            // Question 1: so prob is of shape (B, CONT_ACT), when we compute the objective loss do we sum or mean the probabilities over the action dimension?
            // Question 2: for inference, can we sample with box muller to obtain ã ? (not ãθ(s))
            // Answer : So conv diverges only because the bad initialization :D
            // Question 3: What to do after GRU, LSTM, Attention?
            // Question 4: what about regional entropy decay?


            int batch_size = states.Size(0);
            Tensor aTildeS, u, mu, sigma, ksi;
            model.ContinuousForward(states, out mu, out sigma);
            ksi = Tensor.RandomNormal(sigma.Shape);

            // ãθ(s) = tanh(u)
            Tanh tanh = new Tanh();
            u = mu + sigma * ksi;
            aTildeS = tanh.Forward(u);
            int D = mu.Size(-1);

            Tensor muDist = (MathF.Sqrt(MathF.Pow(2f * MathF.PI, D)) * sigma.Prod(-1, true)).Pow(-1)
                            * Tensor.Exp( -0.5f * ksi.Pow(2).Sum(-1, true)); // mu_dist represents a scalar value, obtained by passing the u value through the dist check appendix C (B, 1)

            // Check appendinx C
            // log πθ(ãθ(s) |s) = log μ(u|s) - E [log (1 - tanh^2(u))]
            Tensor logPiaTildeS = Tensor.Log(muDist) - Tensor.Sum(Tensor.Log(- new Tanh().Predict(u).Pow(2f) + 1 + Utils.EPSILON), -1, true);
                                 // returns NaN     

            Tensor Q1s_aTildeS = model.Q1Forward(states, aTildeS);
            Tensor Q2s_aTildeS = model.Q2Forward(states, aTildeS);

            // ∇θ min[ Qφ1(s,ãθ(s)), Qφ2(s,ãθ(s))] - αlogπθ(ãθ(s)|s) ]
            Tensor objectiveFunctionJ = Tensor.Minimum(Q1s_aTildeS, Q2s_aTildeS) - hp.alpha * logPiaTildeS.Sum(-1, true);
            track.policyLoss.Append(objectiveFunctionJ.Mean(0).Mean(0)[0]);
     
            // Firstly, we compute the derivative of minQ1Q2 wrt ãθ(s)
          
            Tensor dminQ1Q2_dQ1 = Tensor.Zeros(batch_size, 1);
            Tensor dminQ1Q2_dQ2 = Tensor.Zeros(batch_size, 1);
            for (int i = 0; i < batch_size; i++)
            {
                if (Q1s_aTildeS[i, 0] <= Q2s_aTildeS[i, 0])
                {
                    dminQ1Q2_dQ1[i, 0] = 1f;
                    dminQ1Q2_dQ2[i, 0] = 0f;
                }
                else
                {
                    dminQ1Q2_dQ1[i, 0] = 0f;
                    dminQ1Q2_dQ2[i, 0] = 1f;
                }
            }

            Tensor dminQ1Q2_dQinput = model.q1Network.Backward(dminQ1Q2_dQ1) + model.q2Network.Backward(dminQ1Q2_dQ2); // Take the gradients from both networks
            // By backwarding the min, we will receive a Tensor of shape (B, S + A), and because we compute the derivative wrt A,
            // we need only A from this tensor, so we extract it separately
            Tensor dminQ1Q2_daTildeS = ExtractActionFromStateAction(dminQ1Q2_dQinput, model.observationSize, model.continuousDim);

            Tensor dminQ1Q2_du = tanh.Backward(dminQ1Q2_daTildeS);

            Tensor dminQ1Q2_dMu = dminQ1Q2_du * 1f;
            Tensor dminQ1Q2_dSigma = dminQ1Q2_du * ksi;

            // Secondly, we compute the derivative of logπθ(ãθ(s)|s) wrt ãθ(s). Note that we are using another distribution, check Appendix C in SAC paper
           
            Tensor sech_u = u.Select(x => Utils.Hyperbolics.Sech(x));
            Tensor tanh_u = u.Select(x => Utils.Hyperbolics.Tanh(x));

            // So we know that 
            // log πθ(a|s) = log μ(u|s) - log (1 - tanh^2(u))
            // ∇ log πθ(a|s) = dlog μ(u|s)/du - dlog (1 - tanh^2(u))/du = (u - mu)/sigma^2 - (-2 * tanh(u) * sech^2(u) / (1 - tanh^2(u)))
            Tensor dLogMuDist_du = (u - mu) / sigma.Pow(2f); // IT IS POSSIBLE WE MISSED SOMETHING HERE... we'll see. What? https://stackoverflow.com/questions/13299642/how-to-calculate-derivative-of-multivariate-normal-probability-density-function
            Tensor dLog_1mTanh2u_du = -2f * tanh_u * sech_u.Pow(2f) / (- tanh_u.Pow(2f) + 1);

            Tensor dAlphaLogPi_aTildeS_s_du = hp.alpha * (dLogMuDist_du - dLog_1mTanh2u_du);


            Tensor dAlphaLogPiaTildeS_s_dMu = dAlphaLogPi_aTildeS_s_du * 1;
            Tensor dAlphaLogPiaTildeS_s_dSigma = dAlphaLogPi_aTildeS_s_du * ksi;

            Tensor dJ_dMu = dminQ1Q2_dMu - dAlphaLogPiaTildeS_s_dMu;


            print("mu" + mu);
            print("sigma" + sigma);
            print("ksi" + ksi);
            print("u" + u);
            print("aTildeS" + aTildeS);
            print("mu_distribution : " + muDist);
            print("Log pi_aTildeS : " + logPiaTildeS); //some a bit large but oke...
            print("dminQ1Q2_daTildeS" + dminQ1Q2_daTildeS); //ok
            print("dminQ1Q2_du" + dminQ1Q2_du); //ok
            print("dAlphaLogPi_aTildeS_s_du" + dAlphaLogPi_aTildeS_s_du);
            print("dJ_dMu" + dJ_dMu); // This guy got some infinities over here



            model.muOptimizer.ZeroGrad();
            model.muNetwork.Backward(-dJ_dMu); // is negative because we do gradient ascent
            model.muOptimizer.ClipGradNorm(hp.gradClipNorm);
            model.muOptimizer.Step();

            if(model.standardDeviation == StandardDeviationType.Trainable)
            {
                Tensor dJ_dSigma = dminQ1Q2_dSigma - dAlphaLogPiaTildeS_s_dSigma;

                model.sigmaOptimizer.ZeroGrad();
                model.sigmaNetwork.Backward(-dJ_dSigma);
                model.sigmaOptimizer.ClipGradNorm(hp.gradClipNorm);
                model.sigmaOptimizer.Step();
            }              
        }
        public void UpdateTargetNetworks()
        {
            Tensor[] phi1 = model.q1Network.Parameters().Select(x => x.theta).ToArray();
            Tensor[] phi2 = model.q2Network.Parameters().Select(x => x.theta).ToArray();

            Tensor[] phi_targ1 = Qtarg1.Parameters().Select(x => x.theta).ToArray();
            Tensor[] phi_targ2 = Qtarg2.Parameters().Select(x => x.theta).ToArray();


            // We update the target q functions softly...
            // OpenAI algorithm uses polyak = 0.995, the same thing with using τ = 0.005, but we inverse the logic. 
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
        /// <param name="stateActionBatch"></param>
        /// <param name="state_size"></param>
        /// <param name="action_size"></param>
        /// <returns></returns>
        private static Tensor ExtractActionFromStateAction(Tensor stateActionBatch, int state_size, int action_size)
        {
            int batch_size = stateActionBatch.Size(0);
            Tensor actions = Tensor.Zeros(batch_size, action_size);
            for (int i = 0; i < batch_size; i++)
            {
                for (int f = 0; f < action_size; f++)
                {
                    actions[i, f] = stateActionBatch[i, state_size + f];
                }
            }
            return actions;
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