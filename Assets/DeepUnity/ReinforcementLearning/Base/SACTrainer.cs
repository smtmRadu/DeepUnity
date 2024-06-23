using System;
using System.Diagnostics;
using System.Linq;
using UnityEditor;
using System.Threading.Tasks;
using DeepUnity.Models;
using DeepUnity.Activations;
using DeepUnity.Modules;
using DeepUnity.Optimizers;

namespace DeepUnity.ReinforcementLearning
{
    // https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f 
    // https://spinningup.openai.com/en/latest/algorithms/sac.html
    // Actually q networks are receiving both squashed and unsquashed inputs

    // OpenAI SAC trains critic without an aditional value function
    internal sealed class SACTrainer : DeepUnityTrainer, IOffPolicy
    {
        // Q target networks
        private static Sequential Qtarg1;
        private static Sequential Qtarg2;
        private int new_experiences_collected = 0;

        public Optimizer optim_q1 { get; set; }
        public Optimizer optim_q2 { get; set; }
        public Optimizer optim_mu { get; set; }
        public Optimizer optim_sigma { get; set; }



        protected override void Initialize()
        {
            // MEAN module need Tanh for SAC
            if (model.muNetwork.Modules.Last().GetType() != typeof(Tanh))
                model.muNetwork.Modules = model.muNetwork.Modules.Concat(new IModule[] { new Tanh() }).ToArray();

            // Init optimizers
            const float QnetsL2Reg = 0.0F;
            optim_q1 = new Adam(model.q1Network.Parameters(), hp.criticLearningRate, weight_decay: QnetsL2Reg);
            optim_q2 = new Adam(model.q2Network.Parameters(), hp.criticLearningRate, weight_decay: QnetsL2Reg);
            optim_mu = new Adam(model.muNetwork.Parameters(), hp.actorLearningRate);
            optim_sigma = new Adam(model.sigmaNetwork.Parameters(), hp.actorLearningRate);

            // Init schedulers
            optim_q1.Scheduler = new LinearLR(optim_q1, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);
            optim_q2.Scheduler = new LinearLR(optim_q2, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);
            optim_mu.Scheduler = new LinearLR(optim_mu, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);
            optim_sigma.Scheduler = new LinearLR(optim_sigma, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);

            // Init target networks
            Qtarg1 = model.q1Network.Clone() as Sequential;
            Qtarg2 = model.q2Network.Clone() as Sequential;

            // Set devices
            Qtarg1.Device = model.trainingDevice;
            Qtarg2.Device = model.trainingDevice;
            model.q1Network.Device = model.trainingDevice;
            model.q2Network.Device = model.trainingDevice;

            model.muNetwork.Device = model.inferenceDevice;
            model.sigmaNetwork.Device = model.inferenceDevice;

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

                    train_data.TryAppend(agent_mem.frames, hp.replayBufferSize);
                    agent_mem.Clear();
                }

                if (train_data.Count >= hp.updateAfter)
                {
                    model.stochasticity = Stochasticity.TrainebleStandardDeviation;

                    actorLoss = 0;
                    criticLoss = 0;

                    updateBenchmarkClock = Stopwatch.StartNew();
                    Train();
                    updateBenchmarkClock.Stop();

                    updateIterations++;
                    actorLoss /= hp.updatesNum;
                    criticLoss /= hp.updatesNum;
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
                TimestepTuple[] batch = Utils.Random.Sample(hp.minibatchSize, train_data.frames);

                // Batchify
                Tensor states = Tensor.Concat(null, batch.Select(x => x.state).ToArray());
                Tensor raw_continuous_actions = Tensor.Concat(null, batch.Select(x => x.action_continuous).ToArray());
                Tensor q_targets;

                ComputeQTargets(batch, out q_targets);
                UpdateQFunctions(states, raw_continuous_actions, q_targets);
                UpdatePolicy(states);
                UpdateTargetNetworks();


                if (hp.LRSchedule)
                {
                    optim_q1.Scheduler.Step();
                    optim_q2.Scheduler.Step();
                    optim_mu.Scheduler.Step();
                    optim_sigma.Scheduler.Step();
                }
            }
        }
        private void ComputeQTargets(TimestepTuple[] batch, out Tensor y)
        {
            Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());

            Tensor aTildePrime;     // ã => actions newly sampled from πθ(•|s)
            Tensor piTildePrime;    // (B, CONTINUOUS_ACTIONS)
            model.ContinuousEval(sPrime, out aTildePrime, out piTildePrime);

            Tensor pair_sPrime_aTildePrime = StateActionPair(sPrime, aTildePrime);
            Tensor Qtarg1_sPrime_aTildePrime = Qtarg1.Predict(pair_sPrime_aTildePrime); // (B, 1)
            Tensor Qtarg2_sPrime_aTildePrime = Qtarg2.Predict(pair_sPrime_aTildePrime); // (B, 1)

            Tensor logPi = piTildePrime.Prod(-1).Log(); // (B) -- the log prob scalar is the product over the probs vec
            for (int b = 0; b < batch.Length; b++)
            {
                // y(r,s',d) = r + Ɣ(1 - d)[min(Q1t(s',ã'), Q2t(s',ã')) - αlogπθ(ã'|s')]

                float r = batch[b].reward[0];
                float d = batch[b].done[0];
                float Qt1_sa = Qtarg1_sPrime_aTildePrime[b, 0];
                float Qt2_sa = Qtarg2_sPrime_aTildePrime[b, 0];

                float _y = r + hp.gamma * (1f - d) * (MathF.Min(Qt1_sa, Qt2_sa) - hp.alpha * logPi[b]);

                batch[b].q_target = Tensor.Constant(_y);
            }

            y = Tensor.Concat(null, batch.Select(x => x.q_target).ToArray());
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

            optim_q1.ZeroGrad();
            optim_q2.ZeroGrad();
            model.q1Network.Backward(q1Loss.Grad * 0.5f);
            model.q2Network.Backward(q2Loss.Grad * 0.5f);
            optim_q1.Step();
            optim_q2.Step();
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
            u = mu + sigma * ksi;
            aTildeS = u.Tanh();
            int D = aTildeS.Size(-1);

            // https://en.wikipedia.org/wiki/Multivariate_normal_distribution
            Tensor muDist = Tensor.Pow(
                                MathF.Sqrt(MathF.Pow(2f * MathF.PI, D)) *
                                sigma.Prod(-1, true), -1)
                            * Tensor.Exp(-1f / 2f * Tensor.Sum(ksi.Pow(2f), -1, true)); // mu_dist represents a scalar value, obtained by passing the u value through the dist check appendix C 
            // shape (B, 1) // keep shape like this

            // Check appendinx C
            // log πθ(ãθ(s) |s) = log μ(u|s) - E [log (1 - tanh^2(u))]
            Tensor logPiaTildeS = Tensor.Log(muDist) - Tensor.Sum(Tensor.Log(1f - u.Tanh().Square()), -1, true); // (B, 1)

            Tensor pair_states_aTildeS = StateActionPair(states, aTildeS);
            Tensor Q1s_aTildeS = model.q1Network.Forward(pair_states_aTildeS);
            Tensor Q2s_aTildeS = model.q2Network.Forward(pair_states_aTildeS);

            // ∇θ min[ Qφ1(s,ãθ(s)), Qφ2(s,ãθ(s))] - αlogπθ(ãθ(s)|s) ]
            Tensor objectiveFunctionJ = Tensor.Minimum(Q1s_aTildeS, Q2s_aTildeS) - hp.alpha * logPiaTildeS;
            actorLoss = objectiveFunctionJ.Average();

            // Start differentiating the objective loss
      
            Tensor tanh_u = aTildeS; // right?:D

            // Firstly, we compute the derivative of minQ1Q2 wrt ãθ(s)
            Tensor dminQ1Q2_dQ1 = Q1s_aTildeS <= Q2s_aTildeS;
            Tensor dminQ1Q2_dQ2 = Tensor.LogicalNot(dminQ1Q2_dQ1);

            Tensor dminQ1Q2_ds_aTildeS = model.q1Network.Backward(dminQ1Q2_dQ1) + model.q2Network.Backward(dminQ1Q2_dQ2); // Take the gradients from both networks

            // By backwarding the min, we will receive a Tensor of shape (B, S + A), and because we compute the derivative wrt A,
            // we need only A from this tensor, so we extract it separately
            Tensor dminQ1Q2_daTildeS = ExtractActionFromStateAction(dminQ1Q2_ds_aTildeS, states.Size(-1), aTildeS.Size(-1));

            Tensor dminQ1Q2_du = dminQ1Q2_daTildeS * (1f - tanh_u.Square());
            Tensor dminQ1Q2_dMu = dminQ1Q2_du;// * 1f;
            Tensor dminQ1Q2_dSigma = dminQ1Q2_du * ksi;



            // Secondly, we compute the derivative of logπθ(ãθ(s)|s) wrt ãθ(s). Note that we are using Gaussian Distribution squashed, check Appendix C in SAC paper

            // So we know that 
            // MuDist = Multivariate Normal Distribution(u, mu, sigma)
            // dLogMu/dx = CovarianceMatrix^(-1)(x - mu)

            // log πθ(a|s) = log μ(u|s) - log (1 - tanh^2(u))
            // ∇ log πθ(a|s) = dlog μ(u|s)/du - dlog (1 - tanh^2(u))/du = (u - mu)/sigma^2 - (-2 * tanh(u) * sech^2(u) / (1 - tanh^2(u)))

            Tensor dLogMuDist_du = (u - mu) / sigma.Pow(2f); // https://observablehq.com/@herbps10/distributions-and-their-gradients
            Tensor dLog_1mTanh2u_du = -2f * tanh_u * u.Sech().Pow(2f) / (1f - tanh_u.Pow(2f)); // Wolfram Alpha? maybe..

            Tensor dAlphaLogPi_aTildeS_s_du = hp.alpha * (dLogMuDist_du - dLog_1mTanh2u_du);
            Tensor dAlphaLogPiaTildeS_s_dMu = dAlphaLogPi_aTildeS_s_du; // * 1f;
            Tensor dAlphaLogPiaTildeS_s_dSigma = dAlphaLogPi_aTildeS_s_du * ksi;

            Tensor dJ_dMu = dminQ1Q2_dMu - dAlphaLogPiaTildeS_s_dMu;
            optim_mu.ZeroGrad();
            model.muNetwork.Backward(-dJ_dMu);
            optim_mu.Step();

            Tensor dJ_dSigma = dminQ1Q2_dSigma - dAlphaLogPiaTildeS_s_dSigma;
            optim_sigma.ZeroGrad();
            model.sigmaNetwork.Backward(-dJ_dSigma);
            optim_sigma.Step();

        }
        public void UpdateTargetNetworks()
        {
            Tensor[] phi1 = model.q1Network.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi2 = model.q2Network.Parameters().Select(x => x.param).ToArray();

            Tensor[] phi_targ1 = Qtarg1.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi_targ2 = Qtarg2.Parameters().Select(x => x.param).ToArray();

            // We update the target q functions softly...
            // OpenAI algorithm uses polyak = 0.995, the same thing with using τ = 0.005
            // φtarg,i <- (1 - τ)φtarg,i + τφi     for i = 1,2

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
        private static Tensor StateActionPair(Tensor stateBatch, Tensor actionBatch)
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

#if UNITY_EDITOR
    [CustomEditor(typeof(SACTrainer), true), CanEditMultipleObjects]
    sealed class CustomSACTrainerEditor : Editor
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