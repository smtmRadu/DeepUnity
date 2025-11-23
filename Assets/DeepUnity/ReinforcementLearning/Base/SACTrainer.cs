using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Optimizers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Threading.Tasks;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    // https://medium.com/intro-to-artificial-intelligence/soft-actor-critic-reinforcement-learning-algorithm-1934a2c3087f 
    // https://spinningup.openai.com/en/latest/algorithms/sac.html
    // Actually q networks are receiving both squashed and unsquashed inputs

    // OpenAI SAC trains critic without an aditional value function
    internal sealed class SACTrainer : DeepUnityTrainer, IOffPolicy
    {
        // Q target networks
        private static Sequential q1TargNetwork;
        private static Sequential q2TargNetwork;
        private int new_experiences_collected = 0;

        public Optimizer optim_q1q2 { get; set; }
        public Optimizer optim_mu { get; set; }
        public Optimizer optim_sigma { get; set; }



        protected override void Initialize(string[] optimizer_states)
        {
            if (model.IsUsingContinuousActions && model.muNetwork.Modules.Last().GetType() == typeof(Tanh))
                model.muNetwork.Modules = model.muNetwork.Modules.Take(model.muNetwork.Modules.Length - 1).ToArray();


            // Init optimizers
            const float QnetsL2Reg = 0.0F;
            if (optimizer_states == null)
            {
                optim_q1q2 = new AdamW(model.q1Network.Parameters().Concat(model.q2Network.Parameters()).ToArray(), lr: hp.criticLearningRate, weight_decay: -QnetsL2Reg);
                optim_mu = new AdamW(model.muNetwork.Parameters(), lr:hp.actorLearningRate);
                optim_sigma = new AdamW(model.sigmaNetwork.Parameters(), lr:hp.actorLearningRate);
            }
            else
            {
                optim_q1q2 = JsonUtility.FromJson<AdamW>(optimizer_states[0]);
                optim_q1q2.parameters = model.q1Network.Parameters();

                optim_mu = JsonUtility.FromJson<AdamW>(optimizer_states[1]);
                optim_mu.parameters = model.muNetwork.Parameters();

                optim_sigma = JsonUtility.FromJson<AdamW>(optimizer_states[2]);
                optim_sigma.parameters = model.sigmaNetwork.Parameters();
            }


            // Init schedulers
            optim_q1q2.Scheduler = new LinearAnnealing(optim_q1q2, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);
            optim_mu.Scheduler = new LinearAnnealing(optim_mu, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);
            optim_sigma.Scheduler = new LinearAnnealing(optim_sigma, start_factor: 1f, end_factor: 0f, total_iters: (int)model.config.maxSteps);

            // Init target networks
            q1TargNetwork = model.q1Network.Clone() as Sequential;
            q2TargNetwork = model.q2Network.Clone() as Sequential;
            q1TargNetwork.RequiresGrad = false;
            q2TargNetwork.RequiresGrad = false;

            // Set devices
            q1TargNetwork.Device = model.trainingDevice;
            q2TargNetwork.Device = model.trainingDevice;
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
                    train_data.frames.RemoveRange(0, train_data.Count / 4); // If buffer is full, remove old quarter

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
                    optim_q1q2.Scheduler.Step();
                    optim_mu.Scheduler.Step();
                    optim_sigma.Scheduler.Step();
                }
            }
        }
        private void ComputeQTargets(TimestepTuple[] batch, out Tensor y)
        {
            Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());

            Tensor u_prime, probsPi;
            model.ContinuousEval(sPrime, out u_prime, out probsPi);

            // Compute log π(a'|s'):
            // log π = Σ log N(u|μ,σ) - Σ log(1 - tanh²(u))

            Tensor logProbsPi = probsPi.Log().Sum(-1, keepDim: true) - (2.0f * (MathF.Log(2.0f) - u_prime - (-2.0f * u_prime).Softplus())).Sum(-1, keepDim: true); ;

            Tensor a_prime = u_prime.Tanh();
            Tensor pair_sPrime_aPrime = StateActionPair(sPrime, a_prime);
            Tensor Qtarg1 = q1TargNetwork.Predict(pair_sPrime_aPrime);
            Tensor Qtarg2 = q2TargNetwork.Predict(pair_sPrime_aPrime);

            Parallel.For(0, batch.Length, b =>
            {
                float r = batch[b].reward[0];
                float d = batch[b].done[0];
                float Qt1 = Qtarg1[b, 0];
                float Qt2 = Qtarg2[b, 0];
                float logPi = logProbsPi[b, 0];

                // Target: y = r + γ(1-d)[min(Q₁,Q₂) - α·log π]
                float _y = r + hp.gamma * (1f - d) * (MathF.Min(Qt1, Qt2) - hp.alpha * logPi);

                batch[b].q_target = Tensor.Constant(_y);
            });

            y = Tensor.Concat(null, batch.Select(x => x.q_target).ToArray());
        }

        private void UpdateQFunctions(Tensor states, Tensor raw_continuous_actions, Tensor Q_targets)
        {
            model.q1Network.RequiresGrad = true;
            model.q2Network.RequiresGrad = true;

            // raw_continuous_actions should be UNSQUASHED actions from replay buffer
            // Squash them for Q-network input
            Tensor actions_squashed = raw_continuous_actions.Tanh();
            Tensor state_actionPair = StateActionPair(states, actions_squashed);

            Tensor Q1_s_a = model.q1Network.Forward(state_actionPair);
            Tensor Q2_s_a = model.q2Network.Forward(state_actionPair);

            Loss q1Loss = Loss.MSE(Q1_s_a, Q_targets);
            Loss q2Loss = Loss.MSE(Q2_s_a, Q_targets);
            criticLoss = (q1Loss.Item + q2Loss.Item) / 2;

            optim_q1q2.ZeroGrad();
            model.q1Network.Backward(q1Loss.Grad * 0.5f);
            model.q2Network.Backward(q2Loss.Grad * 0.5f);
            optim_q1q2.Step();
        }


        private void UpdatePolicy(Tensor states)
        {
            model.q1Network.RequiresGrad = false;
            model.q2Network.RequiresGrad = false;

            int batch_size = states.Size(0);
            Tensor aTildeS, u, mu, sigma, ksi;
            model.ContinuousForward(states, out mu, out sigma);

            // 1. SAFETY: Even with Softplus, sigma can be near-zero (e.g. 1e-7), 
            // which causes Log(sigma) to explode to -16. Clamp it.
            sigma = sigma.Clip(1e-6f, 10f);

            ksi = Tensor.RandomNormal(sigma.Shape);
            u = mu + sigma * ksi;
            aTildeS = u.Tanh();
            int D = aTildeS.Size(-1);

            // 2. NUMERICALLY STABLE LOG-PROB CALCULATION
            // Instead of calculating 'muDist' (linear prob) and Logging it, compute Log directly.
            // log N(u|mu, sig) = -D/2 * log(2pi) - sum(log(sigma)) - 0.5 * sum(ksi^2)

            float log2Pi = MathF.Log(2f * MathF.PI);
            Tensor logSigmaSum = sigma.Log().Sum(-1, keepDim: true);
            Tensor ksiSqSum = ksi.Pow(2f).Sum(-1, keepDim: true);

            Tensor logPi_Gaussian = -0.5f * D * log2Pi - logSigmaSum - 0.5f * ksiSqSum;

            // Tanh Correction (Same as before, this part was correct)
            Tensor tanh_correction = Tensor.Sum(2.0f * (MathF.Log(2.0f) - u - Tensor.Softplus(-2.0f * u)), -1, true);

            Tensor logPI_aThetaTildeS = logPi_Gaussian - tanh_correction;

            // 3. Compute Q-Loss (Manual Graph Connection)
            // Since you are doing manual backprop, StateActionPair is fine, 
            // provided ExtractActionFromStateAction is correct.
            Tensor pair_states_aTildeS = StateActionPair(states, aTildeS);
            Tensor Q1s_aTildeS = model.q1Network.Forward(pair_states_aTildeS);
            Tensor Q2s_aTildeS = model.q2Network.Forward(pair_states_aTildeS);

            Tensor objectiveFunctionJ = Tensor.Minimum(Q1s_aTildeS, Q2s_aTildeS) - hp.alpha * logPI_aThetaTildeS;
            actorLoss = objectiveFunctionJ.Average();

            // -------------------------------------------------------
            // 4. Manual Differentiation
            // -------------------------------------------------------

            // A. Q-Function Gradients
            Tensor dminQ1Q2_dQ1 = Q1s_aTildeS <= Q2s_aTildeS;
            Tensor dminQ1Q2_dQ2 = Tensor.LogicalNot(dminQ1Q2_dQ1);

            // Get Gradient w.r.t Input (State+Action)
            Tensor dminQ1Q2_ds_aTildeS = model.q1Network.Backward(dminQ1Q2_dQ1) + model.q2Network.Backward(dminQ1Q2_dQ2);
            Tensor dminQ1Q2_daTildeS = ExtractActionFromStateAction(dminQ1Q2_ds_aTildeS, states.Size(-1), aTildeS.Size(-1));

            Tensor dminQ1Q2_du = dminQ1Q2_daTildeS * (1f - aTildeS.Pow(2f)); // 1 - tanh^2
            Tensor dminQ1Q2_dMu = dminQ1Q2_du;
            Tensor dminQ1Q2_dSigma = dminQ1Q2_du * ksi;

            // B. Entropy Gradients
            // We need gradients of: -alpha * logPi
            // Note: Your previous 'dAlphaLogPi' calculations were correct in derivation, 
            // but let's align them with the stable LogPi calculation.

            // Gradient of LogGaussian w.r.t u (Reparameterized) is 0 (as per your correct analysis).
            // Gradient of LogGaussian w.r.t Sigma (Reparameterized) is -1/sigma.

            // Derivative of Tanh Correction w.r.t u
            // d/du [ 2(log2 - u - softplus(-2u)) ] = 2 * ( -1 - sigmoid(-2u)*(-2) ) = -2 + 4*sigmoid(-2u)
            // This matches your previous 'd2_Log2_u_sp2u_du' logic, but let's reuse the formula for clarity
            // or keep your efficient implementation:
            Tensor dCorrection_du = (2f - 2f * Tensor.Exp(2f * u)) / (Tensor.Exp(2f * u) + 1f);

            // Total d(LogPi)/du = d(LogGaussian)/du - d(Correction)/du
            // d(LogGaussian)/du = -(u-mu)/sigma^2.
            Tensor dLogGaussian_du = -ksi / sigma;

            Tensor dLogPi_du = dLogGaussian_du - dCorrection_du;

            // C. Combine Gradients
            // We want to MAXIMIZE J.
            // dJ/dTheta = dQ/dTheta - alpha * dLogPi/dTheta

            // For Mu:
            // dQ/dMu = dQ/du * 1
            // dLogPi/dMu (Total) = dLogPi/du * 1 + Partial_Mu(LogGaussian)
            // Partial_Mu(LogGaussian) = (u-mu)/sigma^2 = ksi/sigma.
            // Total dLogPi/dMu = (-ksi/sigma - dCorrection_du) + ksi/sigma = -dCorrection_du.

            Tensor dJ_dMu = dminQ1Q2_dMu - hp.alpha * (-dCorrection_du);

            // For Sigma:
            // dQ/dSigma = dQ/du * ksi
            // dLogPi/dSigma (Total) = dLogPi/du * ksi + Partial_Sigma(LogGaussian)
            // Partial_Sigma(LogGaussian) = -1/sigma + (u-mu)^2/sigma^3 = -1/sigma + ksi^2/sigma.
            // Total dLogPi/dSigma = (-ksi/sigma - dCorrection_du)*ksi + (-1/sigma + ksi^2/sigma)
            //                     = -ksi^2/sigma - dCorrection_du*ksi - 1/sigma + ksi^2/sigma
            //                     = -dCorrection_du*ksi - 1/sigma.

            Tensor dJ_dSigma = dminQ1Q2_dSigma - hp.alpha * (-dCorrection_du * ksi - 1f / sigma);

            // 5. Step (Ascent)
            // If optimizer minimizes (theta = theta - grad), we pass -dJ.
            optim_mu.ZeroGrad();
            model.muNetwork.Backward(-dJ_dMu);
            optim_mu.Step();

            optim_sigma.ZeroGrad();
            model.sigmaNetwork.Backward(-dJ_dSigma);
            optim_sigma.Step();
        }

        public void UpdateTargetNetworks()
        {
            Tensor[] phi1 = model.q1Network.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi2 = model.q2Network.Parameters().Select(x => x.param).ToArray();

            Tensor[] phi_targ1 = q1TargNetwork.Parameters().Select(x => x.param).ToArray();
            Tensor[] phi_targ2 = q2TargNetwork.Parameters().Select(x => x.param).ToArray();

            // We update the target q functions softly...
            // OpenAI algorithm uses polyak = 0.995, the same thing with using τ = 0.005
            // φtarg,i <- (1 - τ)φtarg,i + τφi     for i = 1,2

            Parallel.For(0, phi1.Length, i =>
            {
                Tensor.CopyTo(fromTensor: (1f - hp.tau) * phi_targ1[i] + hp.tau * phi1[i], toTensor: phi_targ1[i]);
                Tensor.CopyTo(fromTensor: (1f - hp.tau) * phi_targ2[i] + hp.tau * phi2[i], toTensor: phi_targ2[i]);
            });
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

        protected override string[] SerializeOptimizerStates()
        {
            List<string> states = new List<string>();
            states.Add(JsonUtility.ToJson(optim_q1q2, true));
            states.Add(JsonUtility.ToJson(optim_mu, true));
            states.Add(JsonUtility.ToJson(optim_sigma, true));



            return states.ToArray();
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