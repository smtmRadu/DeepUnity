using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Optimizers;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
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
    // [SAC Diagnostics Checklist]
    // 1) Delete this behaviour's SAC OptimStates files.
    // 2) Start from fresh behaviour weights.
    // 3) Start a new training session from step 0.
    // Reusing stale replay/optimizer state can mask SAC root-cause signals.
    internal sealed class SACTrainer : DeepUnityTrainer, IOffPolicy
    {
        private struct SACPolicyDiagnostics
        {
            public float jMean;
            public float minQMean;
            public float entropyTermMean;
            public float logPiMean;
            public float sigmaMean;
            public float sigmaMin;
            public float sigmaMax;
            public float actionSaturationRatio;
            public float dJdMuL2PerElem;
            public float dJdSigmaL2PerElem;
            public float dQdaL2PerElem;
        }
        private struct SACTargetDiagnostics
        {
            public float qTargetMean;
            public float qTargetStd;
            public float qTargetMinMean;
            public float targetEntropyTermMean;
        }

        // Q target networks
        private static Sequential q1TargNetwork;
        private static Sequential q2TargNetwork;
        private int new_experiences_collected = 0;
        private long sacGradientStep = 0;
        private SACPolicyDiagnostics policyDiagnostics;
        private SACTargetDiagnostics targetDiagnostics;

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
                optim_q1q2.parameters = model.q1Network.Parameters().Concat(model.q2Network.Parameters()).ToArray();

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
            if (hp.sacDebugMetrics)
            {
                ConsoleMessage.Info("[SAC] Diagnostics mode is ON. For valid root-cause isolation run a hard reset: clear SAC OptimStates + fresh behaviour weights + new session.");
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

                    // SAC critic always expects environment actions in [-1, 1].
                    // During trainable-std phase, actions are stored unsquashed (u), so squash once before replay insert.
                    // During random warmup, actions are already in [-1, 1], so we keep them unchanged.
                    if (model.stochasticity == Stochasticity.TrainableStandardDeviation)
                    {
                        foreach (var ts in agent_mem.frames)
                        {
                            if (ts.action_continuous != null)
                                ts.action_continuous = ts.action_continuous.Tanh();
                        }
                    }

                    train_data.TryAppend(agent_mem.frames, hp.replayBufferSize);
                    agent_mem.Clear();
                }

                if (train_data.Count >= hp.updateAfter)
                {
                    model.stochasticity = Stochasticity.TrainableStandardDeviation;

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
            model.muNetwork.Device = model.trainingDevice;
            model.sigmaNetwork.Device = model.trainingDevice;

            for (int epoch_index = 0; epoch_index < hp.updatesNum; epoch_index++)
            {
                // Sample a random batch of transitions from the replay buffer
                TimestepTuple[] batch = Utils.Random.Sample(hp.minibatchSize, train_data.frames);

                // Batchify
                Tensor states = Tensor.Concat(null, batch.Select(x => x.state).ToArray());
                Tensor raw_continuous_actions = Tensor.Concat(null, batch.Select(x => x.action_continuous).ToArray());
                Tensor q_targets;
                long currentSacStep = ++sacGradientStep;

                ComputeQTargets(batch, out q_targets, currentSacStep);
                UpdateQFunctions(states, raw_continuous_actions, q_targets);
                UpdatePolicy(states, currentSacStep);
                UpdateTargetNetworks();

                if (ShouldLogSacDiagnostics(currentSacStep))
                    LogSacDiagnostics(currentSacStep);

                if (hp.LRSchedule)
                {
                    optim_q1q2.Scheduler.Step();
                    optim_mu.Scheduler.Step();
                    optim_sigma.Scheduler.Step();
                }
            }

            model.muNetwork.Device = model.inferenceDevice;
            model.sigmaNetwork.Device = model.inferenceDevice;
        }
        private void ComputeQTargets(TimestepTuple[] batch, out Tensor y, long currentSacStep)
        {
            Tensor sPrime = Tensor.Concat(null, batch.Select(x => x.nextState).ToArray());
            Tensor mu_prime, sigma_prime;
            model.ContinuousForward(sPrime, out mu_prime, out sigma_prime);
            sigma_prime = sigma_prime.Clip(1e-6f, 10f);

            Tensor ksi_prime = Tensor.RandomNormal(sigma_prime.Shape);
            Tensor u_prime = mu_prime + sigma_prime * ksi_prime;
            Tensor a_prime = u_prime.Tanh();

            int D = a_prime.Size(-1);
            float log2Pi = MathF.Log(2f * MathF.PI);
            Tensor logSigmaSum = sigma_prime.Log().Sum(-1, keepDim: true);
            Tensor ksiSqSum = ksi_prime.Pow(2f).Sum(-1, keepDim: true);
            Tensor logPiGaussian = -0.5f * D * log2Pi - logSigmaSum - 0.5f * ksiSqSum;
            Tensor tanhCorrection = Tensor.Sum(2.0f * (MathF.Log(2.0f) - u_prime - Tensor.Softplus(-2.0f * u_prime)), -1, true);
            Tensor logProbsPi = logPiGaussian - tanhCorrection;
            Tensor targetEntropyTerm = -hp.alpha * logProbsPi;

            Tensor pair_sPrime_aPrime = StateActionPair(sPrime, a_prime);
            Tensor Qtarg1 = q1TargNetwork.Predict(pair_sPrime_aPrime);
            Tensor Qtarg2 = q2TargNetwork.Predict(pair_sPrime_aPrime);
            Tensor qTargetMin = Tensor.Minimum(Qtarg1, Qtarg2);

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

            targetDiagnostics.qTargetMean = y.Average();
            targetDiagnostics.qTargetStd = StdOf(y);
            targetDiagnostics.qTargetMinMean = qTargetMin.Average();
            targetDiagnostics.targetEntropyTermMean = targetEntropyTerm.Average();

            if (hp.sacDebugMetrics && HasNonFinite(logProbsPi))
            {
                ConsoleMessage.Warning($"[SAC][step {currentSacStep}] Non-finite values detected in target logPi.");
            }
        }

        private void UpdateQFunctions(Tensor states, Tensor continuous_actions, Tensor Q_targets)
        {
            model.q1Network.RequiresGrad = true;
            model.q2Network.RequiresGrad = true;

            Tensor state_actionPair = StateActionPair(states, continuous_actions);

            Tensor Q1_s_a = model.q1Network.Forward(state_actionPair);
            Tensor Q2_s_a = model.q2Network.Forward(state_actionPair);

            Loss q1Loss = Loss.MSE(Q1_s_a, Q_targets);
            Loss q2Loss = Loss.MSE(Q2_s_a, Q_targets);
            criticLoss += (q1Loss.Item + q2Loss.Item) / 2;

            optim_q1q2.ZeroGrad();
            model.q1Network.Backward(q1Loss.Grad * 0.5f);
            model.q2Network.Backward(q2Loss.Grad * 0.5f);
            optim_q1q2.Step();
        }


        private void UpdatePolicy(Tensor states, long currentSacStep)
        {
            model.q1Network.RequiresGrad = false;
            model.q2Network.RequiresGrad = false;

            int batch_size = states.Size(0);
            Tensor aTildeS, u, mu, sigmaPreClip, sigma, ksi;
            model.ContinuousForward(states, out mu, out sigmaPreClip);

            // 1. SAFETY: Even with Softplus, sigma can be near-zero (e.g. 1e-7), 
            // which causes Log(sigma) to explode to -16. Clamp it.
            sigma = sigmaPreClip.Clip(1e-6f, 10f);

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

            Tensor minQ = Tensor.Minimum(Q1s_aTildeS, Q2s_aTildeS);
            Tensor entropyTerm = -hp.alpha * logPI_aThetaTildeS;
            Tensor objectiveFunctionJ = minQ + entropyTerm;
            actorLoss += objectiveFunctionJ.Average();

            // -------------------------------------------------------
            // 4. Manual Differentiation
            // -------------------------------------------------------

            // A. Q-Function Gradients
            Tensor dminQ1Q2_dQ1 = Q1s_aTildeS <= Q2s_aTildeS;
            Tensor dminQ1Q2_dQ2 = Tensor.LogicalNot(dminQ1Q2_dQ1);

            // Get Gradient w.r.t Input (State+Action)
            Tensor dminQ1Q2_ds_aTildeS = model.q1Network.Backward(dminQ1Q2_dQ1) + model.q2Network.Backward(dminQ1Q2_dQ2);
            Tensor dminQ1Q2_daTildeS = ExtractActionFromStateAction(dminQ1Q2_ds_aTildeS, states.Size(-1), aTildeS.Size(-1));

            Tensor dQ_du = dminQ1Q2_daTildeS * (1f - aTildeS.Pow(2f)); // dQ/du = dQ/da * (1 - tanh(u)^2)

            // B. Entropy Gradients
            // We compute log π using the reparameterized noise ksi:
            // log π = const - sum(log σ) - 0.5 * sum(ksi^2) - tanh_correction(u).
            // In this form ksi is sampled noise and does not depend on μ, so the Gaussian term
            // contributes no μ-gradient. Only the tanh correction flows through u = μ + σ·ξ.
            Tensor dCorrection_du = -2f * aTildeS;
            Tensor dLogPi_dMu = -dCorrection_du;
            Tensor dLogPi_dSigma = -dCorrection_du * ksi - 1f / sigma;

            // C. Combine Gradients (maximize J = minQ - alpha * logpi)
            Tensor dJ_dMu = dQ_du - hp.alpha * dLogPi_dMu;
            Tensor dJ_dSigma = dQ_du * ksi - hp.alpha * dLogPi_dSigma;

            // Chain rule for sigma path:
            // sigma = Clip(sigma_pre, lo, hi), sigma_pre = sigma_net_output * standardDeviationScale
            Tensor sigmaClipMask = sigmaPreClip.Select(x => x >= 1e-6f && x <= 10f ? 1f : 0f);

            // #region agent log
            if (currentSacStep % 50 == 0)
            {
                float sigmaClipMaskMean = sigmaClipMask.Average();
                float actorLR = optim_mu.gamma;
                long maxSteps = model.config.maxSteps;
                float stepRatio = maxSteps > 0 ? (float)currentSacStep / maxSteps : 0f;
                float dQduL2 = L2PerElement(dminQ1Q2_daTildeS);
                float dJdMuL2 = L2PerElement(dJ_dMu);
                float jVal = objectiveFunctionJ.Average();
                float minQVal = minQ.Average();
                float entVal = entropyTerm.Average();
                AgentDebugLog("SACTrainer.cs:UpdatePolicy", "sigmaClipMask", "{\"sigmaClipMaskMean\":" + sigmaClipMaskMean + ",\"step\":" + currentSacStep + "}", "D");
                AgentDebugLog("SACTrainer.cs:UpdatePolicy", "LR", "{\"actorLR\":" + actorLR + ",\"sacStep\":" + currentSacStep + ",\"maxSteps\":" + maxSteps + ",\"stepRatio\":" + stepRatio + "}", "E");
                AgentDebugLog("SACTrainer.cs:UpdatePolicy", "gradient magnitude", "{\"dQduL2\":" + dQduL2 + ",\"dJdMuL2\":" + dJdMuL2 + ",\"step\":" + currentSacStep + "}", "F");
                AgentDebugLog("SACTrainer.cs:UpdatePolicy", "J trend", "{\"J\":" + jVal + ",\"minQ\":" + minQVal + ",\"entropyTerm\":" + entVal + ",\"step\":" + currentSacStep + "}", "B");
                if (currentSacStep % 200 == 0)
                    ConsoleMessage.Info($"[SAC-DEBUG step={currentSacStep}] J={jVal:F4} minQ={minQVal:F4} ent={entVal:F4} dQduL2={dQduL2:G4} dJdMuL2={dJdMuL2:G4} actorLR={actorLR:G4}");
            }
            // #endregion agent log
            Tensor dJ_dSigmaPreClip = dJ_dSigma * sigmaClipMask;
            Tensor dJ_dSigmaNetOut = dJ_dSigmaPreClip * model.standardDeviationScale;

            // 5. Step (Ascent)
            // If optimizer minimizes (theta = theta - grad), we pass -dJ.
            optim_mu.ZeroGrad();
            model.muNetwork.Backward(-dJ_dMu);
            if (hp.maxNorm > 0f) optim_mu.ClipGradNorm(hp.maxNorm);
            optim_mu.Step();

            optim_sigma.ZeroGrad();
            model.sigmaNetwork.Backward(-dJ_dSigmaNetOut);
            if (hp.maxNorm > 0f) optim_sigma.ClipGradNorm(hp.maxNorm);
            optim_sigma.Step();

            // restored q1/q2 network device removed

            policyDiagnostics.jMean = objectiveFunctionJ.Average();
            policyDiagnostics.minQMean = minQ.Average();
            policyDiagnostics.entropyTermMean = entropyTerm.Average();
            policyDiagnostics.logPiMean = logPI_aThetaTildeS.Average();
            policyDiagnostics.sigmaMean = sigma.Average();
            policyDiagnostics.sigmaMin = sigma.Min();
            policyDiagnostics.sigmaMax = sigma.Max();
            policyDiagnostics.actionSaturationRatio = aTildeS.Select(x => MathF.Abs(x) > 0.95f ? 1f : 0f).Average();
            policyDiagnostics.dJdMuL2PerElem = L2PerElement(dJ_dMu);
            policyDiagnostics.dJdSigmaL2PerElem = L2PerElement(dJ_dSigma);
            policyDiagnostics.dQdaL2PerElem = L2PerElement(dminQ1Q2_daTildeS);

            if (hp.sacDebugMetrics
                && (HasNonFinite(logPI_aThetaTildeS)
                    || HasNonFinite(sigma)
                    || HasNonFinite(dJ_dMu)
                    || HasNonFinite(dJ_dSigma)
                    || HasNonFinite(dJ_dSigmaNetOut)))
            {
                ConsoleMessage.Warning($"[SAC][step {currentSacStep}] Non-finite values detected in policy path (logPi/sigma/dJdMu/dJdSigma).");
            }
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

        // #region agent log
        private static bool _debugLogPathPrinted;
        private static void AgentDebugLog(string location, string message, string dataJson, string hypothesisId = null)
        {
            var line = "{\"sessionId\":\"b67f4f\",\"location\":\"" + location + "\",\"message\":\"" + message.Replace("\"", "\\\"") + "\",\"data\":" + dataJson + ",\"timestamp\":" + DateTimeOffset.UtcNow.ToUnixTimeMilliseconds() + (string.IsNullOrEmpty(hypothesisId) ? "" : ",\"hypothesisId\":\"" + hypothesisId + "\"") + "}\n";
            var paths = new[] { Path.Combine(Application.dataPath, "DeepUnity", "ReinforcementLearning", "Base", "debug-b67f4f.log"), Path.Combine(Application.persistentDataPath, "debug-b67f4f.log") };
            foreach (var logPath in paths)
            {
                try
                {
                    var logDir = Path.GetDirectoryName(logPath);
                    if (!string.IsNullOrEmpty(logDir) && !Directory.Exists(logDir)) Directory.CreateDirectory(logDir);
                    File.AppendAllText(logPath, line);
                    if (!_debugLogPathPrinted) { ConsoleMessage.Info("[SAC-DEBUG] Log file: " + logPath); _debugLogPathPrinted = true; }
                    return;
                }
                catch { }
            }
        }
        // #endregion agent log

        private bool ShouldLogSacDiagnostics(long step)
        {
            if (!hp.sacDebugMetrics)
                return false;

            int every = Math.Max(1, hp.sacDebugEveryNUpdates);
            return step % every == 0;
        }
        private void LogSacDiagnostics(long step)
        {
            ConsoleMessage.Info(
                $"[SAC-DIAG][step {step}] " +
                $"J={policyDiagnostics.jMean:0.0000} | minQ={policyDiagnostics.minQMean:0.0000} | ent(-a*logpi)={policyDiagnostics.entropyTermMean:0.0000} | logpi={policyDiagnostics.logPiMean:0.0000} | " +
                $"sigma(mean/min/max)=({policyDiagnostics.sigmaMean:0.000000}/{policyDiagnostics.sigmaMin:0.000000}/{policyDiagnostics.sigmaMax:0.000000}) | " +
                $"sat(|a|>.95)={policyDiagnostics.actionSaturationRatio:P2} | " +
                $"||dJdMu||/N={policyDiagnostics.dJdMuL2PerElem:0.000000e+0} | ||dJdSigma||/N={policyDiagnostics.dJdSigmaL2PerElem:0.000000e+0} | ||dQda||/N={policyDiagnostics.dQdaL2PerElem:0.000000e+0} | " +
                $"Qtarget(mean/std)={targetDiagnostics.qTargetMean:0.0000}/{targetDiagnostics.qTargetStd:0.0000} | qtarg_min_mean={targetDiagnostics.qTargetMinMean:0.0000} | target_ent_term={targetDiagnostics.targetEntropyTermMean:0.0000}");
        }
        private static bool HasNonFinite(Tensor tensor)
        {
            return tensor.Any(x => float.IsNaN(x) || float.IsInfinity(x));
        }
        private static float L2PerElement(Tensor tensor)
        {
            float l2 = Tensor.Norm(tensor, NormType.EuclideanL2)[0];
            return l2 / MathF.Max(1f, tensor.Count());
        }
        private static float StdOf(Tensor tensor)
        {
            float mean = tensor.Average();
            float variance = tensor.Select(x => (x - mean) * (x - mean)).Average();
            return MathF.Sqrt(MathF.Max(0f, variance));
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
