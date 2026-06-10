// SACGPUTrainer — sibling to SACTrainer, fully GPU-resident (FullGPU path).
//
// Built on the same proven stack as PPOGPUTrainer (GPUMLP + GPUAdamW + RLBoosterCS)
// plus SACLossCS for the SAC-specific math. Per gradient step:
//   1. CPU samples a replay minibatch, uploads states / nextStates / actions /
//      rewards / dones / truncs / fresh Gaussian noise.
//   2. Target pass  : a' = tanh(mu(s') + sigma(s')*ksi'), logpi' (SacSampleLogPi)
//                     y  = r + gamma*(1-d+tr)*(min(Q1t,Q2t)(s',a') - alpha*logpi')
//   3. Critic pass  : dQ = (Q(s,a) - y) per-sample -> backward -> fused GPU AdamW.
//   4. Actor pass   : a~ = tanh(mu(s)+sigma(s)*ksi); dQ/da via Q-net input grads
//                     (RequiresGrad off, weights untouched); -dJ/dmu, -dJ/dsigma
//                     assembled on GPU (SacActorGrad) -> backward -> AdamW.
//   5. Polyak       : target params lerped towards online params on GPU.
//
// Deliberate config differences from the (broken-by-default) CPU SACTrainer history:
//   - actor optimizers have weight_decay = 0 and NO gradient clipping (canonical SAC)
//   - the LR schedule counts gradient steps, not env steps
// Everything else (replay semantics, warmup, squash-at-insert, UTD gating) matches
// SACTrainer so CPU vs GPU runs are directly comparable.
//
// Networks must be MLP / LnMLP (Dense / RMSNorm / activations / Softplus only).

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using DeepUnity.ReinforcementLearning.FullGPU;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    internal sealed class SACGPUTrainer : DeepUnityTrainer, IOffPolicy
    {
        const float epsilon_adam = 1e-5f;
        const float adamBeta1 = 0.9f;
        const float adamBeta2 = 0.999f;

        // ------------ GPU networks ------------
        GPUMLP gQ1, gQ2, gQ1T, gQ2T, gMu, gSigma;
        GPUAdamW oQ, oMu, oSigma;
        // CPU clones backing the GPU target networks (kept alive for upload/download round-trips).
        Sequential q1TargCpu, q2TargCpu;
        // (online, target) GPU param pairs for the Polyak kernel.
        (GPUParam online, GPUParam target)[] polyakPairs;

        // ------------ shaders ------------
        ComputeShader sacLoss;
        int kSampleLogPi, kWritePair, kQTarget, kCriticGrad, kMinQSelect, kAddInto, kSliceActGrad, kActorGrad, kPolyak;

        // ------------ persistent batch buffers ------------
        int maxB;
        int S, A;
        ComputeBuffer bufStates;      // [maxB, S]
        ComputeBuffer bufNextStates;  // [maxB, S]
        ComputeBuffer bufPairCritic;  // [maxB, S+A] CPU-concatenated (state, replay action)
        ComputeBuffer bufPair;        // [maxB, S+A] GPU-concatenated (state, fresh action)
        ComputeBuffer bufDPair;       // [maxB, S+A] summed Q-net input grads
        ComputeBuffer bufNoise;       // [maxB, A] ksi for the actor pass
        ComputeBuffer bufNoisePrime;  // [maxB, A] ksi' for the target pass
        ComputeBuffer bufAct;         // [maxB, A] sampled squashed actions
        ComputeBuffer bufLogPi;       // [maxB]
        ComputeBuffer bufRew, bufDone, bufTrunc, bufY, bufDQ1, bufDQ2, bufJ; // [maxB]
        ComputeBuffer bufDQda;        // [maxB, A]
        ComputeBuffer bufDMu, bufDSig;// [maxB, A]

        // scratch readback arrays (logging)
        float[] _scrQ1, _scrQ2, _scrY, _scrJ;

        // LR schedule (gradient-step based, like the fixed CPU SAC)
        int totalGradSteps;
        long sacGradientStep = 0;
        float lrA0, lrC0;

        int new_experiences_collected = 0;

        protected override void Initialize(string[] optimizer_states)
        {
            if (model.IsUsingContinuousActions && model.muNetwork.Modules.Last() is Tanh)
                model.muNetwork.Modules = model.muNetwork.Modules.Take(model.muNetwork.Modules.Length - 1).ToArray();

            if (!model.IsUsingContinuousActions)
                throw new Exception("SACGPU supports continuous action spaces only (like SAC).");
            if (model.IsUsingDiscreteActions)
                ConsoleMessage.Warning("SACGPU ignores discrete actions (SAC is continuous-only in DeepUnity).");

            sacLoss = Resources.Load<ComputeShader>("ComputeShaders/SACLossCS");
            if (sacLoss == null)
                throw new Exception("SACGPU shader missing (Resources/ComputeShaders/SACLossCS).");

            kSampleLogPi = sacLoss.FindKernel("SacSampleLogPi");
            kWritePair   = sacLoss.FindKernel("WritePairSA");
            kQTarget     = sacLoss.FindKernel("SacQTarget");
            kCriticGrad  = sacLoss.FindKernel("SacCriticGrad");
            kMinQSelect  = sacLoss.FindKernel("SacMinQSelect");
            kAddInto     = sacLoss.FindKernel("AddInto");
            kSliceActGrad= sacLoss.FindKernel("SliceActionGrad");
            kActorGrad   = sacLoss.FindKernel("SacActorGrad");
            kPolyak      = sacLoss.FindKernel("Polyak");

            ValidateMlpOnly(model.q1Network, "Q1");
            ValidateMlpOnly(model.q2Network, "Q2");
            ValidateMlpOnly(model.muNetwork, "Mu");
            ValidateMlpOnly(model.sigmaNetwork, "Sigma");

            maxB = hp.minibatchSize;
            S = model.observationSize * model.stackedInputs;
            A = model.continuousDim;

            // ---- GPU networks ----
            gQ1 = new GPUMLP(model.q1Network, maxB);
            gQ2 = new GPUMLP(model.q2Network, maxB);
            gMu = new GPUMLP(model.muNetwork, maxB);
            gSigma = new GPUMLP(model.sigmaNetwork, maxB);

            // Target networks: fresh CPU clones -> GPU (recreated every session, like SACTrainer).
            q1TargCpu = model.q1Network.Clone() as Sequential;
            q2TargCpu = model.q2Network.Clone() as Sequential;
            gQ1T = new GPUMLP(q1TargCpu, maxB);
            gQ2T = new GPUMLP(q2TargCpu, maxB);
            gQ1T.RequiresGrad = false;
            gQ2T.RequiresGrad = false;

            polyakPairs = gQ1.Parameters().Zip(gQ1T.Parameters(), (o, t) => (o, t))
                .Concat(gQ2.Parameters().Zip(gQ2T.Parameters(), (o, t) => (o, t)))
                .ToArray();

            // ---- optimizers (canonical SAC config: zero weight decay, no grad clipping) ----
            oQ = new GPUAdamW(gQ1.Parameters().Concat(gQ2.Parameters()),
                              hp.criticLearningRate, adamBeta1, adamBeta2, epsilon_adam, 0f, -1f);
            oMu = new GPUAdamW(gMu.Parameters(), hp.actorLearningRate, adamBeta1, adamBeta2, epsilon_adam, 0f, -1f);
            oSigma = new GPUAdamW(gSigma.Parameters(), hp.actorLearningRate, adamBeta1, adamBeta2, epsilon_adam, 0f, -1f);

            // LR schedule over GRADIENT steps (mirrors the fixed CPU formula).
            totalGradSteps = Math.Max(1, (int)Math.Min(int.MaxValue, (long)hp.maxSteps * hp.updatesNum / Math.Max(1, hp.updateInterval)));
            lrA0 = hp.actorLearningRate;
            lrC0 = hp.criticLearningRate;

            // ---- persistent buffers ----
            bufStates     = New(maxB * S);
            bufNextStates = New(maxB * S);
            bufPairCritic = New(maxB * (S + A));
            bufPair       = New(maxB * (S + A));
            bufDPair      = New(maxB * (S + A));
            bufNoise      = New(maxB * A);
            bufNoisePrime = New(maxB * A);
            bufAct        = New(maxB * A);
            bufLogPi      = New(maxB);
            bufRew  = New(maxB); bufDone = New(maxB); bufTrunc = New(maxB);
            bufY    = New(maxB); bufDQ1  = New(maxB); bufDQ2   = New(maxB); bufJ = New(maxB);
            bufDQda = New(maxB * A);
            bufDMu  = New(maxB * A);
            bufDSig = New(maxB * A);

            _scrQ1 = new float[maxB]; _scrQ2 = new float[maxB]; _scrY = new float[maxB]; _scrJ = new float[maxB];

            // CPU mirrors are only used for rollout inference (mu/sigma Predict) — keep them
            // on the inference device; the actual training math never touches them.
            model.q1Network.Device = model.inferenceDevice;
            model.q2Network.Device = model.inferenceDevice;
            model.muNetwork.Device = model.inferenceDevice;
            model.sigmaNetwork.Device = model.inferenceDevice;

            // Optimizer states from disk — shared JSON format & file key with the CPU SACTrainer,
            // order [q1q2 (combined), mu, sigma].
            if (optimizer_states != null && optimizer_states.Length >= 3)
            {
                try
                {
                    LoadGpuAdam(optimizer_states[0], gQ1.Parameters().Concat(gQ2.Parameters()).ToArray(), oQ, "q1q2");
                    LoadGpuAdam(optimizer_states[1], gMu.Parameters().ToArray(), oMu, "mu");
                    LoadGpuAdam(optimizer_states[2], gSigma.Parameters().ToArray(), oSigma, "sigma");
                    ConsoleMessage.Info("[SACGPU] Optimizer states loaded (shared CPU/GPU AdamW format).");
                }
                catch (Exception e)
                {
                    ConsoleMessage.Warning($"[SACGPU] Failed to load optimizer states, starting fresh: {e.Message}");
                }
            }

            // Warmup: purely random actions until the replay holds `updateAfter` transitions.
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

            float effectiveUtdRatio = hp.updatesNum / (float)Math.Max(1, hp.updateInterval);
            if (effectiveUtdRatio < 0.25f)
                ConsoleMessage.Warning($"[SACGPU] Effective update-to-data ratio is {effectiveUtdRatio:0.###} gradient steps per new environment step (`updatesNum / updateInterval`). Canonical SAC uses ~1.0. Consider `updateInterval = 1` or `updatesNum ≈ updateInterval`.");

            ConsoleMessage.Info("[SACGPU] FullGPU SAC initialized.");
        }

        static ComputeBuffer New(int n) => new ComputeBuffer(n, 4, ComputeBufferType.Structured);

        static void ValidateMlpOnly(Sequential net, string label)
        {
            foreach (var m in net.Modules)
            {
                if (m is Dense || m is RMSNorm || m is ReLU || m is Tanh || m is SiLU || m is GELU || m is Softplus)
                    continue;
                throw new Exception($"SACGPU: network '{label}' contains '{m.GetType().Name}' which is not supported by FullGPU. " +
                                    "Use ArchitectureType.MLP or ArchitectureType.LnMLP, or switch trainer to SAC.");
            }
        }

        protected override void OnBeforeFixedUpdate()
        {
            // Same gating / replay semantics as SACTrainer for 1:1 comparability.
            int decision_freq = parallelAgents[0].DecisionRequester.decisionPeriod;
            new_experiences_collected += parallelAgents.Count;
            if (new_experiences_collected < hp.updateInterval * decision_freq)
                return;

            if (train_data.Count >= hp.replayBufferSize)
                train_data.frames.RemoveRange(0, train_data.Count / 4);

            foreach (var agent_mem in parallelAgents.Select(x => x.Memory))
            {
                if (agent_mem.Count == 0)
                    continue;

                // SAC critic expects env actions in [-1,1]. Trainable-std rollouts store
                // unsquashed u, so squash once at replay insert (random warmup is already [-1,1]).
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

        void Train()
        {
            int B = hp.minibatchSize;

            for (int epoch_index = 0; epoch_index < hp.updatesNum; epoch_index++)
            {
                sacGradientStep++;

                TimestepTuple[] batch = Utils.Random.Sample(hp.minibatchSize, train_data.frames);
                UploadBatch(batch, B);

                ComputeQTargets(B);
                UpdateQFunctions(B);
                UpdatePolicy(B);
                UpdateTargetNetworks();

                if (hp.LRSchedule)
                {
                    float factor = Mathf.Max(0f, 1f - (float)sacGradientStep / totalGradSteps);
                    oQ.lr = lrC0 * factor;
                    oMu.lr = lrA0 * factor;
                    oSigma.lr = lrA0 * factor;
                }
            }

            // Rollout inference reads the CPU mu/sigma mirrors — sync them after training.
            gMu.DownloadToCpu();
            gSigma.DownloadToCpu();
        }

        // -------------------- batch upload --------------------
        void UploadBatch(TimestepTuple[] batch, int B)
        {
            float[] s  = new float[B * S];
            float[] sp = new float[B * S];
            float[] pairSA = new float[B * (S + A)];
            float[] rew = new float[B];
            float[] don = new float[B];
            float[] tru = new float[B];
            float[] n1 = new float[B * A];
            float[] n2 = new float[B * A];

            for (int i = 0; i < B; i++)
            {
                var f = batch[i];
                Buffer.BlockCopy(f.state.ToArray(), 0, s, i * S * 4, S * 4);
                Buffer.BlockCopy(f.nextState.ToArray(), 0, sp, i * S * 4, S * 4);

                // critic input pair: (state, replay action) concatenated on CPU
                Buffer.BlockCopy(f.state.ToArray(), 0, pairSA, i * (S + A) * 4, S * 4);
                Buffer.BlockCopy(f.action_continuous.ToArray(), 0, pairSA, (i * (S + A) + S) * 4, A * 4);

                rew[i] = f.reward[0];
                don[i] = f.done[0];
                tru[i] = f.truncated != null ? f.truncated[0] : 0f;
            }
            for (int i = 0; i < B * A; i++)
            {
                n1[i] = Utils.Random.Normal(0f, 1f, threadsafe: false);
                n2[i] = Utils.Random.Normal(0f, 1f, threadsafe: false);
            }

            bufStates.SetData(s);
            bufNextStates.SetData(sp);
            bufPairCritic.SetData(pairSA);
            bufRew.SetData(rew);
            bufDone.SetData(don);
            bufTrunc.SetData(tru);
            bufNoise.SetData(n1);
            bufNoisePrime.SetData(n2);
        }

        // -------------------- target computation --------------------
        void ComputeQTargets(int B)
        {
            // a' = tanh(mu(s') + sigma(s')*ksi'), logpi'
            ComputeBuffer muP = gMu.Forward(bufNextStates, B);
            ComputeBuffer sigP = gSigma.Forward(bufNextStates, B);
            DispatchSampleLogPi(B, muP, sigP, bufNoisePrime);

            // pair = (s', a')
            DispatchWritePair(B, bufNextStates);

            // y = r + gamma*(1-d+tr)*(min(Q1t,Q2t) - alpha*logpi')
            ComputeBuffer qt1 = gQ1T.Forward(bufPair, B);
            ComputeBuffer qt2 = gQ2T.Forward(bufPair, B);
            sacLoss.SetInt("B", B);
            sacLoss.SetFloat("alpha", hp.alpha);
            sacLoss.SetFloat("gamma_", hp.gamma);
            sacLoss.SetBuffer(kQTarget, "q1Buf", qt1);
            sacLoss.SetBuffer(kQTarget, "q2Buf", qt2);
            sacLoss.SetBuffer(kQTarget, "rewBuf", bufRew);
            sacLoss.SetBuffer(kQTarget, "doneBuf", bufDone);
            sacLoss.SetBuffer(kQTarget, "truncBuf", bufTrunc);
            sacLoss.SetBuffer(kQTarget, "logPiBuf", bufLogPi);
            sacLoss.SetBuffer(kQTarget, "yBuf", bufY);
            sacLoss.Dispatch(kQTarget, Div256(B), 1, 1);
        }

        // -------------------- critic update --------------------
        void UpdateQFunctions(int B)
        {
            ComputeBuffer q1 = gQ1.Forward(bufPairCritic, B);
            ComputeBuffer q2 = gQ2.Forward(bufPairCritic, B);

            // logging readback (B floats; scalar-cost like PPOGPU's per-minibatch readbacks)
            q1.GetData(_scrQ1, 0, 0, B);
            q2.GetData(_scrQ2, 0, 0, B);
            bufY.GetData(_scrY, 0, 0, B);
            float mse1 = 0f, mse2 = 0f;
            for (int i = 0; i < B; i++)
            {
                float d1 = _scrQ1[i] - _scrY[i]; mse1 += d1 * d1;
                float d2 = _scrQ2[i] - _scrY[i]; mse2 += d2 * d2;
            }
            criticLoss += (mse1 + mse2) / (2f * B);

            sacLoss.SetInt("B", B);
            sacLoss.SetBuffer(kCriticGrad, "q1Buf", q1);
            sacLoss.SetBuffer(kCriticGrad, "q2Buf", q2);
            sacLoss.SetBuffer(kCriticGrad, "yBuf", bufY);
            sacLoss.SetBuffer(kCriticGrad, "dQ1Buf", bufDQ1);
            sacLoss.SetBuffer(kCriticGrad, "dQ2Buf", bufDQ2);
            sacLoss.Dispatch(kCriticGrad, Div256(B), 1, 1);

            oQ.ZeroGrad();
            gQ1.Backward(bufDQ1, B);
            gQ2.Backward(bufDQ2, B);
            oQ.Step();
        }

        // -------------------- actor update --------------------
        void UpdatePolicy(int B)
        {
            // a~ = tanh(mu(s) + sigma(s)*ksi), logpi
            ComputeBuffer mu = gMu.Forward(bufStates, B);
            ComputeBuffer sig = gSigma.Forward(bufStates, B);
            DispatchSampleLogPi(B, mu, sig, bufNoise);

            // pair = (s, a~)
            DispatchWritePair(B, bufStates);

            // Q(s,a~) through the freshly-updated critics; weights frozen, input grads only.
            gQ1.RequiresGrad = false;
            gQ2.RequiresGrad = false;
            ComputeBuffer q1 = gQ1.Forward(bufPair, B);
            ComputeBuffer q2 = gQ2.Forward(bufPair, B);

            sacLoss.SetInt("B", B);
            sacLoss.SetFloat("alpha", hp.alpha);
            sacLoss.SetBuffer(kMinQSelect, "q1Buf", q1);
            sacLoss.SetBuffer(kMinQSelect, "q2Buf", q2);
            sacLoss.SetBuffer(kMinQSelect, "logPiBuf", bufLogPi);
            sacLoss.SetBuffer(kMinQSelect, "dQ1Buf", bufDQ1);
            sacLoss.SetBuffer(kMinQSelect, "dQ2Buf", bufDQ2);
            sacLoss.SetBuffer(kMinQSelect, "jBuf", bufJ);
            sacLoss.Dispatch(kMinQSelect, Div256(B), 1, 1);

            // J logging
            bufJ.GetData(_scrJ, 0, 0, B);
            float jSum = 0f;
            for (int i = 0; i < B; i++) jSum += _scrJ[i];
            actorLoss += jSum / B;

            // dQ/d(pair) = q1.Backward(m1) + q2.Backward(1-m1)
            ComputeBuffer d1 = gQ1.Backward(bufDQ1, B);
            CopyVia(d1, bufDPair, B * (S + A));
            ComputeBuffer d2 = gQ2.Backward(bufDQ2, B);
            sacLoss.SetInt("N_", B * (S + A));
            sacLoss.SetBuffer(kAddInto, "dPairBuf", bufDPair);
            sacLoss.SetBuffer(kAddInto, "srcBuf", d2);
            sacLoss.Dispatch(kAddInto, Div256(B * (S + A)), 1, 1);

            gQ1.RequiresGrad = true;
            gQ2.RequiresGrad = true;

            // slice action columns -> dQ/da
            sacLoss.SetInt("B", B);
            sacLoss.SetInt("S_", S);
            sacLoss.SetInt("A_", A);
            sacLoss.SetBuffer(kSliceActGrad, "dPairBuf", bufDPair);
            sacLoss.SetBuffer(kSliceActGrad, "dQdaBuf", bufDQda);
            sacLoss.Dispatch(kSliceActGrad, Div256(B * A), 1, 1);

            // assemble -dJ/dmu, -dJ/dsigma_netout
            sacLoss.SetFloat("alpha", hp.alpha);
            sacLoss.SetFloat("sigma_scale", model.standardDeviationScale);
            sacLoss.SetBuffer(kActorGrad, "sigBuf", sig);
            sacLoss.SetBuffer(kActorGrad, "actBuf", bufAct);
            sacLoss.SetBuffer(kActorGrad, "noiseBuf", bufNoise);
            sacLoss.SetBuffer(kActorGrad, "dQdaBuf", bufDQda);
            sacLoss.SetBuffer(kActorGrad, "dMuBuf", bufDMu);
            sacLoss.SetBuffer(kActorGrad, "dSigBuf", bufDSig);
            sacLoss.Dispatch(kActorGrad, Div256(B * A), 1, 1);

            oMu.ZeroGrad();
            gMu.Backward(bufDMu, B);
            oMu.Step();

            oSigma.ZeroGrad();
            gSigma.Backward(bufDSig, B);
            oSigma.Step();
        }

        // -------------------- target soft update --------------------
        void UpdateTargetNetworks()
        {
            sacLoss.SetFloat("tau_", hp.tau);
            foreach (var (online, target) in polyakPairs)
            {
                sacLoss.SetInt("N_", online.N);
                sacLoss.SetBuffer(kPolyak, "dPairBuf", target.P);
                sacLoss.SetBuffer(kPolyak, "srcBuf", online.P);
                sacLoss.Dispatch(kPolyak, Div256(online.N), 1, 1);
            }
        }

        // -------------------- kernel helpers --------------------
        void DispatchSampleLogPi(int B, ComputeBuffer mu, ComputeBuffer sig, ComputeBuffer noise)
        {
            sacLoss.SetInt("B", B);
            sacLoss.SetInt("A_", A);
            sacLoss.SetFloat("sigma_scale", model.standardDeviationScale);
            sacLoss.SetBuffer(kSampleLogPi, "muBuf", mu);
            sacLoss.SetBuffer(kSampleLogPi, "sigBuf", sig);
            sacLoss.SetBuffer(kSampleLogPi, "noiseBuf", noise);
            sacLoss.SetBuffer(kSampleLogPi, "actBuf", bufAct);
            sacLoss.SetBuffer(kSampleLogPi, "logPiBuf", bufLogPi);
            sacLoss.Dispatch(kSampleLogPi, Div256(B), 1, 1);
        }

        void DispatchWritePair(int B, ComputeBuffer states)
        {
            int total = B * (S + A);
            sacLoss.SetInt("B", B);
            sacLoss.SetInt("S_", S);
            sacLoss.SetInt("A_", A);
            sacLoss.SetBuffer(kWritePair, "stateBuf", states);
            sacLoss.SetBuffer(kWritePair, "actBuf", bufAct);
            sacLoss.SetBuffer(kWritePair, "pairBuf", bufPair);
            sacLoss.Dispatch(kWritePair, Div256(total), 1, 1);
        }

        // dst = src, via the Polyak kernel with tau=1 (avoids reaching into RLBoosterCS bindings).
        void CopyVia(ComputeBuffer src, ComputeBuffer dst, int n)
        {
            sacLoss.SetFloat("tau_", 1f);
            sacLoss.SetInt("N_", n);
            sacLoss.SetBuffer(kPolyak, "dPairBuf", dst);
            sacLoss.SetBuffer(kPolyak, "srcBuf", src);
            sacLoss.Dispatch(kPolyak, Div256(n), 1, 1);
            sacLoss.SetFloat("tau_", hp.tau);
        }

        static int Div256(int n) => (n + 255) / 256;

        // -------------------- save / load --------------------
        // Shares the optimizer-state files with the CPU SACTrainer: order [q1q2, mu, sigma].
        public override string OptimStatesKey => "sactrainer";

        string[] cachedOptimStates;

        protected override string[] SerializeOptimizerStates()
        {
            try
            {
                // Sync GPU weights to the CPU mirrors first so the model save (which follows
                // this call in autosave/quit paths) writes up-to-date networks.
                gQ1.DownloadToCpu(); gQ2.DownloadToCpu(); gMu.DownloadToCpu(); gSigma.DownloadToCpu();
                cachedOptimStates = SerializeLiveOptimStates();
            }
            catch (Exception e)
            {
                if (cachedOptimStates == null)
                {
                    ConsoleMessage.Warning($"[SACGPU] GPU buffers unavailable at save time and no snapshot exists; optimizer states not saved ({e.GetType().Name}).");
                    return Array.Empty<string>();
                }
            }
            return cachedOptimStates;
        }

        string[] SerializeLiveOptimStates()
        {
            return new string[]
            {
                SerializeGpuAdam(model.q1Network.Parameters().Concat(model.q2Network.Parameters()).ToArray(),
                                 gQ1.Parameters().Concat(gQ2.Parameters()).ToArray(), oQ, hp.criticLearningRate),
                SerializeGpuAdam(model.muNetwork.Parameters(), gMu.Parameters().ToArray(), oMu, hp.actorLearningRate),
                SerializeGpuAdam(model.sigmaNetwork.Parameters(), gSigma.Parameters().ToArray(), oSigma, hp.actorLearningRate),
            };
        }

        string SerializeGpuAdam(Parameter[] pars, GPUParam[] gps, GPUAdamW gopt, float lr)
        {
            var adam = new Optimizers.AdamW(pars, lr, beta1: adamBeta1, beta2: adamBeta2, eps: epsilon_adam,
                                            weight_decay: 0f, amsgrad: false, fused: true);
            Tensor[] m = adam.M, v = adam.V;
            int n = Mathf.Min(gps.Length, m.Length);
            for (int i = 0; i < n; i++)
            {
                float[] buf = new float[gps[i].N];
                gps[i].M.GetData(buf);
                Tensor.CopyTo(Tensor.Constant(buf).Reshape(m[i].Shape), m[i]);
                gps[i].V.GetData(buf);
                Tensor.CopyTo(Tensor.Constant(buf).Reshape(v[i].Shape), v[i]);
            }
            adam.SetStepState(gopt.t, gopt.beta1_t, gopt.beta2_t);
            adam.description = "(AdamW) saved by SACGPUTrainer (CPU-compatible)";
            return JsonUtility.ToJson(adam, true);
        }

        static void LoadGpuAdam(string json, GPUParam[] gps, GPUAdamW gopt, string label)
        {
            if (string.IsNullOrEmpty(json)) return;
            var adam = JsonUtility.FromJson<Optimizers.AdamW>(json);
            Tensor[] m = adam.M, v = adam.V;
            if (m == null || v == null || m.Length != gps.Length)
            {
                ConsoleMessage.Warning($"[SACGPU] Optim state '{label}' param-count mismatch ({m?.Length ?? -1} vs {gps.Length}); starting fresh.");
                return;
            }
            for (int i = 0; i < gps.Length; i++)
            {
                if (m[i].Count() != gps[i].N)
                {
                    ConsoleMessage.Warning($"[SACGPU] Optim state '{label}' shape mismatch at param {i}; starting fresh.");
                    return;
                }
            }
            for (int i = 0; i < gps.Length; i++)
            {
                gps[i].M.SetData(m[i].ToArray());
                gps[i].V.SetData(v[i].ToArray());
            }
            var (t, b1t, b2t) = adam.GetStepState();
            gopt.t = t; gopt.beta1_t = b1t; gopt.beta2_t = b2t;
        }

        void OnDestroy()
        {
            try { cachedOptimStates = SerializeLiveOptimStates(); } catch { /* buffers already gone */ }
            try { gQ1.DownloadToCpu(); gQ2.DownloadToCpu(); gMu.DownloadToCpu(); gSigma.DownloadToCpu(); } catch { }

            gQ1?.Dispose(); gQ2?.Dispose(); gQ1T?.Dispose(); gQ2T?.Dispose(); gMu?.Dispose(); gSigma?.Dispose();
            oQ?.Dispose(); oMu?.Dispose(); oSigma?.Dispose();
            bufStates?.Release(); bufNextStates?.Release(); bufPairCritic?.Release(); bufPair?.Release(); bufDPair?.Release();
            bufNoise?.Release(); bufNoisePrime?.Release(); bufAct?.Release(); bufLogPi?.Release();
            bufRew?.Release(); bufDone?.Release(); bufTrunc?.Release(); bufY?.Release();
            bufDQ1?.Release(); bufDQ2?.Release(); bufJ?.Release();
            bufDQda?.Release(); bufDMu?.Release(); bufDSig?.Release();
        }
    }

#if UNITY_EDITOR
    [CustomEditor(typeof(SACGPUTrainer), true), CanEditMultipleObjects]
    sealed class CustomSACGPUTrainerEditor : Editor
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
