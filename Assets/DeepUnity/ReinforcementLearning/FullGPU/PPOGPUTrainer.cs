// PPOGPUTrainer — sibling to PPOTrainer, fully GPU-resident.
//
// Layout mirrors PPOTrainer's lifecycle:
//   Initialize()        - allocate GPUMLPs for v / mu / sigma / discrete networks +
//                         GPUAdamW optimizers + persistent batch buffers.
//   OnBeforeFixedUpdate - same buffer-fill condition as PPO; on full buffer, call Train().
//   Train()             - shuffle on CPU (cheap), upload minibatches to GPU once, run all
//                         epochs/minibatches with no CPU↔GPU bounce except a single scalar
//                         loss readback per minibatch (for logging).
//
// Network architecture must be MLP or LnMLP (Dense / RMSNorm / activation only). If the
// asset has any other module (Conv2D, RNNCell, etc.) we throw a clear error at Initialize.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using DeepUnity.ReinforcementLearning.FullGPU;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    internal sealed class PPOGPUTrainer : DeepUnityTrainer, IOnPolicy
    {
        // ------------ tuned constants (match PPOTrainer) ------------
        const float epsilon_adam = 1e-5f;
        const float valueWD = 0f;
        const float adamBeta1 = 0.9f;
        const float adamBeta2 = 0.999f;

        // ------------ GPU networks ------------
        GPUMLP gV, gMu, gSigma, gDisc;
        GPUAdamW oV, oMu, oSigma, oDisc;

        // ------------ shaders ------------
        ComputeShader booster;
        ComputeShader ppoLoss;
        // booster
        int kCopy;
        // ppoLoss kernels
        int kContSampleAndProb, kContGaussianProb, kContJointProb, kContSurrogateGrad;
        int kSafeSoftmax, kDiscreteSample, kDiscreteSurrogateGrad;
        int kValueMseGrad, kNormalizeAdvantages;

        // ------------ persistent batch buffers ------------
        // sized to max(hp.batchSize, parallelAgents.Count, total dataset for value-prediction)
        int maxB;
        ComputeBuffer bufStates;       // [maxB, S]
        ComputeBuffer bufAdv;          // [maxB]
        ComputeBuffer bufVTargets;     // [maxB]
        ComputeBuffer bufContActions;  // [maxB, A_cont]
        ComputeBuffer bufContPiOld;    // [maxB, A_cont]   per-component old prob
        ComputeBuffer bufContPiNew;    // [maxB, A_cont]
        ComputeBuffer bufContJointNew; // [maxB]
        ComputeBuffer bufNoise;        // [maxB, A_cont] (Gaussian samples for sampling)
        ComputeBuffer bufDMu;          // [maxB, A_cont]
        ComputeBuffer bufDSigma;       // [maxB, A_cont]
        ComputeBuffer bufDiscActions;  // [maxB, A_disc] one-hot
        ComputeBuffer bufDiscPiOld;    // [maxB, A_disc]
        ComputeBuffer bufDiscPiNew;    // [maxB, A_disc]
        ComputeBuffer bufDiscUniform;  // [maxB] uniform[0,1) for Gumbel
        ComputeBuffer bufDLogits;      // [maxB, A_disc]
        ComputeBuffer bufValuesGrad;   // [maxB] dValues for value backward

        // sigma scale & fixed-sigma values (set per-step from model)
        float fixedSigma;
        bool trainableSigma;
        float sigmaScale;

        // LR schedule (LinearAnnealing equivalent of CPU PPOTrainer).
        int totalSchedEpochs;
        int schedEpochsTaken;
        float lrV0, lrA0;

        // Cache so we can match the same stochasticity gating as CPU PPOTrainer
        // (which forces FixedStandardDeviation if not Trainable). FullGPU only supports those two.

        protected override void Initialize(string[] optimizer_states)
        {
            booster = Resources.Load<ComputeShader>("ComputeShaders/RLBoosterCS");
            ppoLoss = Resources.Load<ComputeShader>("ComputeShaders/PPOLossCS");
            if (booster == null || ppoLoss == null)
                throw new Exception("PPOGPU shaders missing (Resources/ComputeShaders/RLBoosterCS or PPOLossCS).");

            kCopy = booster.FindKernel("CopyBuffer");
            kContSampleAndProb = ppoLoss.FindKernel("ContSampleAndProb");
            kContGaussianProb  = ppoLoss.FindKernel("ContGaussianProb");
            kContJointProb     = ppoLoss.FindKernel("ContJointProb");
            kContSurrogateGrad = ppoLoss.FindKernel("ContSurrogateGrad");
            kSafeSoftmax       = ppoLoss.FindKernel("SafeSoftmaxRows");
            kDiscreteSample    = ppoLoss.FindKernel("DiscreteSample");
            kDiscreteSurrogateGrad = ppoLoss.FindKernel("DiscreteSurrogateGrad");
            kValueMseGrad      = ppoLoss.FindKernel("ValueMseGrad");
            kNormalizeAdvantages = ppoLoss.FindKernel("NormalizeAdvantages");

            // Diagnostic: -1 means FindKernel failed (kernel didn't compile or doesn't exist).
            ConsoleMessage.Info($"[PPOGPU][DIAG kernels] ContGaussianProb={kContGaussianProb} ContSurrogateGrad={kContSurrogateGrad} " +
                                $"SafeSoftmax={kSafeSoftmax} DiscSurrogateGrad={kDiscreteSurrogateGrad} " +
                                $"ValueMseGrad={kValueMseGrad} NormAdv={kNormalizeAdvantages} ContSampleAndProb={kContSampleAndProb} " +
                                $"DiscSample={kDiscreteSample} ContJointProb={kContJointProb}  (any -1 = kernel compile failure)");

            // Validate architecture & strip optional Tanh tail of mu (matches CPU PPOTrainer).
            if (model.muNetwork != null && model.muNetwork.Modules.Last() is Tanh)
                model.muNetwork.Modules = model.muNetwork.Modules.Take(model.muNetwork.Modules.Length - 1).ToArray();

            // FullGPU supports only the two PPO-native stochasticity modes.
            if (model.stochasticity != Stochasticity.FixedStandardDeviation &&
                model.stochasticity != Stochasticity.TrainableStandardDeviation)
                model.stochasticity = Stochasticity.FixedStandardDeviation;

            ValidateMlpOnly(model.vNetwork, "Value");
            if (model.muNetwork != null) ValidateMlpOnly(model.muNetwork, "Mu");
            if (model.sigmaNetwork != null) ValidateMlpOnly(model.sigmaNetwork, "Sigma");
            if (model.discreteNetwork != null) ValidateMlpOnly(model.discreteNetwork, "Discrete");

            // GPU buffers only need to hold one minibatch — GAE/Vtarget computation goes
            // through CPU Sequential.Predict and doesn't touch our GPU tape.
            maxB = hp.batchSize;

            gV = new GPUMLP(model.vNetwork, maxB);
            oV = new GPUAdamW(gV.Parameters(), hp.criticLearningRate, adamBeta1, adamBeta2, epsilon_adam, valueWD, hp.maxNorm);
            if (model.muNetwork != null)
            {
                gMu = new GPUMLP(model.muNetwork, maxB);
                // Mu intentionally unclipped to match CPU PPOTrainer (line 479: ClipGradNorm is commented out).
                oMu = new GPUAdamW(gMu.Parameters(), hp.actorLearningRate, adamBeta1, adamBeta2, epsilon_adam, 0f, -1f);
                if (model.stochasticity == Stochasticity.TrainableStandardDeviation && model.sigmaNetwork != null)
                {
                    gSigma = new GPUMLP(model.sigmaNetwork, maxB);
                    oSigma = new GPUAdamW(gSigma.Parameters(), hp.actorLearningRate, adamBeta1, adamBeta2, epsilon_adam, 0f, hp.maxNorm);
                }
            }
            if (model.discreteNetwork != null)
            {
                gDisc = new GPUMLP(model.discreteNetwork, maxB);
                oDisc = new GPUAdamW(gDisc.Parameters(), hp.actorLearningRate, adamBeta1, adamBeta2, epsilon_adam, 0f, hp.maxNorm);
            }

            // LinearAnnealing schedule: lr = initial * max(0, 1 - epochs_taken / total_epochs).
            // Mirrors CPU PPOTrainer's LinearAnnealing(start=1, end=0, total_iters=total_epochs).
            totalSchedEpochs = Mathf.Max(1, (int)hp.maxSteps / hp.bufferSize * hp.numEpoch);
            schedEpochsTaken = 0;
            lrV0 = hp.criticLearningRate;
            lrA0 = hp.actorLearningRate;

            // Allocate persistent batch buffers
            int S = model.observationSize * model.stackedInputs;
            bufStates    = New(maxB * S);
            bufAdv       = New(maxB);
            bufVTargets  = New(maxB);
            bufValuesGrad= New(maxB);
            if (model.IsUsingContinuousActions)
            {
                int Ac = model.continuousDim;
                bufContActions  = New(maxB * Ac);
                bufContPiOld    = New(maxB * Ac);
                bufContPiNew    = New(maxB * Ac);
                bufContJointNew = New(maxB);
                bufNoise        = New(maxB * Ac);
                bufDMu          = New(maxB * Ac);
                bufDSigma       = New(maxB * Ac);
            }
            if (model.IsUsingDiscreteActions)
            {
                int Ad = model.discreteDim;
                bufDiscActions = New(maxB * Ad);
                bufDiscPiOld   = New(maxB * Ad);
                bufDiscPiNew   = New(maxB * Ad);
                bufDiscUniform = New(maxB);
                bufDLogits     = New(maxB * Ad);
            }

            fixedSigma     = model.standardDeviationValue;
            trainableSigma = model.stochasticity == Stochasticity.TrainableStandardDeviation;
            sigmaScale     = model.standardDeviationScale;

            // Network device tags don't really matter for FullGPU but we set them for the editor:
            if (model.muNetwork != null) model.muNetwork.Device = Device.GPU;
            if (model.sigmaNetwork != null) model.sigmaNetwork.Device = Device.GPU;
            if (model.discreteNetwork != null) model.discreteNetwork.Device = Device.GPU;

            // Optimizer states from disk: out of scope for v1 (CPU-format adam states aren't
            // reusable for the GPU buffers). On first run, momentum/variance start at zero —
            // identical to a fresh AdamW.
            ConsoleMessage.Info("[PPOGPU] FullGPU PPO initialized.");
        }

        static ComputeBuffer New(int n) => new ComputeBuffer(n, 4, ComputeBufferType.Structured);

        static void ValidateMlpOnly(Sequential net, string label)
        {
            foreach (var m in net.Modules)
            {
                if (m is Dense || m is RMSNorm || m is ReLU || m is Tanh || m is SiLU || m is Softplus || m is Softmax)
                    continue;
                throw new Exception($"PPOGPU: network '{label}' contains '{m.GetType().Name}' which is not supported by FullGPU. " +
                                    "Use ArchitectureType.MLP or ArchitectureType.LnMLP, or switch trainer to PPO.");
            }
        }

        protected override void OnBeforeFixedUpdate()
        {
            if (MemoriesCount < hp.bufferSize) return;

            // GAE & Vtargets — runs on CPU (uses Sequential.Predict which still uses the CPU
            // path). It's per-trajectory and only happens once per buffer-fill, so the cost
            // is negligible vs the train loop.
            // First sync GPU weights back to CPU so Predict reflects the latest training.
            DownloadAllToCpu();

            foreach (var agent_memory in parallelAgents.Select(x => x.Memory))
            {
                if (agent_memory.Count == 0) continue;
                ComputeGAE_andVtargets(agent_memory, hp.gamma, hp.lambda, hp.horizon, model.vNetwork);
                train_data.TryAppend(agent_memory.frames, hp.bufferSize);
                agent_memory.Clear();
            }

            actorLoss = 0; criticLoss = 0; entropy = 0;
            updateBenchmarkClock = Stopwatch.StartNew();
            Train();
            updateBenchmarkClock.Stop();
            updateIterations++;
            int nMinibatches = Mathf.Max(1, hp.bufferSize / hp.batchSize) * hp.numEpoch;
            actorLoss /= nMinibatches; criticLoss /= nMinibatches; entropy /= nMinibatches;
            currentSteps += hp.bufferSize;

            // Push GPU weights back to CPU so any subsequent CPU inference reads up-to-date params.
            DownloadAllToCpu();
        }

        void DownloadAllToCpu()
        {
            gV.DownloadToCpu();
            gMu?.DownloadToCpu();
            gSigma?.DownloadToCpu();
            gDisc?.DownloadToCpu();
        }

        // Diagnostic snapshot of first 4 weights of each network's first parameter buffer
        // (lets us verify the optimizer is actually moving weights).
        float[] _diagBeforeV = new float[4];
        float[] _diagBeforeMu = new float[4];
        bool _diagDMuLogged = false;

        void Train()
        {
            int N = train_data.Count;
            int batch = hp.batchSize;
            int S = model.observationSize * model.stackedInputs;
            int Ac = model.continuousDim;
            int Ad = model.discreteDim;
            int mbCount = 0;
            var sw = System.Diagnostics.Stopwatch.StartNew();
            ConsoleMessage.Info($"[PPOGPU] Train start  buffer={N}  batch={batch}  epochs={hp.numEpoch}");

            // [DIAG-1] Snapshot first 4 weights of v and mu networks.
            var vFirst = gV.Parameters().FirstOrDefault();
            var muFirst = gMu?.Parameters().FirstOrDefault();
            vFirst?.P.GetData(_diagBeforeV, 0, 0, 4);
            muFirst?.P.GetData(_diagBeforeMu, 0, 0, 4);
            _diagDMuLogged = false;

            // [DIAG-3] CPU vs GPU forward parity check on the first batch element.
            //  - run CPU model.vNetwork.Predict on the first state in the buffer
            //  - read GPU vOut[0] after first UpdateValue forward
            //  - compare. They should match to within numerical noise.
            bool didParityCheck = false;

            for (int epoch = 0; epoch < hp.numEpoch; epoch++)
            {
                if (batch != hp.bufferSize) train_data.Shuffle();

                for (int start = 0; start + batch <= N; start += batch)
                {
                    // ---- Stage states / advantages / vtargets / actions / oldprobs into GPU ----
                    UploadBatch(start, batch, S, Ac, Ad);

                    // ---- Optionally normalize advantages on GPU ----
                    if (hp.normalizeAdvantages)
                    {
                        ppoLoss.SetInt("B", batch);
                        ppoLoss.SetBuffer(kNormalizeAdvantages, "advantages", bufAdv);
                        ppoLoss.Dispatch(kNormalizeAdvantages, 1, 1, 1);
                    }

                    // ---- Value head update ----
                    UpdateValue(batch, S);

                    // [DIAG-3] After first UpdateValue (forward done, weights still pre-step
                    // because vOut readback in UpdateValue happens immediately after Forward).
                    if (!didParityCheck)
                    {
                        didParityCheck = true;
                        try
                        {
                            var firstFrame = train_data.frames[start];
                            float gpuV = _scratchV != null ? _scratchV[0] : float.NaN;
                            float cpuV = model.vNetwork.Predict(firstFrame.state).ToArray()[0];
                            float diff = Mathf.Abs(gpuV - cpuV);
                            ConsoleMessage.Info($"[PPOGPU][DIAG fwd] cpuV={cpuV:0.000000}  gpuV={gpuV:0.000000}  |diff|={diff:0.000000}  (>1e-3 = bug)");
                        }
                        catch (System.Exception ex) { ConsoleMessage.Warning($"[PPOGPU][DIAG fwd] readback failed: {ex.Message}"); }
                    }

                    // ---- Continuous head update ----
                    if (model.IsUsingContinuousActions) UpdateContinuous(batch, S, Ac);

                    // ---- Discrete head update ----
                    if (model.IsUsingDiscreteActions) UpdateDiscrete(batch, S, Ad);

                    // [DIAG-2] Once per Train (only for first minibatch): grad-norm of v / mu.
                    if (mbCount == 0)
                    {
                        float gnV = ComputeGradNorm(gV);
                        float gnMu = gMu != null ? ComputeGradNorm(gMu) : 0f;
                        ConsoleMessage.Info($"[PPOGPU][DIAG grad] ||grad_v||={gnV:0.000000}  ||grad_mu||={gnMu:0.000000}  (zero = bug)");
                    }

                    mbCount++;
                }

                // LinearAnnealing per-epoch step (matches CPU PPOTrainer when hp.LRSchedule is on).
                if (hp.LRSchedule)
                {
                    schedEpochsTaken++;
                    float factor = Mathf.Max(0f, 1f - (float)schedEpochsTaken / (float)totalSchedEpochs);
                    oV.lr = lrV0 * factor;
                    if (oMu != null) oMu.lr = lrA0 * factor;
                    if (oSigma != null) oSigma.lr = lrA0 * factor;
                    if (oDisc != null) oDisc.lr = lrA0 * factor;
                }
            }
            sw.Stop();
            int divisor = Mathf.Max(1, mbCount);
            ConsoleMessage.Info($"[PPOGPU] Train done  minibatches={mbCount}  elapsed={sw.Elapsed.TotalMilliseconds:0.0}ms  " +
                                $"avg_critic={(criticLoss / divisor):0.0000}  avg_actor={(actorLoss / divisor):0.0000}  avg_ent={(entropy / divisor):0.0000}");

            // [DIAG-1] Post-train weight delta — confirms optimizer is actually moving the parameters.
            float[] afterV = new float[4]; float[] afterMu = new float[4];
            float dV = 0f, dMu = 0f;
            if (vFirst != null) { vFirst.P.GetData(afterV, 0, 0, 4); for (int i = 0; i < 4; i++) dV += Mathf.Abs(afterV[i] - _diagBeforeV[i]); }
            if (muFirst != null) { muFirst.P.GetData(afterMu, 0, 0, 4); for (int i = 0; i < 4; i++) dMu += Mathf.Abs(afterMu[i] - _diagBeforeMu[i]); }
            ConsoleMessage.Info($"[PPOGPU][DIAG step] |Δw_v|={dV:0.0000000}  |Δw_mu|={dMu:0.0000000}  (zero = no update)");

            train_data.Clear();
        }

        // [DIAG-2] Sum of squares of all gradient buffers, sqrt'd. Done on CPU readback (cheap once per Train).
        static float ComputeGradNorm(GPUMLP net)
        {
            double sum = 0;
            foreach (var p in net.Parameters())
            {
                float[] g = new float[p.N];
                p.G.GetData(g);
                for (int i = 0; i < g.Length; i++) sum += (double)g[i] * g[i];
            }
            return Mathf.Sqrt((float)sum);
        }

        // -------------------- value head --------------------
        // scratch arrays for diagnostic readback
        float[] _scratchV; float[] _scratchVT;
        float[] _scratchPi; float[] _scratchPiOld;
        void UpdateValue(int batch, int S)
        {
            ComputeBuffer vOut = gV.Forward(bufStates, batch);   // [batch, 1]

            // Diagnostic: read V and Vt to compute MSE for the runtime UI / loss display.
            // One readback per minibatch, scalar-cost.
            if (_scratchV == null || _scratchV.Length < batch) { _scratchV = new float[batch]; _scratchVT = new float[batch]; }
            vOut.GetData(_scratchV, 0, 0, batch);
            bufVTargets.GetData(_scratchVT, 0, 0, batch);
            float mseSum = 0f;
            for (int i = 0; i < batch; i++) { float d = _scratchV[i] - _scratchVT[i]; mseSum += d * d; }
            criticLoss += (mseSum / batch) * hp.valueCoeff;

            // dV = (V - Vt) * (2/B) * value_coeff
            ppoLoss.SetInt("B", batch);
            ppoLoss.SetFloat("value_coeff", hp.valueCoeff);
            ppoLoss.SetBuffer(kValueMseGrad, "values", vOut);
            ppoLoss.SetBuffer(kValueMseGrad, "valueTargets", bufVTargets);
            ppoLoss.SetBuffer(kValueMseGrad, "dValues", bufValuesGrad);
            ppoLoss.Dispatch(kValueMseGrad, (batch + 255) / 256, 1, 1);

            oV.ZeroGrad();
            gV.Backward(bufValuesGrad, batch);
            oV.Step();
        }

        // -------------------- continuous head --------------------
        void UpdateContinuous(int batch, int S, int Ac)
        {
            // Forward mu (= [batch, Ac]) and optional sigma
            ComputeBuffer muOut = gMu.Forward(bufStates, batch);
            ComputeBuffer sigOut = trainableSigma ? gSigma.Forward(bufStates, batch) : null;

            // piNew[b,a] = N(actions[b,a]; mu, sigma_eff)  (per-component prob, matches CPU Tensor.Probability)
            ppoLoss.SetInt("B", batch);
            ppoLoss.SetInt("A", Ac);
            ppoLoss.SetInt("trainable_sigma", trainableSigma ? 1 : 0);
            ppoLoss.SetInt("sigma_scale_enable", trainableSigma ? 1 : 0);
            ppoLoss.SetFloat("sigma_scale", sigmaScale);
            ppoLoss.SetFloat("fixed_sigma", fixedSigma);
            ppoLoss.SetBuffer(kContGaussianProb, "mu", muOut);
            if (trainableSigma) ppoLoss.SetBuffer(kContGaussianProb, "sigma", sigOut);
            else ppoLoss.SetBuffer(kContGaussianProb, "sigma", muOut); // dummy bind
            ppoLoss.SetBuffer(kContGaussianProb, "actions", bufContActions);
            ppoLoss.SetBuffer(kContGaussianProb, "piNew", bufContPiNew);
            ppoLoss.Dispatch(kContGaussianProb, (batch * Ac + 255) / 256, 1, 1);

            // [DIAG-5] Pre-fill bufDMu with sentinel 7777 so we can tell if ContSurrogateGrad
            // actually ran (kernel didn't run -> 7777 stays; kernel ran but produced zero -> 0).
            if (_diagDMuLogged == false)
            {
                int totSent = batch * Ac;
                float[] sentinel = new float[totSent];
                for (int i = 0; i < totSent; i++) sentinel[i] = 7777.0f;
                bufDMu.SetData(sentinel);
            }

            // Surrogate gradient → dMu, dSigma
            ppoLoss.SetFloat("clip_eps", hp.epsilon);
            ppoLoss.SetFloat("beta_entropy", hp.beta);
            ppoLoss.SetBuffer(kContSurrogateGrad, "mu", muOut);
            if (trainableSigma) ppoLoss.SetBuffer(kContSurrogateGrad, "sigma", sigOut);
            else ppoLoss.SetBuffer(kContSurrogateGrad, "sigma", muOut);
            ppoLoss.SetBuffer(kContSurrogateGrad, "actions", bufContActions);
            ppoLoss.SetBuffer(kContSurrogateGrad, "piNew", bufContPiNew);
            ppoLoss.SetBuffer(kContSurrogateGrad, "piOld", bufContPiOld);
            ppoLoss.SetBuffer(kContSurrogateGrad, "advantages", bufAdv);
            ppoLoss.SetBuffer(kContSurrogateGrad, "dMu", bufDMu);
            if (trainableSigma) ppoLoss.SetBuffer(kContSurrogateGrad, "dSigma", bufDSigma);
            else ppoLoss.SetBuffer(kContSurrogateGrad, "dSigma", bufDMu); // dummy bind
            ppoLoss.Dispatch(kContSurrogateGrad, (batch * Ac + 255) / 256, 1, 1);

            // [DIAG-4] Read bufDMu after ContSurrogateGrad to isolate where the zero is.
            // If this is zero, the kernel produced no gradient. If non-zero, gMu.Backward is the culprit.
            if (_diagDMuLogged == false)
            {
                _diagDMuLogged = true;
                int totalDiag = batch * Ac;
                float[] dmu = new float[totalDiag];
                bufDMu.GetData(dmu, 0, 0, totalDiag);
                double sumSq = 0; double sumAbs = 0; float maxAbs = 0;
                for (int i = 0; i < totalDiag; i++) { sumSq += (double)dmu[i] * dmu[i]; float a = Mathf.Abs(dmu[i]); sumAbs += a; if (a > maxAbs) maxAbs = a; }
                ConsoleMessage.Info($"[PPOGPU][DIAG dMu] ||dMu||={Mathf.Sqrt((float)sumSq):0.000000}  mean|dMu|={(sumAbs / totalDiag):0.000000}  max|dMu|={maxAbs:0.000000}  (zero = ContSurrogateGrad bug)");

                // Also sample piNew, piOld, advantages to verify rollout-time data is sane.
                float[] piN = new float[totalDiag]; bufContPiNew.GetData(piN, 0, 0, totalDiag);
                float[] piO = new float[totalDiag]; bufContPiOld.GetData(piO, 0, 0, totalDiag);
                float[] adv = new float[batch]; bufAdv.GetData(adv, 0, 0, batch);
                float[] act = new float[totalDiag]; bufContActions.GetData(act, 0, 0, totalDiag);
                float[] muF = new float[totalDiag]; muOut.GetData(muF, 0, 0, totalDiag);
                ConsoleMessage.Info($"[PPOGPU][DIAG inp] piNew[0..3]=[{piN[0]:0.0000},{piN[1]:0.0000},{piN[2]:0.0000}]  piOld[0..3]=[{piO[0]:0.0000},{piO[1]:0.0000},{piO[2]:0.0000}]");
                ConsoleMessage.Info($"[PPOGPU][DIAG inp] adv[0..3]=[{adv[0]:0.0000},{adv[1]:0.0000},{adv[2]:0.0000}]  act[0..3]=[{act[0]:0.0000},{act[1]:0.0000},{act[2]:0.0000}]  mu[0..3]=[{muF[0]:0.0000},{muF[1]:0.0000},{muF[2]:0.0000}]");
            }

            // Diagnostic: read piNew & piOld to compute |LCLIP| for runtime UI.
            int total = batch * Ac;
            if (_scratchPi == null || _scratchPi.Length < total) { _scratchPi = new float[total]; _scratchPiOld = new float[total]; }
            bufContPiNew.GetData(_scratchPi, 0, 0, total);
            bufContPiOld.GetData(_scratchPiOld, 0, 0, total);
            float surrSum = 0f;
            float entSum = 0f;
            float invB = 1f / batch;
            for (int i = 0; i < total; i++)
            {
                float pNew = _scratchPi[i]; float pOld = _scratchPiOld[i];
                if (pOld < 1e-12f) pOld = 1e-12f;
                float r = pNew / pOld;
                float clipR = Mathf.Clamp(r, 1f - hp.epsilon, 1f + hp.epsilon);
                int b = i / Ac;
                // advantages was uploaded as bufAdv; reuse _scratchVT slot for adv readback would require another buffer.
                // Approximation: use absolute |min(r,clipR)| averaged. (PPO loss is min(r*A, clip*A); without A here we
                // log the surrogate-ratio magnitude as a coarse signal.)
                float lc = Mathf.Min(r, clipR);
                surrSum += Mathf.Abs(lc);
            }
            actorLoss += surrSum / total;
            entropy += fixedSigma; // matches CPU PPO's "entropy += sigma.Average()" for FixedStdDev

            // Mu backward + step
            oMu.ZeroGrad();
            gMu.Backward(bufDMu, batch);
            oMu.Step();

            // Sigma backward + step (if trainable)
            if (trainableSigma)
            {
                oSigma.ZeroGrad();
                gSigma.Backward(bufDSigma, batch);
                oSigma.Step();
            }
        }

        // -------------------- discrete head --------------------
        void UpdateDiscrete(int batch, int S, int Ad)
        {
            // Forward logits (Softmax tail in the Sequential is recognized as a no-op pass-through).
            ComputeBuffer logits = gDisc.Forward(bufStates, batch);  // [batch, Ad] = logits

            // Compute phi = safe softmax(logits) -> bufDiscPiNew
            ppoLoss.SetInt("B", batch);
            ppoLoss.SetInt("A", Ad);
            ppoLoss.SetBuffer(kSafeSoftmax, "mu", logits);          // bind logits as `mu` in the kernel (logit slot)
            ppoLoss.SetBuffer(kSafeSoftmax, "piNew", bufDiscPiNew);
            ppoLoss.Dispatch(kSafeSoftmax, batch, 1, 1);

            // Surrogate grad → dLogits  (writes into bufDLogits)
            ppoLoss.SetFloat("clip_eps", hp.epsilon);
            ppoLoss.SetFloat("beta_entropy", hp.beta);
            ppoLoss.SetBuffer(kDiscreteSurrogateGrad, "piNew", bufDiscPiNew);
            ppoLoss.SetBuffer(kDiscreteSurrogateGrad, "piOld", bufDiscPiOld);
            ppoLoss.SetBuffer(kDiscreteSurrogateGrad, "advantages", bufAdv);
            ppoLoss.SetBuffer(kDiscreteSurrogateGrad, "actions", bufDiscActions);
            ppoLoss.SetBuffer(kDiscreteSurrogateGrad, "dLogits", bufDLogits);
            ppoLoss.Dispatch(kDiscreteSurrogateGrad, batch, 1, 1);

            oDisc.ZeroGrad();
            gDisc.Backward(bufDLogits, batch);
            oDisc.Step();
        }

        // -------------------- batch upload --------------------
        void UploadBatch(int start, int batch, int S, int Ac, int Ad)
        {
            float[] s   = new float[batch * S];
            float[] adv = new float[batch];
            float[] vt  = new float[batch];
            float[] ca  = null, cp = null;
            float[] da  = null, dp = null;
            if (Ac > 0) { ca = new float[batch * Ac]; cp = new float[batch * Ac]; }
            if (Ad > 0) { da = new float[batch * Ad]; dp = new float[batch * Ad]; }

            for (int i = 0; i < batch; i++)
            {
                var f = train_data.frames[start + i];
                Buffer.BlockCopy(f.state.ToArray(), 0, s, i * S * 4, S * 4);
                adv[i] = f.advantage[0];
                vt[i]  = f.v_target[0];
                if (Ac > 0)
                {
                    Buffer.BlockCopy(f.action_continuous.ToArray(), 0, ca, i * Ac * 4, Ac * 4);
                    Buffer.BlockCopy(f.prob_continuous.ToArray(), 0, cp, i * Ac * 4, Ac * 4);
                }
                if (Ad > 0)
                {
                    Buffer.BlockCopy(f.action_discrete.ToArray(), 0, da, i * Ad * 4, Ad * 4);
                    Buffer.BlockCopy(f.prob_discrete.ToArray(), 0, dp, i * Ad * 4, Ad * 4);
                }
            }
            bufStates.SetData(s);
            bufAdv.SetData(adv);
            bufVTargets.SetData(vt);
            if (Ac > 0) { bufContActions.SetData(ca); bufContPiOld.SetData(cp); }
            if (Ad > 0) { bufDiscActions.SetData(da); bufDiscPiOld.SetData(dp); }
        }

        // -------------------- GAE (CPU helper, identical to PPOTrainer) --------------------
        static void ComputeGAE_andVtargets(in MemoryBuffer memory, float GAMMA, float LAMBDA, int HORIZON, Sequential valueNetwork)
        {
            var frames = memory.frames;
            int T = memory.Count;
            Tensor[] all = new Tensor[T + 1];
            for (int i = 0; i < T; i++) all[i] = frames[i].state;
            all[T] = frames[T - 1].nextState;

            float[] Vw_s = valueNetwork.Predict(Tensor.Concat(null, all)).ToArray();

            float gae = 0f;
            for (int t = T - 1; t >= 0; t--)
            {
                float r_t = frames[t].reward[0];
                float V_st = Vw_s[t];
                float done = frames[t].done[0];
                float V_next = done == 1 ? 0 : Vw_s[t + 1];
                float delta = r_t + GAMMA * V_next * (1f - done) - V_st;
                gae = delta + GAMMA * LAMBDA * (1f - done) * gae;
                frames[t].advantage = Tensor.Constant(gae);
                frames[t].v_target = Tensor.Constant(gae + V_st);
            }
        }

        protected override string[] SerializeOptimizerStates()
        {
            // GPU adam state isn't json-friendly. v1: skip; first session-restart starts fresh.
            return Array.Empty<string>();
        }

        // ============================================================
        // Inference path: batched ContinuousEval / DiscreteEval over the
        // parallelAgents states, fully on GPU. The callsite is
        // DeepUnityTrainer.ParallelInference which currently calls
        // model.ContinuousEval(stateBatch, ...). For FullGPU we override at
        // the trainer level by intercepting via OnBeforeFixedUpdate? Simpler:
        // PPOGPUTrainer leaves CPU inference in place — `model.muNetwork.Predict`
        // still works because we DownloadAllToCpu() after every Train() call,
        // and CPU predict on small MLP/LnMLP is already fast. If we need pure
        // GPU inference later, override ParallelInference on PPOGPUTrainer.
        // ============================================================

        void OnDestroy()
        {
            gV?.Dispose(); gMu?.Dispose(); gSigma?.Dispose(); gDisc?.Dispose();
            oV?.Dispose(); oMu?.Dispose(); oSigma?.Dispose(); oDisc?.Dispose();
            bufStates?.Release(); bufAdv?.Release(); bufVTargets?.Release(); bufValuesGrad?.Release();
            bufContActions?.Release(); bufContPiOld?.Release(); bufContPiNew?.Release(); bufContJointNew?.Release();
            bufNoise?.Release(); bufDMu?.Release(); bufDSigma?.Release();
            bufDiscActions?.Release(); bufDiscPiOld?.Release(); bufDiscPiNew?.Release(); bufDiscUniform?.Release(); bufDLogits?.Release();
        }
    }
}
