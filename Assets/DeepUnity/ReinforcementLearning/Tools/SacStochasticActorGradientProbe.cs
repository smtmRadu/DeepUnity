using DeepUnity.Modules;
using DeepUnity.Models;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Numerically validates the full stochastic SAC actor objective
    /// J = min(Q1, Q2)(s, tanh(mu + sigma * eps)) - alpha * log pi(a|s)
    /// against the manual derivative path used by SACTrainerDepr.
    /// </summary>
    public sealed class SacStochasticActorGradientProbe : MonoBehaviour
    {
        [SerializeField] private AgentBehaviour behaviour;
        [SerializeField] private Device probeDevice = Device.CPU;
        [SerializeField, Min(1)] private int samples = 3;
        [SerializeField, Min(1)] private int checkedMuParameters = 8;
        [SerializeField, Min(1)] private int checkedSigmaParameters = 8;
        [SerializeField, Min(1e-6f)] private float epsilon = 1e-3f;
        [SerializeField, Min(1e-4f)] private float stateRange = 0.5f;

        [Button(nameof(RunGradientCheck))]
        [SerializeField] private bool runGradientCheckButton;

        public void RunGradientCheck()
        {
            if (behaviour == null)
            {
                Debug.LogError("[SacStochasticActorGradientProbe] Behaviour is not assigned.");
                return;
            }

            RunForBehaviour(behaviour, probeDevice, samples, checkedMuParameters, checkedSigmaParameters, epsilon, stateRange);
        }

        public static string RunForBehaviour(
            AgentBehaviour behaviour,
            Device probeDevice,
            int samples = 3,
            int checkedMuParameters = 8,
            int checkedSigmaParameters = 8,
            float epsilon = 1e-3f,
            float stateRange = 0.5f)
        {
            if (behaviour == null)
                throw new ArgumentNullException(nameof(behaviour));
            if (behaviour.config == null)
                throw new InvalidOperationException("Behaviour is missing config.");
            if (behaviour.muNetwork == null || behaviour.sigmaNetwork == null || behaviour.q1Network == null || behaviour.q2Network == null)
                throw new InvalidOperationException("Behaviour is missing one or more SAC networks.");
            if (behaviour.continuousDim <= 0)
                throw new InvalidOperationException("Behaviour does not use continuous actions.");

            int stateSize = behaviour.observationSize * behaviour.stackedInputs;
            int actionSize = behaviour.continuousDim;

            Sequential muNetwork = behaviour.muNetwork;
            Sequential sigmaNetwork = behaviour.sigmaNetwork;
            Sequential q1Network = behaviour.q1Network;
            Sequential q2Network = behaviour.q2Network;

            ValidateNetwork(muNetwork, nameof(behaviour.muNetwork));
            ValidateNetwork(sigmaNetwork, nameof(behaviour.sigmaNetwork));
            ValidateNetwork(q1Network, nameof(behaviour.q1Network));
            ValidateNetwork(q2Network, nameof(behaviour.q2Network));

            ILearnable[] muLearnables = muNetwork.Modules.OfType<ILearnable>().ToArray();
            ILearnable[] sigmaLearnables = sigmaNetwork.Modules.OfType<ILearnable>().ToArray();
            ILearnable[] q1Learnables = q1Network.Modules.OfType<ILearnable>().ToArray();
            ILearnable[] q2Learnables = q2Network.Modules.OfType<ILearnable>().ToArray();

            Device[] muDevices = muLearnables.Select(x => x.Device).ToArray();
            Device[] sigmaDevices = sigmaLearnables.Select(x => x.Device).ToArray();
            Device[] q1Devices = q1Learnables.Select(x => x.Device).ToArray();
            Device[] q2Devices = q2Learnables.Select(x => x.Device).ToArray();

            bool[] muReqGrad = muLearnables.Select(x => x.RequiresGrad).ToArray();
            bool[] sigmaReqGrad = sigmaLearnables.Select(x => x.RequiresGrad).ToArray();
            bool[] q1ReqGrad = q1Learnables.Select(x => x.RequiresGrad).ToArray();
            bool[] q2ReqGrad = q2Learnables.Select(x => x.RequiresGrad).ToArray();

            try
            {
                foreach (var item in muLearnables)
                {
                    item.Device = probeDevice;
                    item.RequiresGrad = true;
                }
                foreach (var item in sigmaLearnables)
                {
                    item.Device = probeDevice;
                    item.RequiresGrad = true;
                }
                foreach (var item in q1Learnables)
                {
                    item.Device = probeDevice;
                    item.RequiresGrad = false;
                }
                foreach (var item in q2Learnables)
                {
                    item.Device = probeDevice;
                    item.RequiresGrad = false;
                }

                Parameter[] muParams = muNetwork.Parameters();
                Parameter[] sigmaParams = sigmaNetwork.Parameters();
                List<(Tensor tensor, int index, string label)> muSelected = SelectParameters(muParams, checkedMuParameters, "mu");
                List<(Tensor tensor, int index, string label)> sigmaSelected = SelectParameters(sigmaParams, checkedSigmaParameters, "sigma");

                if (muSelected.Count == 0 && sigmaSelected.Count == 0)
                    throw new InvalidOperationException("No actor parameters available to probe.");

                float totalOutputMuMeanAbsError = 0f;
                float totalOutputMuMaxAbsError = 0f;
                float totalOutputMuCosine = 0f;

                float totalOutputSigmaMeanAbsError = 0f;
                float totalOutputSigmaMaxAbsError = 0f;
                float totalOutputSigmaCosine = 0f;

                float totalParamMuMeanAbsError = 0f;
                float totalParamMuMaxAbsError = 0f;
                float totalParamMuCosine = 0f;

                float totalParamSigmaMeanAbsError = 0f;
                float totalParamSigmaMaxAbsError = 0f;
                float totalParamSigmaCosine = 0f;

                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"[SacStochasticActorGradientProbe] full SAC actor objective on {probeDevice}");
                sb.AppendLine(
                    $"samples={samples}, epsilon={epsilon}, checkedMuParameters={muSelected.Count}, " +
                    $"checkedSigmaParameters={sigmaSelected.Count}, stateSize={stateSize}, actionSize={actionSize}, alpha={behaviour.config.alpha:0.0000}");

                for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
                {
                    ZeroGradients(muParams);
                    ZeroGradients(sigmaParams);

                    Tensor state = Tensor.RandomRange((-stateRange, stateRange), 1, stateSize);
                    Tensor ksi = Tensor.RandomNormal(1, actionSize);

                    SampleComputation analytic = ComputeAnalyticSample(behaviour, muNetwork, sigmaNetwork, q1Network, q2Network, state, ksi);

                    float[] numericOutputMu = new float[actionSize];
                    float[] numericOutputSigmaPreClip = new float[actionSize];
                    float[] outputMuAbsErr = new float[actionSize];
                    float[] outputSigmaAbsErr = new float[actionSize];

                    for (int dim = 0; dim < actionSize; dim++)
                    {
                        Tensor muPlus = analytic.mu.Clone() as Tensor;
                        Tensor muMinus = analytic.mu.Clone() as Tensor;
                        muPlus[0, dim] += epsilon;
                        muMinus[0, dim] -= epsilon;

                        float jPlusMu = EvaluateObjectiveFromOutputs(behaviour, q1Network, q2Network, state, muPlus, analytic.sigmaPreClip, ksi);
                        float jMinusMu = EvaluateObjectiveFromOutputs(behaviour, q1Network, q2Network, state, muMinus, analytic.sigmaPreClip, ksi);
                        numericOutputMu[dim] = (jPlusMu - jMinusMu) / (2f * epsilon);
                        outputMuAbsErr[dim] = MathF.Abs(analytic.dJdMu[0, dim] - numericOutputMu[dim]);

                        Tensor sigmaPrePlus = analytic.sigmaPreClip.Clone() as Tensor;
                        Tensor sigmaPreMinus = analytic.sigmaPreClip.Clone() as Tensor;
                        sigmaPrePlus[0, dim] += epsilon;
                        sigmaPreMinus[0, dim] -= epsilon;

                        float jPlusSigma = EvaluateObjectiveFromOutputs(behaviour, q1Network, q2Network, state, analytic.mu, sigmaPrePlus, ksi);
                        float jMinusSigma = EvaluateObjectiveFromOutputs(behaviour, q1Network, q2Network, state, analytic.mu, sigmaPreMinus, ksi);
                        numericOutputSigmaPreClip[dim] = (jPlusSigma - jMinusSigma) / (2f * epsilon);
                        outputSigmaAbsErr[dim] = MathF.Abs(analytic.dJdSigmaPreClip[0, dim] - numericOutputSigmaPreClip[dim]);
                    }

                    float outputMuMeanAbsError = outputMuAbsErr.Average();
                    float outputMuMaxAbsError = outputMuAbsErr.Max();
                    float outputMuCosine = CosineSimilarity(analytic.dJdMu.ToArray(), numericOutputMu);
                    totalOutputMuMeanAbsError += outputMuMeanAbsError;
                    totalOutputMuMaxAbsError = MathF.Max(totalOutputMuMaxAbsError, outputMuMaxAbsError);
                    totalOutputMuCosine += outputMuCosine;

                    float outputSigmaMeanAbsError = outputSigmaAbsErr.Average();
                    float outputSigmaMaxAbsError = outputSigmaAbsErr.Max();
                    float outputSigmaCosine = CosineSimilarity(analytic.dJdSigmaPreClip.ToArray(), numericOutputSigmaPreClip);
                    totalOutputSigmaMeanAbsError += outputSigmaMeanAbsError;
                    totalOutputSigmaMaxAbsError = MathF.Max(totalOutputSigmaMaxAbsError, outputSigmaMaxAbsError);
                    totalOutputSigmaCosine += outputSigmaCosine;

                    sb.AppendLine($"sample={sampleIndex + 1} objective={analytic.objective:F6}");
                    sb.AppendLine($"  output(mu): mean|err|={outputMuMeanAbsError:E3} max|err|={outputMuMaxAbsError:E3} cos={outputMuCosine:F6}");
                    sb.AppendLine($"    analytic: [{string.Join(", ", analytic.dJdMu.ToArray().Select(x => x.ToString("E4")))}]");
                    sb.AppendLine($"    numeric : [{string.Join(", ", numericOutputMu.Select(x => x.ToString("E4")))}]");
                    sb.AppendLine($"  output(sigma_preclip): mean|err|={outputSigmaMeanAbsError:E3} max|err|={outputSigmaMaxAbsError:E3} cos={outputSigmaCosine:F6}");
                    sb.AppendLine($"    analytic: [{string.Join(", ", analytic.dJdSigmaPreClip.ToArray().Select(x => x.ToString("E4")))}]");
                    sb.AppendLine($"    numeric : [{string.Join(", ", numericOutputSigmaPreClip.Select(x => x.ToString("E4")))}]");

                    if (muSelected.Count > 0)
                    {
                        float[] analyticMuParams = new float[muSelected.Count];
                        float[] numericMuParams = new float[muSelected.Count];
                        float[] muAbsErr = new float[muSelected.Count];

                        for (int paramIndex = 0; paramIndex < muSelected.Count; paramIndex++)
                        {
                            var picked = muSelected[paramIndex];
                            analyticMuParams[paramIndex] = FindGradientForTensor(muParams, picked.tensor)[picked.index];

                            float original = picked.tensor[picked.index];
                            picked.tensor[picked.index] = original + epsilon;
                            float jPlus = EvaluateObjective(behaviour, muNetwork, sigmaNetwork, q1Network, q2Network, state, ksi);

                            picked.tensor[picked.index] = original - epsilon;
                            float jMinus = EvaluateObjective(behaviour, muNetwork, sigmaNetwork, q1Network, q2Network, state, ksi);

                            picked.tensor[picked.index] = original;

                            numericMuParams[paramIndex] = (jPlus - jMinus) / (2f * epsilon);
                            muAbsErr[paramIndex] = MathF.Abs(analyticMuParams[paramIndex] - numericMuParams[paramIndex]);
                        }

                        float muMeanAbsError = muAbsErr.Average();
                        float muMaxAbsError = muAbsErr.Max();
                        float muCos = CosineSimilarity(analyticMuParams, numericMuParams);
                        totalParamMuMeanAbsError += muMeanAbsError;
                        totalParamMuMaxAbsError = MathF.Max(totalParamMuMaxAbsError, muMaxAbsError);
                        totalParamMuCosine += muCos;

                        sb.AppendLine($"  params(mu): mean|err|={muMeanAbsError:E3} max|err|={muMaxAbsError:E3} cos={muCos:F6}");
                        for (int paramIndex = 0; paramIndex < muSelected.Count; paramIndex++)
                        {
                            sb.AppendLine(
                                $"    {muSelected[paramIndex].label}: analytic={analyticMuParams[paramIndex]:E4} numeric={numericMuParams[paramIndex]:E4} absErr={muAbsErr[paramIndex]:E4}");
                        }
                    }

                    if (sigmaSelected.Count > 0)
                    {
                        float[] analyticSigmaParams = new float[sigmaSelected.Count];
                        float[] numericSigmaParams = new float[sigmaSelected.Count];
                        float[] sigmaAbsErr = new float[sigmaSelected.Count];

                        for (int paramIndex = 0; paramIndex < sigmaSelected.Count; paramIndex++)
                        {
                            var picked = sigmaSelected[paramIndex];
                            analyticSigmaParams[paramIndex] = FindGradientForTensor(sigmaParams, picked.tensor)[picked.index];

                            float original = picked.tensor[picked.index];
                            picked.tensor[picked.index] = original + epsilon;
                            float jPlus = EvaluateObjective(behaviour, muNetwork, sigmaNetwork, q1Network, q2Network, state, ksi);

                            picked.tensor[picked.index] = original - epsilon;
                            float jMinus = EvaluateObjective(behaviour, muNetwork, sigmaNetwork, q1Network, q2Network, state, ksi);

                            picked.tensor[picked.index] = original;

                            numericSigmaParams[paramIndex] = (jPlus - jMinus) / (2f * epsilon);
                            sigmaAbsErr[paramIndex] = MathF.Abs(analyticSigmaParams[paramIndex] - numericSigmaParams[paramIndex]);
                        }

                        float sigmaMeanAbsError = sigmaAbsErr.Average();
                        float sigmaMaxAbsError = sigmaAbsErr.Max();
                        float sigmaCos = CosineSimilarity(analyticSigmaParams, numericSigmaParams);
                        totalParamSigmaMeanAbsError += sigmaMeanAbsError;
                        totalParamSigmaMaxAbsError = MathF.Max(totalParamSigmaMaxAbsError, sigmaMaxAbsError);
                        totalParamSigmaCosine += sigmaCos;

                        sb.AppendLine($"  params(sigma): mean|err|={sigmaMeanAbsError:E3} max|err|={sigmaMaxAbsError:E3} cos={sigmaCos:F6}");
                        for (int paramIndex = 0; paramIndex < sigmaSelected.Count; paramIndex++)
                        {
                            sb.AppendLine(
                                $"    {sigmaSelected[paramIndex].label}: analytic={analyticSigmaParams[paramIndex]:E4} numeric={numericSigmaParams[paramIndex]:E4} absErr={sigmaAbsErr[paramIndex]:E4}");
                        }
                    }
                }

                sb.AppendLine(
                    $"summary output(mu) mean|err|={(totalOutputMuMeanAbsError / samples):E3} " +
                    $"max|err|={totalOutputMuMaxAbsError:E3} avgCos={(totalOutputMuCosine / samples):F6}");
                sb.AppendLine(
                    $"summary output(sigma_preclip) mean|err|={(totalOutputSigmaMeanAbsError / samples):E3} " +
                    $"max|err|={totalOutputSigmaMaxAbsError:E3} avgCos={(totalOutputSigmaCosine / samples):F6}");

                if (muSelected.Count > 0)
                {
                    sb.AppendLine(
                        $"summary params(mu) mean|err|={(totalParamMuMeanAbsError / samples):E3} " +
                        $"max|err|={totalParamMuMaxAbsError:E3} avgCos={(totalParamMuCosine / samples):F6}");
                }

                if (sigmaSelected.Count > 0)
                {
                    sb.AppendLine(
                        $"summary params(sigma) mean|err|={(totalParamSigmaMeanAbsError / samples):E3} " +
                        $"max|err|={totalParamSigmaMaxAbsError:E3} avgCos={(totalParamSigmaCosine / samples):F6}");
                }

                string report = sb.ToString();
                string path = WriteReport($"{SanitizeName(behaviour.behaviourName)}_{probeDevice}", report);
                Debug.Log(report);
                Debug.Log($"[SacStochasticActorGradientProbe] Report written to {path}");
                return path;
            }
            finally
            {
                Restore(muLearnables, muDevices, muReqGrad);
                Restore(sigmaLearnables, sigmaDevices, sigmaReqGrad);
                Restore(q1Learnables, q1Devices, q1ReqGrad);
                Restore(q2Learnables, q2Devices, q2ReqGrad);
            }
        }

        private static SampleComputation ComputeAnalyticSample(
            AgentBehaviour behaviour,
            Sequential muNetwork,
            Sequential sigmaNetwork,
            Sequential q1Network,
            Sequential q2Network,
            Tensor state,
            Tensor ksi)
        {
            Parameter[] muParams = muNetwork.Parameters();
            Parameter[] sigmaParams = sigmaNetwork.Parameters();
            ZeroGradients(muParams);
            ZeroGradients(sigmaParams);

            Tensor mu = muNetwork.Forward(state);
            Tensor sigmaNetworkOut = sigmaNetwork.Forward(state);
            Tensor sigmaPreClip = sigmaNetworkOut * behaviour.standardDeviationScale;
            Tensor sigma = sigmaPreClip.Clip(1e-6f, 10f);
            Tensor u = mu + sigma * ksi;
            Tensor action = u.Tanh();
            Tensor pair = Pairify(state, action);

            Tensor q1 = q1Network.Forward(pair);
            Tensor q2 = q2Network.Forward(pair);

            Tensor dMinQdQ1 = q1 <= q2;
            Tensor dMinQdQ2 = Tensor.LogicalNot(dMinQdQ1);

            Tensor dMinQdStateAction = q1Network.Backward(dMinQdQ1) + q2Network.Backward(dMinQdQ2);
            Tensor dMinQdAction = ExtractActionFromStateAction(dMinQdStateAction, state.Size(-1), action.Size(-1));
            Tensor dQdu = dMinQdAction * (1f - action.Pow(2f));

            float log2Pi = MathF.Log(2f * MathF.PI);
            int dims = action.Size(-1);
            Tensor logSigmaSum = sigma.Log().Sum(-1, keepDim: true);
            Tensor ksiSqSum = ksi.Pow(2f).Sum(-1, keepDim: true);
            Tensor logPiGaussian = -0.5f * dims * log2Pi - logSigmaSum - 0.5f * ksiSqSum;
            Tensor tanhCorrection = Tensor.Sum(2f * (MathF.Log(2f) - u - Tensor.Softplus(-2f * u)), -1, true);
            Tensor logPi = logPiGaussian - tanhCorrection;

            Tensor dCorrectionDu = -2f * action;
            Tensor dLogPiDmu = -dCorrectionDu;
            Tensor dLogPiDsigma = -dCorrectionDu * ksi - 1f / sigma;
            Tensor dJdMu = dQdu - behaviour.config.alpha * dLogPiDmu;
            Tensor dJdSigma = dQdu * ksi - behaviour.config.alpha * dLogPiDsigma;
            Tensor sigmaClipMask = sigmaPreClip.Select(x => x >= 1e-6f && x <= 10f ? 1f : 0f);
            Tensor dJdSigmaPreClip = dJdSigma * sigmaClipMask;
            Tensor dJdSigmaNetOut = dJdSigmaPreClip * behaviour.standardDeviationScale;

            muNetwork.Backward(dJdMu);
            sigmaNetwork.Backward(dJdSigmaNetOut);

            float minQ = Tensor.Minimum(q1, q2)[0];
            float entropyTerm = (-behaviour.config.alpha * logPi)[0];

            return new SampleComputation
            {
                mu = mu,
                sigmaPreClip = sigmaPreClip,
                dJdMu = dJdMu,
                dJdSigmaPreClip = dJdSigmaPreClip,
                objective = minQ + entropyTerm
            };
        }

        private static float EvaluateObjective(
            AgentBehaviour behaviour,
            Sequential muNetwork,
            Sequential sigmaNetwork,
            Sequential q1Network,
            Sequential q2Network,
            Tensor state,
            Tensor ksi)
        {
            Tensor mu = muNetwork.Predict(state);
            Tensor sigmaPreClip = sigmaNetwork.Predict(state) * behaviour.standardDeviationScale;
            return EvaluateObjectiveFromOutputs(behaviour, q1Network, q2Network, state, mu, sigmaPreClip, ksi);
        }

        private static float EvaluateObjectiveFromOutputs(
            AgentBehaviour behaviour,
            Sequential q1Network,
            Sequential q2Network,
            Tensor state,
            Tensor mu,
            Tensor sigmaPreClip,
            Tensor ksi)
        {
            Tensor sigma = sigmaPreClip.Clip(1e-6f, 10f);
            Tensor u = mu + sigma * ksi;
            Tensor action = u.Tanh();
            Tensor pair = Pairify(state, action);

            Tensor q1 = q1Network.Predict(pair);
            Tensor q2 = q2Network.Predict(pair);

            float log2Pi = MathF.Log(2f * MathF.PI);
            int dims = action.Size(-1);
            Tensor logSigmaSum = sigma.Log().Sum(-1, keepDim: true);
            Tensor ksiSqSum = ksi.Pow(2f).Sum(-1, keepDim: true);
            Tensor logPiGaussian = -0.5f * dims * log2Pi - logSigmaSum - 0.5f * ksiSqSum;
            Tensor tanhCorrection = Tensor.Sum(2f * (MathF.Log(2f) - u - Tensor.Softplus(-2f * u)), -1, true);
            Tensor logPi = logPiGaussian - tanhCorrection;

            float minQ = MathF.Min(q1[0], q2[0]);
            float entropyTerm = (-behaviour.config.alpha * logPi)[0];
            return minQ + entropyTerm;
        }

        private static void Restore(ILearnable[] learnables, Device[] devices, bool[] requiresGrad)
        {
            for (int i = 0; i < learnables.Length; i++)
            {
                learnables[i].Device = devices[i];
                learnables[i].RequiresGrad = requiresGrad[i];
            }
        }

        private static void ValidateNetwork(Sequential network, string name)
        {
            if (network == null)
                throw new InvalidOperationException($"{name} is null.");
            if (network.Modules == null)
                throw new InvalidOperationException($"{name}.Modules is null.");
            if (network.Modules.Length == 0)
                throw new InvalidOperationException($"{name}.Modules is empty.");
            if (network.Modules.Any(module => module == null))
                throw new InvalidOperationException($"{name}.Modules contains null entries.");
        }

        private static void ZeroGradients(Parameter[] parameters)
        {
            foreach (var parameter in parameters)
            {
                Tensor.CopyTo(Tensor.Zeros(parameter.g.Shape), parameter.g);
            }
        }

        private static List<(Tensor tensor, int index, string label)> SelectParameters(Parameter[] parameters, int maxCount, string prefix)
        {
            List<(Tensor tensor, int index, string label)> selected = new List<(Tensor tensor, int index, string label)>();
            int paramGroup = 0;

            foreach (var parameter in parameters)
            {
                int count = parameter.param.Count();
                if (count == 0)
                {
                    paramGroup++;
                    continue;
                }

                int[] picks = count <= 2
                    ? Enumerable.Range(0, count).ToArray()
                    : new[] { 0, count / 2, count - 1 }.Distinct().ToArray();

                foreach (int pick in picks)
                {
                    selected.Add((parameter.param, pick, $"{prefix}[{paramGroup}][{pick}]"));
                    if (selected.Count >= maxCount)
                        return selected;
                }

                paramGroup++;
            }

            return selected;
        }

        private static Tensor FindGradientForTensor(Parameter[] parameters, Tensor targetTensor)
        {
            foreach (var parameter in parameters)
            {
                if (ReferenceEquals(parameter.param, targetTensor))
                    return parameter.g;
            }

            throw new InvalidOperationException("Gradient tensor for selected parameter was not found.");
        }

        private static Tensor Pairify(Tensor stateBatch, Tensor actionBatch)
        {
            int batchSize = stateBatch.Size(0);
            int stateSize = stateBatch.Size(-1);
            int actionSize = actionBatch.Size(-1);
            Tensor pair = Tensor.Zeros(batchSize, stateSize + actionSize);

            for (int b = 0; b < batchSize; b++)
            {
                for (int s = 0; s < stateSize; s++)
                {
                    pair[b, s] = stateBatch[b, s];
                }

                for (int a = 0; a < actionSize; a++)
                {
                    pair[b, stateSize + a] = actionBatch[b, a];
                }
            }

            return pair;
        }

        private static Tensor ExtractActionFromStateAction(Tensor stateActionBatch, int stateSize, int actionSize)
        {
            int batchSize = stateActionBatch.Size(0);
            Tensor actions = Tensor.Zeros(batchSize, actionSize);

            for (int b = 0; b < batchSize; b++)
            {
                for (int a = 0; a < actionSize; a++)
                {
                    actions[b, a] = stateActionBatch[b, stateSize + a];
                }
            }

            return actions;
        }

        private static float CosineSimilarity(IEnumerable<float> left, IEnumerable<float> right)
        {
            float dot = 0f;
            float leftNorm = 0f;
            float rightNorm = 0f;

            using IEnumerator<float> leftEnum = left.GetEnumerator();
            using IEnumerator<float> rightEnum = right.GetEnumerator();
            while (leftEnum.MoveNext() && rightEnum.MoveNext())
            {
                float l = leftEnum.Current;
                float r = rightEnum.Current;
                dot += l * r;
                leftNorm += l * l;
                rightNorm += r * r;
            }

            if (leftNorm <= 1e-20f && rightNorm <= 1e-20f)
                return 1f;
            if (leftNorm <= 1e-20f || rightNorm <= 1e-20f)
                return 0f;

            return dot / (MathF.Sqrt(leftNorm) * MathF.Sqrt(rightNorm));
        }

        private static string WriteReport(string stem, string report)
        {
            string logsDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "ProbeLogs"));
            Directory.CreateDirectory(logsDir);
            string path = Path.Combine(logsDir, $"sac_stochastic_actor_gradient_probe_{stem}.txt");
            File.WriteAllText(path, report);
            return path;
        }

        private static string SanitizeName(string value)
        {
            if (string.IsNullOrWhiteSpace(value))
                return "unnamed";

            char[] invalid = Path.GetInvalidFileNameChars();
            StringBuilder sb = new StringBuilder(value.Length);
            foreach (char c in value)
            {
                sb.Append(invalid.Contains(c) ? '_' : c);
            }

            return sb.ToString();
        }

        private sealed class SampleComputation
        {
            public Tensor mu;
            public Tensor sigmaPreClip;
            public Tensor dJdMu;
            public Tensor dJdSigmaPreClip;
            public float objective;
        }
    }
}
