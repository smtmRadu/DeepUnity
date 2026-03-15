using DeepUnity.Models;
using DeepUnity.Modules;
using System;
using System.IO;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Checks the actor-chain gradient for J = Q(s, mu(s)) by comparing analytic actor parameter
    /// gradients against finite differences on selected actor parameters.
    /// </summary>
    public sealed class ActorChainGradientProbe : MonoBehaviour
    {
        [SerializeField] private AgentBehaviour behaviour;
        [SerializeField] private Device probeDevice = Device.CPU;
        [SerializeField, Min(1)] private int samples = 3;
        [SerializeField, Min(1)] private int checkedParameters = 8;
        [SerializeField, Min(1e-6f)] private float epsilon = 1e-3f;
        [SerializeField, Min(1e-4f)] private float stateRange = 0.5f;

        [Button(nameof(RunGradientCheck))]
        [SerializeField] private bool runGradientCheckButton;

        public void RunGradientCheck()
        {
            if (behaviour == null)
            {
                Debug.LogError("[ActorChainGradientProbe] Behaviour is not assigned.");
                return;
            }

            if (behaviour.muNetwork == null || behaviour.q1Network == null)
            {
                Debug.LogError("[ActorChainGradientProbe] Behaviour is missing muNetwork or q1Network.");
                return;
            }

            if (behaviour.continuousDim <= 0)
            {
                Debug.LogError("[ActorChainGradientProbe] Behaviour does not use continuous actions.");
                return;
            }

            ProbeActorChain();
        }

        private void ProbeActorChain()
        {
            int stateSize = behaviour.observationSize * behaviour.stackedInputs;
            int actionSize = behaviour.continuousDim;
            Sequential actor = behaviour.muNetwork;
            Sequential critic = behaviour.q1Network;

            ILearnable[] actorLearnables = actor.Modules.OfType<ILearnable>().ToArray();
            ILearnable[] criticLearnables = critic.Modules.OfType<ILearnable>().ToArray();
            Device[] actorDevices = actorLearnables.Select(x => x.Device).ToArray();
            Device[] criticDevices = criticLearnables.Select(x => x.Device).ToArray();
            bool[] actorReqGrad = actorLearnables.Select(x => x.RequiresGrad).ToArray();
            bool[] criticReqGrad = criticLearnables.Select(x => x.RequiresGrad).ToArray();

            try
            {
                foreach (var item in actorLearnables)
                {
                    item.Device = probeDevice;
                    item.RequiresGrad = true;
                }

                foreach (var item in criticLearnables)
                {
                    item.Device = probeDevice;
                    item.RequiresGrad = false;
                }

                Parameter[] actorParams = actor.Parameters();
                List<(Tensor tensor, int index, string label)> selected = SelectParameters(actorParams, checkedParameters);
                if (selected.Count == 0)
                {
                    Debug.LogError("[ActorChainGradientProbe] No actor parameters available to check.");
                    return;
                }

                float totalMeanAbsError = 0f;
                float totalMaxAbsError = 0f;
                float totalCosine = 0f;
                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"[ActorChainGradientProbe] Q1(mu(s)) on {probeDevice}");
                sb.AppendLine($"samples={samples}, epsilon={epsilon}, checkedParameters={selected.Count}, stateSize={stateSize}, actionSize={actionSize}");

                for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
                {
                    ZeroActorGradients(actorParams);

                    Tensor state = Tensor.RandomRange((-stateRange, stateRange), 1, stateSize);
                    Tensor mu = actor.Forward(state);
                    Tensor q = critic.Forward(Pairify(state, mu));
                    Tensor dQdStateAction = critic.Backward(Tensor.Ones(q.Shape));
                    Tensor dQdAction = ExtractActionFromStateAction(dQdStateAction, stateSize, actionSize);
                    actor.Backward(dQdAction);

                    float[] analytic = new float[selected.Count];
                    float[] numeric = new float[selected.Count];
                    float[] absErrors = new float[selected.Count];

                    for (int paramIndex = 0; paramIndex < selected.Count; paramIndex++)
                    {
                        var picked = selected[paramIndex];
                        analytic[paramIndex] = FindGradientForTensor(actorParams, picked.tensor)[picked.index];

                        float original = picked.tensor[picked.index];
                        picked.tensor[picked.index] = original + epsilon;
                        float qPlus = EvaluateObjective(actor, critic, state);

                        picked.tensor[picked.index] = original - epsilon;
                        float qMinus = EvaluateObjective(actor, critic, state);

                        picked.tensor[picked.index] = original;

                        numeric[paramIndex] = (qPlus - qMinus) / (2f * epsilon);
                        absErrors[paramIndex] = MathF.Abs(analytic[paramIndex] - numeric[paramIndex]);
                    }

                    float meanAbsError = absErrors.Average();
                    float maxAbsError = absErrors.Max();
                    float cosine = CosineSimilarity(analytic, numeric);
                    totalMeanAbsError += meanAbsError;
                    totalMaxAbsError = MathF.Max(totalMaxAbsError, maxAbsError);
                    totalCosine += cosine;

                    sb.AppendLine($"sample={sampleIndex + 1} q={q[0]:F6} mean|err|={meanAbsError:E3} max|err|={maxAbsError:E3} cos={cosine:F6}");
                    for (int paramIndex = 0; paramIndex < selected.Count; paramIndex++)
                    {
                        sb.AppendLine(
                            $"  {selected[paramIndex].label}: analytic={analytic[paramIndex]:E4} numeric={numeric[paramIndex]:E4} absErr={absErrors[paramIndex]:E4}");
                    }
                }

                sb.AppendLine(
                    $"summary mean|err|={(totalMeanAbsError / samples):E3} " +
                    $"max|err|={totalMaxAbsError:E3} avgCos={(totalCosine / samples):F6}");

                string report = sb.ToString();
                string path = WriteReport($"{SanitizeName(behaviour.behaviourName)}_{probeDevice}", report);
                Debug.Log(report);
                Debug.Log($"[ActorChainGradientProbe] Report written to {path}");
            }
            finally
            {
                for (int i = 0; i < actorLearnables.Length; i++)
                {
                    actorLearnables[i].Device = actorDevices[i];
                    actorLearnables[i].RequiresGrad = actorReqGrad[i];
                }

                for (int i = 0; i < criticLearnables.Length; i++)
                {
                    criticLearnables[i].Device = criticDevices[i];
                    criticLearnables[i].RequiresGrad = criticReqGrad[i];
                }
            }
        }

        private static void ZeroActorGradients(Parameter[] actorParams)
        {
            foreach (var param in actorParams)
            {
                Tensor.CopyTo(Tensor.Zeros(param.g.Shape), param.g);
            }
        }

        private static List<(Tensor tensor, int index, string label)> SelectParameters(Parameter[] parameters, int maxCount)
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
                    selected.Add((parameter.param, pick, $"param[{paramGroup}][{pick}]"));
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

        private static float EvaluateObjective(Sequential actor, Sequential critic, Tensor state)
        {
            Tensor mu = actor.Predict(state);
            Tensor q = critic.Predict(Pairify(state, mu));
            return q[0];
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

        private static float CosineSimilarity(float[] left, float[] right)
        {
            float dot = 0f;
            float leftNorm = 0f;
            float rightNorm = 0f;

            for (int i = 0; i < left.Length; i++)
            {
                dot += left[i] * right[i];
                leftNorm += left[i] * left[i];
                rightNorm += right[i] * right[i];
            }

            if (leftNorm <= 0f || rightNorm <= 0f)
                return 0f;

            return dot / (MathF.Sqrt(leftNorm) * MathF.Sqrt(rightNorm));
        }

        private static string WriteReport(string stem, string report)
        {
            string logsDir = Path.GetFullPath(Path.Combine(Application.dataPath, "..", "ProbeLogs"));
            Directory.CreateDirectory(logsDir);
            string path = Path.Combine(logsDir, $"actor_chain_gradient_probe_{stem}.txt");
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
    }
}
