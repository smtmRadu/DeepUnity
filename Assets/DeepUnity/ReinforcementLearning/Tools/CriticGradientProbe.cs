using DeepUnity.Models;
using DeepUnity.Modules;
using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Numerically checks dQ/da by comparing critic backpropagation against finite differences.
    /// Attach to any GameObject, assign a behaviour, and press the inspector button.
    /// </summary>
    public sealed class CriticGradientProbe : MonoBehaviour
    {
        [SerializeField] private AgentBehaviour behaviour;
        [SerializeField] private Device probeDevice = Device.CPU;
        [SerializeField, Min(1)] private int samples = 4;
        [SerializeField, Min(1e-6f)] private float epsilon = 1e-3f;
        [SerializeField, Min(1e-4f)] private float stateRange = 0.5f;
        [SerializeField, Min(1e-4f)] private float actionRange = 0.5f;
        [SerializeField] private bool probeQ1 = true;
        [SerializeField] private bool probeQ2 = true;

        [Button(nameof(RunGradientCheck))]
        [SerializeField] private bool runGradientCheckButton;

        public void RunGradientCheck()
        {
            if (behaviour == null)
            {
                Debug.LogError("[CriticGradientProbe] Behaviour is not assigned.");
                return;
            }

            if (behaviour.continuousDim <= 0)
            {
                Debug.LogError("[CriticGradientProbe] Behaviour does not use continuous actions.");
                return;
            }

            if (!probeQ1 && !probeQ2)
            {
                Debug.LogWarning("[CriticGradientProbe] Nothing selected. Enable Q1 and/or Q2.");
                return;
            }

            if (probeQ1)
                ProbeNetwork("Q1", behaviour.q1Network);

            if (probeQ2)
                ProbeNetwork("Q2", behaviour.q2Network);
        }

        private void ProbeNetwork(string label, Sequential network)
        {
            if (network == null)
            {
                Debug.LogError($"[CriticGradientProbe] {label} network is null.");
                return;
            }

            int stateSize = behaviour.observationSize * behaviour.stackedInputs;
            int actionSize = behaviour.continuousDim;
            ILearnable[] learnables = network.Modules.OfType<ILearnable>().ToArray();
            Device[] originalDevices = learnables.Select(x => x.Device).ToArray();
            bool[] originalRequiresGrad = learnables.Select(x => x.RequiresGrad).ToArray();

            try
            {
                for (int i = 0; i < learnables.Length; i++)
                {
                    learnables[i].Device = probeDevice;
                    learnables[i].RequiresGrad = false;
                }

                float totalMeanAbsError = 0f;
                float totalMaxAbsError = 0f;
                float totalCosine = 0f;

                StringBuilder sb = new StringBuilder();
                sb.AppendLine($"[CriticGradientProbe] {label} on {probeDevice}");
                sb.AppendLine($"samples={samples}, epsilon={epsilon}, stateSize={stateSize}, actionSize={actionSize}");

                for (int sampleIndex = 0; sampleIndex < samples; sampleIndex++)
                {
                    Tensor state = Tensor.RandomRange((-stateRange, stateRange), 1, stateSize);
                    Tensor action = Tensor.RandomRange((-actionRange, actionRange), 1, actionSize);

                    Tensor pair = Pairify(state, action);
                    Tensor q = network.Forward(pair);
                    Tensor dQdStateAction = network.Backward(Tensor.Ones(q.Shape));
                    Tensor dQdAction = ExtractActionFromStateAction(dQdStateAction, stateSize, actionSize);

                    float[] analytic = dQdAction.ToArray();
                    float[] numeric = new float[actionSize];
                    float[] absErrors = new float[actionSize];

                    for (int actionIndex = 0; actionIndex < actionSize; actionIndex++)
                    {
                        Tensor actionPlus = action.Clone() as Tensor;
                        Tensor actionMinus = action.Clone() as Tensor;
                        actionPlus[0, actionIndex] += epsilon;
                        actionMinus[0, actionIndex] -= epsilon;

                        float qPlus = network.Predict(Pairify(state, actionPlus))[0];
                        float qMinus = network.Predict(Pairify(state, actionMinus))[0];
                        numeric[actionIndex] = (qPlus - qMinus) / (2f * epsilon);
                        absErrors[actionIndex] = MathF.Abs(analytic[actionIndex] - numeric[actionIndex]);
                    }

                    float meanAbsError = absErrors.Average();
                    float maxAbsError = absErrors.Max();
                    float cosine = CosineSimilarity(analytic, numeric);

                    totalMeanAbsError += meanAbsError;
                    totalMaxAbsError = MathF.Max(totalMaxAbsError, maxAbsError);
                    totalCosine += cosine;

                    sb.AppendLine(
                        $"sample={sampleIndex + 1} q={q[0]:F6} " +
                        $"mean|err|={meanAbsError:E3} max|err|={maxAbsError:E3} cos={cosine:F6}");
                    sb.AppendLine($"  analytic: [{string.Join(", ", analytic.Select(x => x.ToString("E4")))}]");
                    sb.AppendLine($"  numeric : [{string.Join(", ", numeric.Select(x => x.ToString("E4")))}]");
                }

                sb.AppendLine(
                    $"summary mean|err|={(totalMeanAbsError / samples):E3} " +
                    $"max|err|={totalMaxAbsError:E3} avgCos={(totalCosine / samples):F6}");

                string report = sb.ToString();
                string path = WriteReport($"{SanitizeName(behaviour.behaviourName)}_{label}_{probeDevice}", report);
                Debug.Log(report);
                Debug.Log($"[CriticGradientProbe] Report written to {path}");
            }
            finally
            {
                for (int i = 0; i < learnables.Length; i++)
                {
                    learnables[i].Device = originalDevices[i];
                    learnables[i].RequiresGrad = originalRequiresGrad[i];
                }
            }
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
            string path = Path.Combine(logsDir, $"critic_gradient_probe_{stem}.txt");
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
