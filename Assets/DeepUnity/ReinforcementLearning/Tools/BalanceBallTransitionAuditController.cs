using System;
using System.IO;
using System.Linq;
using System.Text;
using UnityEngine;

#if UNITY_EDITOR
using UnityEditor;
#endif

namespace DeepUnity.ReinforcementLearning
{
    public sealed class BalanceBallTransitionAuditController : MonoBehaviour
    {
        [SerializeField] private Agent agent;
        [SerializeField] private string reportPath;
        [SerializeField] private int targetTransitions = 32;
        [SerializeField] private float timeoutSeconds = 120f;

        private float startedAtRealtime;
        private int observedTransitions;
        private int terminalTransitions;
        private int firstTerminalReplayIndex = -1;
        private bool finished;
        private readonly StringBuilder report = new StringBuilder();

        private void Awake()
        {
            startedAtRealtime = Time.realtimeSinceStartup;
            Application.runInBackground = true;
            report.AppendLine("# BalanceBall Transition Audit");
            report.AppendLine();
            report.AppendLine($"- Started UTC: `{DateTime.UtcNow:O}`");
            report.AppendLine($"- Target transitions: `{targetTransitions}`");
            report.AppendLine();
        }

        public void Configure(Agent agent, string reportPath, int targetTransitions, float timeoutSeconds)
        {
            this.agent = agent;
            this.reportPath = reportPath;
            this.targetTransitions = targetTransitions;
            this.timeoutSeconds = timeoutSeconds;
        }

        private void Update()
        {
            if (finished)
                return;

            if (agent == null)
            {
                if (Time.realtimeSinceStartup - startedAtRealtime >= timeoutSeconds)
                    Finish(false, "Agent was never assigned before timeout.");
                return;
            }

            MemoryBuffer memory = agent.Memory;
            if (memory != null)
            {
                while (observedTransitions < memory.Count)
                {
                    TimestepTuple ts = memory.frames[observedTransitions];
                    AppendTransition(observedTransitions, ts);
                    if (ts.done != null && ts.done[0] > 0.5f)
                    {
                        terminalTransitions++;
                        if (firstTerminalReplayIndex < 0)
                            firstTerminalReplayIndex = observedTransitions;
                    }
                    observedTransitions++;
                }
            }

            if (firstTerminalReplayIndex >= 0 && observedTransitions >= firstTerminalReplayIndex + 6)
            {
                Finish(true, $"Captured {observedTransitions} transitions with {terminalTransitions} terminal transitions. First terminal at replay index {firstTerminalReplayIndex}.");
                return;
            }

            if (Time.realtimeSinceStartup - startedAtRealtime >= timeoutSeconds)
                Finish(false, $"Timeout after {timeoutSeconds:0} seconds. Captured {observedTransitions} transitions.");
        }

        private void AppendTransition(int replayIndex, TimestepTuple ts)
        {
            report.AppendLine($"## Replay[{replayIndex}]");
            report.AppendLine($"- timestepIndex: `{ts?.index}`");
            report.AppendLine($"- reward: `{SafeScalar(ts?.reward):F6}`");
            report.AppendLine($"- done: `{SafeScalar(ts?.done):F1}`");
            report.AppendLine($"- truncated: `{SafeScalar(ts?.truncated):F1}`");
            report.AppendLine($"- state null: `{ts?.state == null}`");
            report.AppendLine($"- action null: `{ts?.action_continuous == null}`");
            report.AppendLine($"- nextState null: `{ts?.nextState == null}`");
            report.AppendLine($"- state: `{FormatTensor(ts?.state)}`");
            report.AppendLine($"- action: `{FormatTensor(ts?.action_continuous)}`");
            report.AppendLine($"- nextState: `{FormatTensor(ts?.nextState)}`");
            report.AppendLine($"- state->next abs mean delta: `{MeanAbsDelta(ts?.state, ts?.nextState):F6}`");
            report.AppendLine();
        }

        private void Finish(bool success, string reason)
        {
            if (finished)
                return;

            finished = true;
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");

            report.Insert(report.ToString().IndexOf(Environment.NewLine, StringComparison.Ordinal) + Environment.NewLine.Length,
                $"{Environment.NewLine}- Success: `{success}`{Environment.NewLine}- Reason: {reason}{Environment.NewLine}- Finished UTC: `{DateTime.UtcNow:O}`{Environment.NewLine}- Terminal transitions: `{terminalTransitions}`{Environment.NewLine}");

            File.WriteAllText(reportPath, report.ToString());
            Debug.Log($"[BalanceBallTransitionAuditController] {reason} Report: {reportPath}");

#if UNITY_EDITOR
            EditorApplication.Exit(success ? 0 : 1);
#else
            Application.Quit();
#endif
        }

        private static float SafeScalar(Tensor tensor)
        {
            if (tensor == null)
                return float.NaN;

            float[] values = tensor.ToArray();
            if (values == null || values.Length == 0)
                return float.NaN;

            return values[0];
        }

        private static float MeanAbsDelta(Tensor a, Tensor b)
        {
            if (a == null || b == null)
                return float.NaN;

            float[] av = a.ToArray();
            float[] bv = b.ToArray();
            if (av == null || bv == null || av.Length != bv.Length)
                return float.NaN;

            float sum = 0f;
            for (int i = 0; i < av.Length; i++)
                sum += Mathf.Abs(av[i] - bv[i]);

            return sum / Mathf.Max(1, av.Length);
        }

        private static string FormatTensor(Tensor tensor)
        {
            if (tensor == null)
                return "null";

            float[] values = tensor.ToArray();
            return string.Join(", ", values.Select(x => x.ToString("0.0000")));
        }
    }
}
