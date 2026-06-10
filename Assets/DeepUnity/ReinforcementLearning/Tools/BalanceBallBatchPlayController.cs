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
    public sealed class BalanceBallBatchPlayController : MonoBehaviour
    {
        [SerializeField] private string reportDirectory;
        [SerializeField] private string runId;
        [SerializeField] private string behaviourName;
        [SerializeField] private int targetSteps = 5000;
        [SerializeField] private float timeoutSeconds = 600f;

        private float startedAtRealtime;
        private float lastProgressRealtime;
        private bool finished;

        private void Awake()
        {
            startedAtRealtime = Time.realtimeSinceStartup;
            lastProgressRealtime = startedAtRealtime;
            Application.runInBackground = true;
        }

        private void Update()
        {
            if (finished)
                return;

            DeepUnityTrainer trainer = DeepUnityTrainer.Instance;
            if (trainer == null)
            {
                if (Time.realtimeSinceStartup - startedAtRealtime >= timeoutSeconds)
                    Finish(success: false, reason: "Trainer was never initialized before timeout.");

                return;
            }

            if (Time.realtimeSinceStartup - lastProgressRealtime >= 5f)
            {
                lastProgressRealtime = Time.realtimeSinceStartup;
                int effectiveGradientSteps = trainer.updateIterations * Mathf.Max(1, trainer.hp.updatesNum);
                float approxUtd = trainer.currentSteps > 0 ? effectiveGradientSteps / (float)trainer.currentSteps : 0f;
                Debug.Log($"[BalanceBallBatchPlayController] step={trainer.currentSteps} updates={trainer.updateIterations} gradSteps~={effectiveGradientSteps} utd~={approxUtd:F3} actorLoss={trainer.actorLoss:F4} criticLoss={trainer.criticLoss:F4}");
            }

            if (trainer.currentSteps >= targetSteps)
            {
                Finish(success: true, reason: $"Reached target steps ({targetSteps}).");
                return;
            }

            if (Time.realtimeSinceStartup - startedAtRealtime >= timeoutSeconds)
                Finish(success: false, reason: $"Timeout after {timeoutSeconds:0} seconds.");
        }

        private void Finish(bool success, string reason)
        {
            if (finished)
                return;

            finished = true;
            Directory.CreateDirectory(reportDirectory);

            TrainingStatistics stats = GetComponent<TrainingStatistics>();
            if (stats == null)
                stats = FindObjectOfType<TrainingStatistics>(); // batch runners attach it to the agent, not this controller
            DeepUnityTrainer trainer = DeepUnityTrainer.Instance;

            string reportPath = Path.Combine(reportDirectory, "report.md");
            File.WriteAllText(reportPath, BuildReport(success, reason, trainer, stats));

            if (stats != null)
            {
                WriteCurveCsv(Path.Combine(reportDirectory, "episode_reward.csv"), stats.cumulativeReward.Keys);
                WriteCurveCsv(Path.Combine(reportDirectory, "episode_length.csv"), stats.episodeLength.Keys);
                WriteCurveCsv(Path.Combine(reportDirectory, "actor_loss.csv"), stats.actorLoss.Keys);
                WriteCurveCsv(Path.Combine(reportDirectory, "critic_loss.csv"), stats.criticLoss.Keys);
                WriteCurveCsv(Path.Combine(reportDirectory, "entropy.csv"), stats.entropy.Keys);
            }

            Debug.Log($"[BalanceBallBatchPlayController] {reason} Report: {reportPath}");

#if UNITY_EDITOR
            EditorApplication.Exit(success ? 0 : 1);
#else
            Application.Quit();
#endif
        }

        private string BuildReport(bool success, string reason, DeepUnityTrainer trainer, TrainingStatistics stats)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("# BalanceBall SAC Autonomous Run");
            sb.AppendLine();
            sb.AppendLine($"- Author: Codex");
            sb.AppendLine($"- Run ID: `{runId}`");
            sb.AppendLine($"- Behaviour Name: `{behaviourName}`");
            sb.AppendLine($"- Success: `{success}`");
            sb.AppendLine($"- Reason: {reason}");
            sb.AppendLine($"- Started UTC: `{DateTime.UtcNow.AddSeconds(-(Time.realtimeSinceStartup - startedAtRealtime)):O}`");
            sb.AppendLine($"- Finished UTC: `{DateTime.UtcNow:O}`");
            sb.AppendLine();

            if (trainer != null)
            {
                int effectiveGradientSteps = trainer.updateIterations * Mathf.Max(1, trainer.hp.updatesNum);
                float approxUtd = trainer.currentSteps > 0 ? effectiveGradientSteps / (float)trainer.currentSteps : 0f;

                sb.AppendLine("## Trainer");
                sb.AppendLine($"- Steps: `{trainer.currentSteps}`");
                sb.AppendLine($"- Update iterations: `{trainer.updateIterations}`");
                sb.AppendLine($"- Effective Gradient Steps (approx): `{effectiveGradientSteps}`");
                sb.AppendLine($"- Approx UTD Ratio: `{approxUtd:F4}`");
                sb.AppendLine($"- Actor loss: `{trainer.actorLoss}`");
                sb.AppendLine($"- Critic loss: `{trainer.criticLoss}`");
                sb.AppendLine($"- Entropy: `{trainer.entropy}`");
                sb.AppendLine();
            }

            if (stats != null)
            {
                Keyframe[] rewards = stats.cumulativeReward.Keys;
                Keyframe[] actorLoss = stats.actorLoss.Keys;
                Keyframe[] criticLoss = stats.criticLoss.Keys;

                sb.AppendLine("## Statistics");
                sb.AppendLine($"- Episode count: `{stats.episodeCount}`");
                sb.AppendLine($"- Last reward: `{GetLastValue(rewards):F4}`");
                sb.AppendLine($"- Mean reward last 10: `{MeanOfLast(rewards, 10):F4}`");
                sb.AppendLine($"- Mean reward last 25: `{MeanOfLast(rewards, 25):F4}`");
                sb.AppendLine($"- Last actor loss: `{GetLastValue(actorLoss):F4}`");
                sb.AppendLine($"- Last critic loss: `{GetLastValue(criticLoss):F4}`");
            }

            return sb.ToString();
        }

        private static void WriteCurveCsv(string path, Keyframe[] keys)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("time,value");
            foreach (Keyframe key in keys ?? Array.Empty<Keyframe>())
                sb.AppendLine($"{key.time},{key.value}");

            File.WriteAllText(path, sb.ToString());
        }

        private static float GetLastValue(Keyframe[] keys)
        {
            if (keys == null || keys.Length == 0)
                return float.NaN;

            return keys[keys.Length - 1].value;
        }

        private static float MeanOfLast(Keyframe[] keys, int count)
        {
            if (keys == null || keys.Length == 0)
                return float.NaN;

            Keyframe[] tail = keys.Skip(Mathf.Max(0, keys.Length - count)).ToArray();
            return tail.Average(x => x.value);
        }

        public void Configure(string reportDirectory, string runId, string behaviourName, int targetSteps, float timeoutSeconds)
        {
            this.reportDirectory = reportDirectory;
            this.runId = runId;
            this.behaviourName = behaviourName;
            this.targetSteps = targetSteps;
            this.timeoutSeconds = timeoutSeconds;
        }
    }
}
