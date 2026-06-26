#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Text;
using DeepUnity.Tutorials;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class BalanceBallSacAutonomousRunner
    {
        private const string ScenePath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBall.unity";
        private const int TargetSteps = 5000;
        private const double TimeoutSeconds = 600.0;

        private static DateTime startedAtUtc;
        private static DateTime lastProgressAtUtc;
        private static string runId;
        private static string reportDirectory;
        private static string tempBehaviourName;
        private static bool runCompleted;

        public static void RunFreshBalanceBallSacCpu()
        {
            try
            {
                PrepareScene();
                EditorApplication.update += MonitorRun;
                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        private static void PrepareScene()
        {
            runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            startedAtUtc = DateTime.UtcNow;
            lastProgressAtUtc = startedAtUtc;
            runCompleted = false;

            reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"balanceball_sac_run_{runId}");
            Directory.CreateDirectory(reportDirectory);

            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            BalanceBall[] agents = UnityEngine.Object.FindObjectsOfType<BalanceBall>(true);
            if (agents.Length == 0)
                throw new InvalidOperationException("No BalanceBall agents were found in the scene.");

            BalanceBall targetAgent = agents.FirstOrDefault(IsMlpBalanceBallAgent);

            if (targetAgent == null)
                throw new InvalidOperationException("Could not find the MLP BalanceBall agent in the scene.");

            foreach (Agent agent in UnityEngine.Object.FindObjectsOfType<Agent>(true))
            {
                agent.behaviourType = agent == targetAgent ? BehaviourType.Learn : BehaviourType.Off;
            }

            tempBehaviourName = $"AutonomousBalanceBallSac_{runId}";
            AgentBehaviour freshBehaviour = CreateFreshBehaviourFromAgent(targetAgent, tempBehaviourName);
            ApplySacConfig(freshBehaviour);
            targetAgent.model = freshBehaviour;

            if (targetAgent.GetComponent<TrainingStatistics>() == null)
                targetAgent.gameObject.AddComponent<TrainingStatistics>();

            EditorUtility.SetDirty(targetAgent);
            EditorUtility.SetDirty(freshBehaviour);
            EditorUtility.SetDirty(freshBehaviour.config);
            AssetDatabase.SaveAssets();

            Utils.Random.Seed = 0;

            Debug.Log($"[BalanceBallSacAutonomousRunner] Prepared fresh SAC run {runId} using {tempBehaviourName}.");
        }

        private static bool IsMlpBalanceBallAgent(BalanceBall agent)
        {
            if (agent == null)
                return false;

            SerializedObject serializedAgent = new SerializedObject(agent);
            int archType = serializedAgent.FindProperty("archType").enumValueIndex;
            int stateSize = serializedAgent.FindProperty("spaceSize").intValue;
            int continuousActions = serializedAgent.FindProperty("continuousActions").intValue;
            return archType == (int)ArchitectureType.MLP && stateSize == 10 && continuousActions == 2;
        }

        private static AgentBehaviour CreateFreshBehaviourFromAgent(BalanceBall agent, string behaviourName)
        {
            SerializedObject serializedAgent = new SerializedObject(agent);

            int stateSize = serializedAgent.FindProperty("spaceSize").intValue;
            int stackedInputs = serializedAgent.FindProperty("stackedInputs").intValue;
            int widthSize = serializedAgent.FindProperty("spaceWidth").intValue;
            int heightSize = serializedAgent.FindProperty("spaceHeight").intValue;
            int channelSize = serializedAgent.FindProperty("spaceChannels").intValue;
            int continuousActions = serializedAgent.FindProperty("continuousActions").intValue;
            int discreteActions = serializedAgent.FindProperty("discreteActions").intValue;
            int numLayers = serializedAgent.FindProperty("numLayers").intValue;
            int hidUnits = serializedAgent.FindProperty("hidUnits").intValue;
            ArchitectureType archType = (ArchitectureType)serializedAgent.FindProperty("archType").enumValueIndex;
            NonLinearity nonLinearity = (NonLinearity)serializedAgent.FindProperty("activation").enumValueIndex;

            AgentBehaviour fresh = AgentBehaviour.CreateOrLoadAsset(
                behaviourName,
                stateSize,
                stackedInputs,
                widthSize,
                heightSize,
                channelSize,
                continuousActions,
                discreteActions,
                numLayers,
                hidUnits,
                archType,
                nonLinearity);

            AgentBehaviour source = agent.model;
            fresh.inferenceDevice = Device.CPU;
            fresh.trainingDevice = Device.CPU;
            fresh.targetFPS = source.targetFPS;
            fresh.clipping = source.clipping;
            fresh.normalize = false;
            fresh.stochasticity = source.stochasticity;
            fresh.standardDeviationValue = source.standardDeviationValue;
            fresh.standardDeviationScale = source.standardDeviationScale;
            fresh.noiseValue = source.noiseValue;

            return fresh;
        }

        private static void ApplySacConfig(AgentBehaviour behaviour)
        {
            Hyperparameters hp = behaviour.config;
            hp.trainer = TrainerType.SACDepr;
            hp.maxSteps = 2_000_000_000;
            hp.actorLearningRate = 1e-3f;
            hp.criticLearningRate = 1e-3f;
            hp.gamma = 0.99f;
            hp.LRSchedule = false;

            hp.replayBufferSize = 1_000_000;
            hp.minibatchSize = 64;
            hp.updateInterval = 50;
            hp.updateAfter = 1024;
            hp.updatesNum = 1;
            hp.alpha = 0.2f;
            hp.tau = 0.005f;
            hp.sacDebugMetrics = true;
            hp.sacDebugEveryNUpdates = 25;

            hp.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            hp.timescale = 20f;
        }

        private static void MonitorRun()
        {
            if (!EditorApplication.isPlaying)
                return;

            DeepUnityTrainer trainer = DeepUnityTrainer.Instance;
            if (trainer == null)
                return;

            if ((DateTime.UtcNow - lastProgressAtUtc).TotalSeconds >= 5.0)
            {
                lastProgressAtUtc = DateTime.UtcNow;
                Debug.Log($"[BalanceBallSacAutonomousRunner] step={trainer.currentSteps} updates={trainer.updateIterations} actorLoss={trainer.actorLoss:F4} criticLoss={trainer.criticLoss:F4}");
            }

            if (trainer.currentSteps >= TargetSteps)
            {
                FinishRun(success: true, reason: $"Reached target steps ({TargetSteps}).");
                return;
            }

            if ((DateTime.UtcNow - startedAtUtc).TotalSeconds >= TimeoutSeconds)
            {
                FinishRun(success: false, reason: $"Timeout after {TimeoutSeconds:0} seconds.");
            }
        }

        private static void FinishRun(bool success, string reason)
        {
            if (runCompleted)
                return;

            runCompleted = true;
            EditorApplication.update -= MonitorRun;

            TrainingStatistics stats = UnityEngine.Object.FindObjectOfType<TrainingStatistics>();
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

            Debug.Log($"[BalanceBallSacAutonomousRunner] {reason} Report: {reportPath}");
            EditorApplication.isPlaying = false;
            EditorApplication.delayCall += () => EditorApplication.Exit(success ? 0 : 1);
        }

        private static string BuildReport(bool success, string reason, DeepUnityTrainer trainer, TrainingStatistics stats)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("# BalanceBall SAC Autonomous Run");
            sb.AppendLine();
            sb.AppendLine($"- Author: Codex");
            sb.AppendLine($"- Run ID: `{runId}`");
            sb.AppendLine($"- Success: `{success}`");
            sb.AppendLine($"- Reason: {reason}");
            sb.AppendLine($"- Temp behaviour: `{tempBehaviourName}`");
            sb.AppendLine($"- Started UTC: `{startedAtUtc:O}`");
            sb.AppendLine($"- Finished UTC: `{DateTime.UtcNow:O}`");
            sb.AppendLine();

            if (trainer != null)
            {
                sb.AppendLine("## Trainer");
                sb.AppendLine($"- Steps: `{trainer.currentSteps}`");
                sb.AppendLine($"- Update iterations: `{trainer.updateIterations}`");
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
            {
                sb.AppendLine($"{key.time},{key.value}");
            }
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
    }
}
#endif
