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
    [InitializeOnLoad]
    public static class BalanceBallBatchTrainingRunner
    {
        [Serializable]
        private sealed class BatchRunSpec
        {
            public string scenario;
            public string runId;
            public string behaviourName;
            public string reportPath;
            public int randomSeed;
            public int targetSteps;
            public long startedAtUtcTicks;
            public long lastHeartbeatUtcTicks;
            public double timeoutSeconds;
            public float actorLearningRate;
            public float criticLearningRate;
            public float gamma;
            public int replayBufferSize;
            public int minibatchSize;
            public int updateInterval;
            public int updateAfter;
            public int updatesNum;
            public float alpha;
            public float tau;
            public float maxNorm;
            public int timescale;
            public Device inferenceDevice;
            public Device trainingDevice;
        }

        private const string ScenePath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBall.unity";
        private const string SpecKey = "DeepUnity.BalanceBallBatchTrainingRunner.Spec";
        private const string ActiveKey = "DeepUnity.BalanceBallBatchTrainingRunner.Active";
        private const string StopRequestedKey = "DeepUnity.BalanceBallBatchTrainingRunner.StopRequested";
        private const string ExitCodeKey = "DeepUnity.BalanceBallBatchTrainingRunner.ExitCode";

        static BalanceBallBatchTrainingRunner()
        {
            EditorApplication.update -= OnEditorUpdate;
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        public static void RunBalanceBallSacBaseline()
        {
            StartRun(new BatchRunSpec
            {
                scenario = "baseline",
                randomSeed = 0,
                targetSteps = 1000,
                timeoutSeconds = 240d,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                gamma = 0.99f,
                replayBufferSize = 1_000_000,
                minibatchSize = 64,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 1,
                alpha = 0.2f,
                tau = 0.005f,
                maxNorm = 0.5f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
            });
        }

        public static void RunBalanceBallSacDenseUpdates()
        {
            StartRun(new BatchRunSpec
            {
                scenario = "dense_updates",
                randomSeed = 0,
                targetSteps = 1000,
                timeoutSeconds = 240d,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                gamma = 0.99f,
                replayBufferSize = 1_000_000,
                minibatchSize = 64,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                alpha = 0.2f,
                tau = 0.005f,
                maxNorm = 0.5f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
            });
        }

        private static void StartRun(BatchRunSpec spec)
        {
            try
            {
                ClearRunState();

                spec.startedAtUtcTicks = DateTime.UtcNow.Ticks;
                spec.lastHeartbeatUtcTicks = spec.startedAtUtcTicks;
                spec.runId = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                spec.behaviourName = $"__BalanceBallSac_{spec.scenario}_{spec.runId}";
                spec.reportPath = Path.Combine(ProjectRoot(), "ProbeLogs", $"balanceball_batch_{spec.scenario}_{spec.runId}.md");

                SessionState.SetString(SpecKey, JsonUtility.ToJson(spec));
                SessionState.SetBool(ActiveKey, true);
                SessionState.SetBool(StopRequestedKey, false);
                SessionState.SetInt(ExitCodeKey, 0);

                PrepareScene(spec);
                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                FailRun($"Failed before play mode: {ex}");
            }
        }

        private static void PrepareScene(BatchRunSpec spec)
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            var balanceBall = UnityEngine.Object.FindObjectsOfType<BalanceBall>(true).FirstOrDefault();
            if (balanceBall == null)
                throw new InvalidOperationException("BalanceBall agent was not found in the scene.");

            foreach (var agent in UnityEngine.Object.FindObjectsOfType<Agent>(true))
            {
                bool keep = agent == balanceBall;
                agent.behaviourType = keep ? BehaviourType.Learn : BehaviourType.Off;
                agent.enabled = keep;
                if (!keep)
                    agent.gameObject.SetActive(false);

                EditorUtility.SetDirty(agent);
            }

            var decisionRequester = balanceBall.GetComponent<DecisionRequester>();
            if (decisionRequester == null)
                throw new InvalidOperationException("BalanceBall agent is missing DecisionRequester.");

            decisionRequester.decisionPeriod = 1;
            decisionRequester.takeActionsBetweenDecisions = true;
            decisionRequester.maxStep = 10_000;
            EditorUtility.SetDirty(decisionRequester);

            var stats = balanceBall.GetComponent<TrainingStatistics>();
            if (stats == null)
                stats = balanceBall.gameObject.AddComponent<TrainingStatistics>();

            EditorUtility.SetDirty(stats);

            Utils.Random.Seed = spec.randomSeed;

            AgentBehaviour behaviour = AgentBehaviour.CreateOrLoadAsset(
                spec.behaviourName,
                stateSize: 10,
                stackedInputs: 1,
                widthSize: 16,
                heightSize: 16,
                channelSize: 3,
                continuousActions: 2,
                discreteActions: 0,
                numLayers: 2,
                hidUnits: 32,
                aType: ArchitectureType.MLP,
                nonlinearity: NonLinearity.Tanh);

            ConfigureBehaviour(behaviour, spec);

            balanceBall.model = behaviour;
            balanceBall.behaviourType = BehaviourType.Learn;
            balanceBall.enabled = true;
            EditorUtility.SetDirty(balanceBall);

            AssetDatabase.SaveAssets();
        }

        private static void ConfigureBehaviour(AgentBehaviour behaviour, BatchRunSpec spec)
        {
            behaviour.inferenceDevice = spec.inferenceDevice;
            behaviour.trainingDevice = spec.trainingDevice;
            behaviour.targetFPS = 50;
            behaviour.clipping = 5f;
            behaviour.normalize = false;
            behaviour.stochasticity = Stochasticity.FixedStandardDeviation;
            behaviour.standardDeviationValue = 1f;
            behaviour.standardDeviationScale = 1.5f;
            behaviour.noiseValue = 0f;

            Hyperparameters config = behaviour.config;
            config.trainer = TrainerType.SACDepr;
            config.maxSteps = int.MaxValue;
            config.actorLearningRate = spec.actorLearningRate;
            config.criticLearningRate = spec.criticLearningRate;
            config.gamma = spec.gamma;
            config.LRSchedule = false;
            config.maxNorm = spec.maxNorm;
            config.replayBufferSize = spec.replayBufferSize;
            config.minibatchSize = spec.minibatchSize;
            config.updateInterval = spec.updateInterval;
            config.updateAfter = spec.updateAfter;
            config.updatesNum = spec.updatesNum;
            config.alpha = spec.alpha;
            config.tau = spec.tau;
            config.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            config.timescale = spec.timescale;
            config.debug = false;
            config.sacDebugMetrics = false;

            EditorUtility.SetDirty(config);
            EditorUtility.SetDirty(behaviour);
        }

        private static void OnEditorUpdate()
        {
            if (!SessionState.GetBool(ActiveKey, false))
                return;

            if (!EditorApplication.isPlaying)
                return;

            try
            {
                BatchRunSpec spec = LoadSpec();
                DeepUnityTrainer trainer = DeepUnityTrainer.Instance;
                if (trainer == null)
                {
                    if (ElapsedSeconds(spec.startedAtUtcTicks) >= spec.timeoutSeconds)
                        StopRun("timeout_waiting_for_trainer");
                    return;
                }

                if (ElapsedSeconds(spec.lastHeartbeatUtcTicks) >= 5d)
                {
                    spec.lastHeartbeatUtcTicks = DateTime.UtcNow.Ticks;
                    SaveSpec(spec);
                    Debug.Log($"[BalanceBallBatchTrainingRunner] scenario={spec.scenario} step={trainer.currentSteps} updates={trainer.updateIterations} actorLoss={trainer.actorLoss:F4} criticLoss={trainer.criticLoss:F4}");
                }

                if (trainer.ended)
                {
                    StopRun("trainer_ended");
                    return;
                }

                if (ElapsedSeconds(spec.startedAtUtcTicks) >= spec.timeoutSeconds)
                {
                    StopRun("timeout");
                    return;
                }

                if (trainer.currentSteps >= spec.targetSteps)
                {
                    StopRun("target_steps_reached");
                }
            }
            catch (Exception ex)
            {
                FailRun($"Failed during play mode update: {ex}");
            }
        }

        private static void StopRun(string reason)
        {
            if (SessionState.GetBool(StopRequestedKey, false))
                return;

            try
            {
                BatchRunSpec spec = LoadSpec();
                WriteReport(spec, reason);
                SessionState.SetBool(StopRequestedKey, true);
                EditorApplication.isPlaying = false;
            }
            catch (Exception ex)
            {
                FailRun($"Failed while stopping run: {ex}");
            }
        }

        private static void OnPlayModeStateChanged(PlayModeStateChange state)
        {
            if (!SessionState.GetBool(ActiveKey, false))
                return;

            if (state == PlayModeStateChange.EnteredEditMode && SessionState.GetBool(StopRequestedKey, false))
            {
                int exitCode = SessionState.GetInt(ExitCodeKey, 0);
                ClearRunState();
                EditorApplication.Exit(exitCode);
            }
        }

        private static void WriteReport(BatchRunSpec spec, string reason)
        {
            DeepUnityTrainer trainer = DeepUnityTrainer.Instance;
            if (trainer == null)
                throw new InvalidOperationException("Trainer was null while writing the report.");

            TrainingStatistics stats = UnityEngine.Object.FindObjectOfType<TrainingStatistics>();
            if (stats == null)
                throw new InvalidOperationException("TrainingStatistics component was not found while writing the report.");

            stats.startedAt = trainer.timeWhenTheTrainingStarted.ToLongTimeString() + ", " + trainer.timeWhenTheTrainingStarted.ToLongDateString();
            stats.finishedAt = DateTime.Now.ToLongTimeString() + ", " + DateTime.Now.ToLongDateString();

            string svgPath = "(not exported by batch runner)";

            string probeDir = Path.GetDirectoryName(spec.reportPath);
            if (!Directory.Exists(probeDir))
                Directory.CreateDirectory(probeDir);

            float[] rewards = stats.cumulativeReward.Keys.Select(x => x.value).ToArray();
            float[] actorLoss = stats.actorLoss.Keys.Select(x => x.value).ToArray();
            float[] criticLoss = stats.criticLoss.Keys.Select(x => x.value).ToArray();

            float meanRewardLast10 = AverageTail(rewards, 10);
            float meanRewardLast20 = AverageTail(rewards, 20);
            float latestReward = rewards.Length > 0 ? rewards[^1] : float.NaN;
            float latestActorLoss = actorLoss.Length > 0 ? actorLoss[^1] : float.NaN;
            float latestCriticLoss = criticLoss.Length > 0 ? criticLoss[^1] : float.NaN;

            StringBuilder sb = new StringBuilder();
            sb.AppendLine($"# BalanceBall SAC Batch Run");
            sb.AppendLine();
            sb.AppendLine($"- Scenario: `{spec.scenario}`");
            sb.AppendLine($"- Run Id: `{spec.runId}`");
            sb.AppendLine($"- Stop Reason: `{reason}`");
            sb.AppendLine($"- Duration Seconds: `{ElapsedSeconds(spec.startedAtUtcTicks):F1}`");
            sb.AppendLine($"- Behaviour Asset: `Assets/{spec.behaviourName}/_{spec.behaviourName}.asset`");
            sb.AppendLine($"- SVG Log: `{svgPath}`");
            sb.AppendLine($"- Current Steps: `{trainer.currentSteps}`");
            sb.AppendLine($"- Update Iterations: `{trainer.updateIterations}`");
            sb.AppendLine($"- Episode Count: `{stats.episodeCount}`");
            sb.AppendLine($"- Mean Reward Last 10 Episodes: `{meanRewardLast10}`");
            sb.AppendLine($"- Mean Reward Last 20 Episodes: `{meanRewardLast20}`");
            sb.AppendLine($"- Latest Episode Reward: `{latestReward}`");
            sb.AppendLine($"- Latest Actor Loss: `{latestActorLoss}`");
            sb.AppendLine($"- Latest Critic Loss: `{latestCriticLoss}`");
            sb.AppendLine($"- Alpha: `{trainer.hp.alpha}`");
            sb.AppendLine($"- Minibatch Size: `{trainer.hp.minibatchSize}`");
            sb.AppendLine($"- Update Interval: `{trainer.hp.updateInterval}`");
            sb.AppendLine($"- Updates Num: `{trainer.hp.updatesNum}`");
            sb.AppendLine($"- Update After: `{trainer.hp.updateAfter}`");
            sb.AppendLine($"- Timescale: `{trainer.hp.timescale}`");

            File.WriteAllText(spec.reportPath, sb.ToString());
            SessionState.SetInt(ExitCodeKey, 0);
        }

        private static void FailRun(string message)
        {
            try
            {
                Directory.CreateDirectory(Path.Combine(ProjectRoot(), "ProbeLogs"));
                string fallbackName = $"balanceball_batch_failure_{DateTime.Now:yyyyMMdd_HHmmss}.md";
                string fallbackPath = Path.Combine(ProjectRoot(), "ProbeLogs", fallbackName);

                string configuredReportPath = SessionState.GetString(SpecKey, string.Empty);
                if (!string.IsNullOrWhiteSpace(configuredReportPath))
                {
                    BatchRunSpec spec = JsonUtility.FromJson<BatchRunSpec>(configuredReportPath);
                    fallbackPath = spec.reportPath;
                }

                File.WriteAllText(fallbackPath, $"# BalanceBall SAC Batch Run Failure{Environment.NewLine}{Environment.NewLine}```text{Environment.NewLine}{message}{Environment.NewLine}```{Environment.NewLine}");
            }
            catch
            {
            }

            SessionState.SetInt(ExitCodeKey, 1);
            SessionState.SetBool(ActiveKey, false);
            SessionState.SetBool(StopRequestedKey, false);
            EditorApplication.Exit(1);
        }

        private static BatchRunSpec LoadSpec()
        {
            string json = SessionState.GetString(SpecKey, string.Empty);
            if (string.IsNullOrWhiteSpace(json))
                throw new InvalidOperationException("No batch run spec was found in SessionState.");

            BatchRunSpec spec = JsonUtility.FromJson<BatchRunSpec>(json);
            if (spec == null)
                throw new InvalidOperationException("Batch run spec could not be deserialized.");

            return spec;
        }

        private static void SaveSpec(BatchRunSpec spec)
        {
            SessionState.SetString(SpecKey, JsonUtility.ToJson(spec));
        }

        private static float AverageTail(float[] values, int tail)
        {
            if (values == null || values.Length == 0)
                return float.NaN;

            int start = Math.Max(0, values.Length - tail);
            return values.Skip(start).Average();
        }

        private static double ElapsedSeconds(long utcTicks)
        {
            return (DateTime.UtcNow - new DateTime(utcTicks, DateTimeKind.Utc)).TotalSeconds;
        }

        private static void ClearRunState()
        {
            SessionState.EraseString(SpecKey);
            SessionState.SetBool(ActiveKey, false);
            SessionState.SetBool(StopRequestedKey, false);
            SessionState.SetInt(ExitCodeKey, 0);
        }

        private static string ProjectRoot()
        {
            string assetsPath = Application.dataPath;
            return Directory.GetParent(assetsPath)?.FullName ?? assetsPath;
        }
    }
}
#endif
