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
    public static class SacSmokeBatchRunner
    {
        [Serializable]
        private sealed class BatchRunSpec
        {
            public string scenario;
            public string runId;
            public string reportPath;
            public string behaviourName;
            public string behaviourAssetPath;
            public int targetSteps;
            public int randomSeed;
            public long startedAtUtcTicks;
            public long lastHeartbeatUtcTicks;
            public double timeoutSeconds;
            public int updateInterval;
            public int updatesNum;
            public int updateAfter;
            public int minibatchSize;
            public int replayBufferSize;
            public float alpha;
            public float tau;
            public float actorLearningRate;
            public float criticLearningRate;
            public float gamma;
            public int timescale;
        }

        private const string SpecKey = "DeepUnity.SacSmokeBatchRunner.Spec";
        private const string ActiveKey = "DeepUnity.SacSmokeBatchRunner.Active";
        private const string StopRequestedKey = "DeepUnity.SacSmokeBatchRunner.StopRequested";
        private const string ExitCodeKey = "DeepUnity.SacSmokeBatchRunner.ExitCode";

        static SacSmokeBatchRunner()
        {
            EditorApplication.update -= OnEditorUpdate;
            EditorApplication.update += OnEditorUpdate;
            EditorApplication.playModeStateChanged -= OnPlayModeStateChanged;
            EditorApplication.playModeStateChanged += OnPlayModeStateChanged;
        }

        public static void RunSacSmokeBaseline()
        {
            StartRun(new BatchRunSpec
            {
                scenario = "baseline",
                randomSeed = 0,
                targetSteps = 3000,
                timeoutSeconds = 120d,
                updateInterval = 50,
                updatesNum = 1,
                updateAfter = 1024,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                gamma = 0.99f,
                timescale = 20,
            });
        }

        public static void RunSacSmokeDenseUpdates()
        {
            StartRun(new BatchRunSpec
            {
                scenario = "dense_updates",
                randomSeed = 0,
                targetSteps = 3000,
                timeoutSeconds = 120d,
                updateInterval = 1,
                updatesNum = 1,
                updateAfter = 1024,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                gamma = 0.99f,
                timescale = 20,
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
                spec.behaviourName = $"__SacSmoke_{spec.scenario}_{spec.runId}";
                spec.reportPath = Path.Combine(ProjectRoot(), "ProbeLogs", $"sac_smoke_{spec.scenario}_{spec.runId}.md");

                PrepareScene(spec);

                SessionState.SetString(SpecKey, JsonUtility.ToJson(spec));
                SessionState.SetBool(ActiveKey, true);
                SessionState.SetBool(StopRequestedKey, false);
                SessionState.SetInt(ExitCodeKey, 0);

                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                FailRun($"Failed before play mode: {ex}");
            }
        }

        private static void PrepareScene(BatchRunSpec spec)
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            GameObject agentGo = new GameObject("QuadraticBanditAgent");
            QuadraticBanditAgent agent = agentGo.AddComponent<QuadraticBanditAgent>();
            DecisionRequester requester = agentGo.AddComponent<DecisionRequester>();
            TrainingStatistics stats = agentGo.AddComponent<TrainingStatistics>();

            requester.decisionPeriod = 1;
            requester.takeActionsBetweenDecisions = true;
            requester.maxStep = 1;

            agent.behaviourType = BehaviourType.Learn;
            ConfigureSerializedAgent(agent);

            AgentBehaviour behaviour = AgentBehaviour.CreateOrLoadAsset(
                spec.behaviourName,
                stateSize: 2,
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

            agent.model = behaviour;

            spec.behaviourAssetPath = AssetDatabase.GetAssetPath(behaviour);

            EditorUtility.SetDirty(requester);
            EditorUtility.SetDirty(stats);
            EditorUtility.SetDirty(agent);
            EditorUtility.SetDirty(behaviour);
            EditorUtility.SetDirty(behaviour.config);
            AssetDatabase.SaveAssets();

            Utils.Random.Seed = spec.randomSeed;
        }

        private static void ConfigureSerializedAgent(QuadraticBanditAgent agent)
        {
            SerializedObject serializedAgent = new SerializedObject(agent);
            serializedAgent.FindProperty("spaceSize").intValue = 2;
            serializedAgent.FindProperty("stackedInputs").intValue = 1;
            serializedAgent.FindProperty("continuousActions").intValue = 2;
            serializedAgent.FindProperty("discreteActions").intValue = 0;
            serializedAgent.FindProperty("archType").enumValueIndex = (int)ArchitectureType.MLP;
            serializedAgent.FindProperty("numLayers").intValue = 2;
            serializedAgent.FindProperty("hidUnits").intValue = 32;
            serializedAgent.FindProperty("activation").enumValueIndex = (int)NonLinearity.Tanh;
            serializedAgent.FindProperty("onEpisodeEnd").enumValueIndex = (int)OnEpisodeEndType.NothingHappens;
            serializedAgent.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void ConfigureBehaviour(AgentBehaviour behaviour, BatchRunSpec spec)
        {
            behaviour.inferenceDevice = Device.CPU;
            behaviour.trainingDevice = Device.CPU;
            behaviour.targetFPS = 50;
            behaviour.clipping = 5f;
            behaviour.normalize = false;
            behaviour.stochasticity = Stochasticity.FixedStandardDeviation;
            behaviour.standardDeviationValue = 1f;
            behaviour.standardDeviationScale = 1.5f;
            behaviour.noiseValue = 0f;

            Hyperparameters hp = behaviour.config;
            hp.trainer = TrainerType.SAC;
            hp.maxSteps = int.MaxValue;
            hp.actorLearningRate = spec.actorLearningRate;
            hp.criticLearningRate = spec.criticLearningRate;
            hp.gamma = spec.gamma;
            hp.LRSchedule = false;
            hp.replayBufferSize = spec.replayBufferSize;
            hp.minibatchSize = spec.minibatchSize;
            hp.updateInterval = spec.updateInterval;
            hp.updateAfter = spec.updateAfter;
            hp.updatesNum = spec.updatesNum;
            hp.alpha = spec.alpha;
            hp.tau = spec.tau;
            hp.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            hp.timescale = spec.timescale;
            hp.debug = false;
            hp.sacDebugMetrics = false;
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

                if (ElapsedSeconds(spec.lastHeartbeatUtcTicks) >= 2d)
                {
                    spec.lastHeartbeatUtcTicks = DateTime.UtcNow.Ticks;
                    SaveSpec(spec);
                    Debug.Log($"[SacSmokeBatchRunner] scenario={spec.scenario} step={trainer.currentSteps} updates={trainer.updateIterations} actorLoss={trainer.actorLoss:F4} criticLoss={trainer.criticLoss:F4}");
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

            string reportDir = Path.GetDirectoryName(spec.reportPath);
            if (!Directory.Exists(reportDir))
                Directory.CreateDirectory(reportDir);

            float[] rewards = stats.cumulativeReward.Keys.Select(x => x.value).ToArray();
            float[] actorLoss = stats.actorLoss.Keys.Select(x => x.value).ToArray();
            float[] criticLoss = stats.criticLoss.Keys.Select(x => x.value).ToArray();

            WriteCurveCsv(Path.Combine(reportDir, $"sac_smoke_{spec.scenario}_reward.csv"), rewards);
            WriteCurveCsv(Path.Combine(reportDir, $"sac_smoke_{spec.scenario}_actor_loss.csv"), actorLoss);
            WriteCurveCsv(Path.Combine(reportDir, $"sac_smoke_{spec.scenario}_critic_loss.csv"), criticLoss);

            StringBuilder sb = new StringBuilder();
            sb.AppendLine("# SAC Smoke Batch Run");
            sb.AppendLine();
            sb.AppendLine($"- Author: Codex");
            sb.AppendLine($"- Scenario: `{spec.scenario}`");
            sb.AppendLine($"- Run Id: `{spec.runId}`");
            sb.AppendLine($"- Stop Reason: `{reason}`");
            sb.AppendLine($"- Duration Seconds: `{ElapsedSeconds(spec.startedAtUtcTicks):F1}`");
            sb.AppendLine($"- Behaviour Asset: `{spec.behaviourAssetPath}`");
            sb.AppendLine($"- Current Steps: `{trainer.currentSteps}`");
            sb.AppendLine($"- Update Iterations: `{trainer.updateIterations}`");
            sb.AppendLine($"- Episode Count: `{stats.episodeCount}`");
            sb.AppendLine($"- Mean Reward Last 20 Episodes: `{AverageTail(rewards, 20):F4}`");
            sb.AppendLine($"- Mean Reward Last 100 Episodes: `{AverageTail(rewards, 100):F4}`");
            sb.AppendLine($"- Latest Reward: `{Latest(rewards):F4}`");
            sb.AppendLine($"- Latest Actor Loss: `{Latest(actorLoss):F4}`");
            sb.AppendLine($"- Latest Critic Loss: `{Latest(criticLoss):F4}`");
            sb.AppendLine($"- Update Interval: `{spec.updateInterval}`");
            sb.AppendLine($"- Updates Num: `{spec.updatesNum}`");
            sb.AppendLine($"- Update After: `{spec.updateAfter}`");
            sb.AppendLine();
            sb.AppendLine("This environment is a one-step 2D quadratic bandit.");
            sb.AppendLine("Optimal reward is 1.0 when the action matches the observed target exactly.");

            File.WriteAllText(spec.reportPath, sb.ToString());
            Debug.Log($"[SacSmokeBatchRunner] Completed scenario={spec.scenario}. Report: {spec.reportPath}");
        }

        private static void WriteCurveCsv(string path, float[] values)
        {
            StringBuilder sb = new StringBuilder();
            sb.AppendLine("index,value");
            for (int i = 0; i < values.Length; i++)
                sb.AppendLine($"{i},{values[i]}");
            File.WriteAllText(path, sb.ToString());
        }

        private static float AverageTail(float[] values, int count)
        {
            if (values == null || values.Length == 0)
                return float.NaN;

            int take = Math.Min(count, values.Length);
            return values.Skip(values.Length - take).Average();
        }

        private static float Latest(float[] values)
        {
            return values != null && values.Length > 0 ? values[^1] : float.NaN;
        }

        private static BatchRunSpec LoadSpec()
        {
            string json = SessionState.GetString(SpecKey, string.Empty);
            if (string.IsNullOrWhiteSpace(json))
                throw new InvalidOperationException("No smoke-run spec was found in SessionState.");

            return JsonUtility.FromJson<BatchRunSpec>(json);
        }

        private static void SaveSpec(BatchRunSpec spec)
        {
            SessionState.SetString(SpecKey, JsonUtility.ToJson(spec));
        }

        private static double ElapsedSeconds(long startedAtUtcTicks)
        {
            return (DateTime.UtcNow - new DateTime(startedAtUtcTicks, DateTimeKind.Utc)).TotalSeconds;
        }

        private static string ProjectRoot()
        {
            return Directory.GetCurrentDirectory();
        }

        private static void FailRun(string reason)
        {
            try
            {
                string reportPath = "(unavailable)";
                if (!string.IsNullOrWhiteSpace(SessionState.GetString(SpecKey, string.Empty)))
                {
                    BatchRunSpec spec = LoadSpec();
                    reportPath = spec.reportPath;
                    string directory = Path.GetDirectoryName(reportPath);
                    if (!Directory.Exists(directory))
                        Directory.CreateDirectory(directory);
                    File.WriteAllText(reportPath, $"# SAC Smoke Batch Run Failure\n\n- Reason: {reason}\n");
                }

                Debug.LogError($"[SacSmokeBatchRunner] {reason} Report: {reportPath}");
            }
            catch (Exception ex)
            {
                Debug.LogError($"[SacSmokeBatchRunner] Failed while logging error: {ex}");
            }

            SessionState.SetInt(ExitCodeKey, 1);
            SessionState.SetBool(ActiveKey, false);
            SessionState.SetBool(StopRequestedKey, false);
            EditorApplication.Exit(1);
        }

        private static void ClearRunState()
        {
            SessionState.EraseString(SpecKey);
            SessionState.SetBool(ActiveKey, false);
            SessionState.SetBool(StopRequestedKey, false);
            SessionState.SetInt(ExitCodeKey, 0);
        }
    }
}
#endif
