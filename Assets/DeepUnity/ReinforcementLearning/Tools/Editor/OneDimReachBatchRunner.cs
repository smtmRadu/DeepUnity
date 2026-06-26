#if UNITY_EDITOR
using System;
using System.IO;
using DeepUnity.Tutorials;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class OneDimReachBatchRunner
    {
        private sealed class RunSpec
        {
            public string scenario;
            public int agentCount;
            public int targetSteps;
            public float timeoutSeconds;
            public int updateInterval;
            public int updateAfter;
            public int updatesNum;
            public int minibatchSize;
            public int replayBufferSize;
            public float alpha;
            public float tau;
            public float actorLearningRate;
            public float criticLearningRate;
            public int timescale;
            public TrainerType trainer = TrainerType.SACDepr;
        }

        public static void RunOneDimReachSacBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "baseline",
                agentCount = 1,
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 100_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
            });
        }

        public static void RunOneDimReachSacDenseUpdates()
        {
            StartRun(new RunSpec
            {
                scenario = "dense_updates",
                agentCount = 1,
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 100_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
            });
        }

        public static void RunOneDimReachSacGpuBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "sacgpu_baseline",
                agentCount = 1,
                targetSteps = 5000,
                timeoutSeconds = 240f,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 100_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                trainer = TrainerType.SAC,
            });
        }

        public static void RunOneDimReachSacGpuDenseUpdates()
        {
            StartRun(new RunSpec
            {
                scenario = "sacgpu_dense_updates",
                agentCount = 1,
                targetSteps = 5000,
                timeoutSeconds = 240f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 100_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                trainer = TrainerType.SAC,
            });
        }

        public static void RunOneDimReachSacTwoAgentsBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "two_agents_baseline",
                agentCount = 2,
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 100_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
            });
        }

        public static void RunOneDimReachSacTwoAgentsDenseUpdates()
        {
            StartRun(new RunSpec
            {
                scenario = "two_agents_dense_updates",
                agentCount = 2,
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 100_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
            });
        }

        private static void StartRun(RunSpec spec)
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"one_dim_reach_{spec.scenario}_{runId}");
            string behaviourName = $"__OneDimReachSac_{spec.scenario}_{runId}";

            try
            {
                SetupScene(spec, runId, reportDirectory, behaviourName);
                Debug.Log($"[OneDimReachBatchRunner] Prepared {spec.scenario} run. Report dir: {reportDirectory}");
                EditorApplication.isPlaying = true;
            }
            catch (Exception ex)
            {
                Directory.CreateDirectory(reportDirectory);
                File.WriteAllText(Path.Combine(reportDirectory, "failure.txt"), ex.ToString());
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        private static void SetupScene(RunSpec spec, string runId, string reportDirectory, string behaviourName)
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            AgentBehaviour behaviour = AgentBehaviour.CreateOrLoadAsset(
                behaviourName,
                stateSize: 1,
                stackedInputs: 1,
                widthSize: 16,
                heightSize: 16,
                channelSize: 3,
                continuousActions: 1,
                discreteActions: 0,
                numLayers: 2,
                hidUnits: 32,
                aType: ArchitectureType.MLP,
                nonlinearity: NonLinearity.Tanh);

            ConfigureBehaviour(behaviour, spec);

            for (int i = 0; i < spec.agentCount; i++)
            {
                GameObject agentGO = new GameObject($"OneDimReachAgent_{i + 1}");
                OneDimReachAgent agent = agentGO.AddComponent<OneDimReachAgent>();
                TrainingStatistics stats = agentGO.AddComponent<TrainingStatistics>();

                ConfigureAgentSerializedFields(agent);

                DecisionRequester requester = agent.GetComponent<DecisionRequester>();
                if (requester == null)
                    throw new InvalidOperationException("DecisionRequester was not auto-created for OneDimReachAgent.");

                requester.decisionPeriod = 1;
                requester.takeActionsBetweenDecisions = true;
                requester.maxStep = 25;

                agent.model = behaviour;
                agent.behaviourType = BehaviourType.Learn;
                agent.enabled = true;

                EditorUtility.SetDirty(agent);
                EditorUtility.SetDirty(requester);
                EditorUtility.SetDirty(stats);
            }

            GameObject controllerGO = new GameObject("BatchController");
            BalanceBallBatchPlayController playController = controllerGO.AddComponent<BalanceBallBatchPlayController>();

            playController.Configure(reportDirectory, runId, behaviourName, spec.targetSteps, spec.timeoutSeconds);

            EditorUtility.SetDirty(playController);
            EditorUtility.SetDirty(behaviour);
            EditorUtility.SetDirty(behaviour.config);
            AssetDatabase.SaveAssets();

            Utils.Random.Seed = 0;
        }

        private static void ConfigureAgentSerializedFields(OneDimReachAgent agent)
        {
            SerializedObject serializedAgent = new SerializedObject(agent);
            serializedAgent.FindProperty("spaceSize").intValue = 1;
            serializedAgent.FindProperty("stackedInputs").intValue = 1;
            serializedAgent.FindProperty("continuousActions").intValue = 1;
            serializedAgent.FindProperty("discreteActions").intValue = 0;
            serializedAgent.FindProperty("archType").enumValueIndex = (int)ArchitectureType.MLP;
            serializedAgent.FindProperty("numLayers").intValue = 2;
            serializedAgent.FindProperty("hidUnits").intValue = 32;
            serializedAgent.FindProperty("activation").enumValueIndex = (int)NonLinearity.Tanh;
            serializedAgent.FindProperty("onEpisodeEnd").enumValueIndex = (int)OnEpisodeEndType.NothingHappens;
            serializedAgent.FindProperty("useSensors").enumValueIndex = (int)UseSensorsType.Off;
            serializedAgent.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void ConfigureBehaviour(AgentBehaviour behaviour, RunSpec spec)
        {
            behaviour.inferenceDevice = Device.CPU;
            behaviour.trainingDevice = spec.trainer == TrainerType.SAC ? Device.GPU : Device.CPU;
            behaviour.targetFPS = 50;
            behaviour.clipping = 5f;
            behaviour.normalize = false;
            behaviour.stochasticity = Stochasticity.FixedStandardDeviation;
            behaviour.standardDeviationValue = 1f;
            behaviour.standardDeviationScale = 1.5f;
            behaviour.noiseValue = 0f;

            Hyperparameters hp = behaviour.config;
            hp.trainer = spec.trainer;
            hp.maxSteps = int.MaxValue;
            hp.actorLearningRate = spec.actorLearningRate;
            hp.criticLearningRate = spec.criticLearningRate;
            hp.gamma = 0.99f;
            hp.LRSchedule = false;
            hp.maxNorm = 0.5f;
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
    }
}
#endif
