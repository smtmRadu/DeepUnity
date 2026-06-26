#if UNITY_EDITOR
using System;
using System.IO;
using DeepUnity.Tutorials;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class QuadraticBanditBatchRunner
    {
        private sealed class RunSpec
        {
            public string scenario;
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
        }

        public static void RunQuadraticBanditSacBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "baseline",
                targetSteps = 3000,
                timeoutSeconds = 90f,
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

        public static void RunQuadraticBanditSacDenseUpdates()
        {
            StartRun(new RunSpec
            {
                scenario = "dense_updates",
                targetSteps = 3000,
                timeoutSeconds = 90f,
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

        public static void RunQuadraticBanditSacDenseReplay()
        {
            StartRun(new RunSpec
            {
                scenario = "dense_replay",
                targetSteps = 3000,
                timeoutSeconds = 90f,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 8,
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
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"quadratic_bandit_{spec.scenario}_{runId}");
            string behaviourName = $"__QuadraticBanditSac_{spec.scenario}_{runId}";

            try
            {
                SetupScene(spec, runId, reportDirectory, behaviourName);
                Debug.Log($"[QuadraticBanditBatchRunner] Prepared {spec.scenario} run. Report dir: {reportDirectory}");
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

            GameObject agentGO = new GameObject("QuadraticBanditAgent");
            QuadraticBanditAgent agent = agentGO.AddComponent<QuadraticBanditAgent>();
            TrainingStatistics stats = agentGO.AddComponent<TrainingStatistics>();
            BalanceBallBatchPlayController playController = agentGO.AddComponent<BalanceBallBatchPlayController>();

            ConfigureAgentSerializedFields(agent);

            DecisionRequester requester = agent.GetComponent<DecisionRequester>();
            if (requester == null)
                throw new InvalidOperationException("DecisionRequester was not auto-created for QuadraticBanditAgent.");

            requester.decisionPeriod = 1;
            requester.takeActionsBetweenDecisions = true;
            requester.maxStep = 1;

            AgentBehaviour behaviour = AgentBehaviour.CreateOrLoadAsset(
                behaviourName,
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
            agent.behaviourType = BehaviourType.Learn;
            agent.enabled = true;

            playController.Configure(reportDirectory, runId, behaviourName, spec.targetSteps, spec.timeoutSeconds);

            EditorUtility.SetDirty(agent);
            EditorUtility.SetDirty(requester);
            EditorUtility.SetDirty(stats);
            EditorUtility.SetDirty(playController);
            EditorUtility.SetDirty(behaviour);
            EditorUtility.SetDirty(behaviour.config);
            AssetDatabase.SaveAssets();

            Utils.Random.Seed = 0;
        }

        private static void ConfigureAgentSerializedFields(QuadraticBanditAgent agent)
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
            serializedAgent.FindProperty("useSensors").enumValueIndex = (int)UseSensorsType.Off;
            serializedAgent.ApplyModifiedPropertiesWithoutUndo();
        }

        private static void ConfigureBehaviour(AgentBehaviour behaviour, RunSpec spec)
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
            hp.trainer = TrainerType.SACDepr;
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
