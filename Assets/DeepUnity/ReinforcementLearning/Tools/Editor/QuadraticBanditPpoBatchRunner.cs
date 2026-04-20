#if UNITY_EDITOR
using System;
using System.IO;
using DeepUnity.Tutorials;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class QuadraticBanditPpoBatchRunner
    {
        private sealed class RunSpec
        {
            public string scenario;
            public int targetSteps;
            public float timeoutSeconds;
            public int bufferSize;
            public int batchSize;
            public int numEpoch;
            public float actorLearningRate;
            public float criticLearningRate;
            public float standardDeviationValue;
            public int timescale;
        }

        public static void RunQuadraticBanditPpoBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "ppo_baseline",
                targetSteps = 3000,
                timeoutSeconds = 90f,
                bufferSize = 512,
                batchSize = 64,
                numEpoch = 4,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                standardDeviationValue = 1f,
                timescale = 20,
            });
        }

        private static void StartRun(RunSpec spec)
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"quadratic_bandit_{spec.scenario}_{runId}");
            string behaviourName = $"__QuadraticBanditPpo_{spec.scenario}_{runId}";

            try
            {
                SetupScene(spec, runId, reportDirectory, behaviourName);
                Debug.Log($"[QuadraticBanditPpoBatchRunner] Prepared {spec.scenario} run. Report dir: {reportDirectory}");
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
            behaviour.standardDeviationValue = spec.standardDeviationValue;
            behaviour.standardDeviationScale = 1.5f;
            behaviour.noiseValue = 0f;

            Hyperparameters hp = behaviour.config;
            hp.trainer = TrainerType.PPO;
            hp.maxSteps = int.MaxValue;
            hp.actorLearningRate = spec.actorLearningRate;
            hp.criticLearningRate = spec.criticLearningRate;
            hp.gamma = 0.99f;
            hp.LRSchedule = false;
            hp.bufferSize = spec.bufferSize;
            hp.batchSize = spec.batchSize;
            hp.numEpoch = spec.numEpoch;
            hp.horizon = -1;
            hp.beta = 0.0f;
            hp.epsilon = 0.2f;
            hp.lambda = 0.96f;
            hp.valueCoeff = 0.5f;
            hp.maxNorm = 0.5f;
            hp.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            hp.timescale = spec.timescale;
            hp.debug = false;

            EditorUtility.SetDirty(hp);
            EditorUtility.SetDirty(behaviour);
        }
    }
}
#endif
