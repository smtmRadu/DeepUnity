#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using DeepUnity.Tutorials;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class BalanceBallPpoBatchRunner
    {
        private const string ScenePath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBall.unity";

        private sealed class RunSpec
        {
            public string scenario;
            public int targetSteps;
            public float timeoutSeconds;
            public int bufferSize;
            public int batchSize;
            public int numEpoch;
            public int horizon;
            public float beta;
            public float epsilon;
            public float lambda;
            public float valueCoeff;
            public float maxNorm;
            public float actorLearningRate;
            public float criticLearningRate;
            public int timescale;
            public Device inferenceDevice;
            public Device trainingDevice;
        }

        public static void RunBalanceBallPpoBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "ppo_baseline",
                targetSteps = 6144,
                timeoutSeconds = 120f,
                bufferSize = 2048,
                batchSize = 128,
                numEpoch = 8,
                horizon = 256,
                beta = 5e-3f,
                epsilon = 0.2f,
                lambda = 0.96f,
                valueCoeff = 0.5f,
                maxNorm = 0.5f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
            });
        }

        private static void StartRun(RunSpec spec)
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"balanceball_{spec.scenario}_{runId}");
            string behaviourName = $"__BalanceBallPpo_{spec.scenario}_{runId}";

            try
            {
                SetupScene(spec, runId, reportDirectory, behaviourName);
                Debug.Log($"[BalanceBallPpoBatchRunner] Prepared {spec.scenario} run. Report dir: {reportDirectory}");
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
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            BalanceBall balanceBall = UnityEngine.Object.FindObjectsOfType<BalanceBall>(true).FirstOrDefault(IsTargetBalanceBallAgent);
            if (balanceBall == null)
                throw new InvalidOperationException("Could not find the MLP BalanceBall agent in the scene.");

            foreach (Agent agent in UnityEngine.Object.FindObjectsOfType<Agent>(true))
            {
                bool keep = agent == balanceBall;
                agent.behaviourType = keep ? BehaviourType.Learn : BehaviourType.Off;
                agent.enabled = keep;
                if (!keep)
                    agent.gameObject.SetActive(false);

                EditorUtility.SetDirty(agent);
            }

            DecisionRequester requester = balanceBall.GetComponent<DecisionRequester>();
            if (requester == null)
                throw new InvalidOperationException("BalanceBall agent is missing DecisionRequester.");

            requester.decisionPeriod = 1;
            requester.takeActionsBetweenDecisions = true;
            requester.maxStep = 10_000;

            TrainingStatistics stats = balanceBall.GetComponent<TrainingStatistics>();
            if (stats == null)
                stats = balanceBall.gameObject.AddComponent<TrainingStatistics>();

            BalanceBallBatchPlayController playController = balanceBall.GetComponent<BalanceBallBatchPlayController>();
            if (playController == null)
                playController = balanceBall.gameObject.AddComponent<BalanceBallBatchPlayController>();

            AgentBehaviour behaviour = CreateFreshBehaviourFromAgent(balanceBall, behaviourName);
            ConfigureBehaviour(behaviour, spec);

            balanceBall.model = behaviour;
            balanceBall.behaviourType = BehaviourType.Learn;
            balanceBall.enabled = true;
            playController.Configure(reportDirectory, runId, behaviourName, spec.targetSteps, spec.timeoutSeconds);

            EditorUtility.SetDirty(balanceBall);
            EditorUtility.SetDirty(requester);
            EditorUtility.SetDirty(stats);
            EditorUtility.SetDirty(playController);
            EditorUtility.SetDirty(behaviour);
            EditorUtility.SetDirty(behaviour.config);
            AssetDatabase.SaveAssets();

            Utils.Random.Seed = 0;
        }

        private static bool IsTargetBalanceBallAgent(BalanceBall agent)
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
            if (source != null)
            {
                fresh.targetFPS = source.targetFPS;
                fresh.clipping = source.clipping;
                fresh.normalize = false;
                fresh.stochasticity = source.stochasticity;
                fresh.standardDeviationValue = source.standardDeviationValue;
                fresh.standardDeviationScale = source.standardDeviationScale;
                fresh.noiseValue = source.noiseValue;
            }

            return fresh;
        }

        private static void ConfigureBehaviour(AgentBehaviour behaviour, RunSpec spec)
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
            hp.horizon = spec.horizon;
            hp.beta = spec.beta;
            hp.epsilon = spec.epsilon;
            hp.lambda = spec.lambda;
            hp.valueCoeff = spec.valueCoeff;
            hp.maxNorm = spec.maxNorm;
            hp.earlyStopping = EarlyStopType.Off;
            hp.targetKL = 0.015f;
            hp.normalizeAdvantages = true;
            hp.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            hp.timescale = spec.timescale;
            hp.debug = false;
        }
    }
}
#endif
