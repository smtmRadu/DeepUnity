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
    public static class BalanceBallTransitionAuditBatchRunner
    {
        private const string ScenePath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBall.unity";

        public static void RunSingleAgentAudit()
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"balanceball_transition_audit_{runId}");
            string behaviourName = $"__BalanceBallTransitionAudit_{runId}";

            try
            {
                SetupScene(reportDirectory, behaviourName);
                string reportPath = Path.Combine(reportDirectory, "report.md");

                BalanceBall balanceBall = UnityEngine.Object.FindObjectsOfType<BalanceBall>(true).FirstOrDefault(IsTargetBalanceBallAgent);
                if (balanceBall == null)
                    throw new InvalidOperationException("Could not find the target BalanceBall agent.");

                BalanceBallTransitionAuditController audit = balanceBall.GetComponent<BalanceBallTransitionAuditController>();
                if (audit == null)
                    audit = balanceBall.gameObject.AddComponent<BalanceBallTransitionAuditController>();

                audit.Configure(balanceBall, reportPath, targetTransitions: 32, timeoutSeconds: 120f);

                EditorUtility.SetDirty(audit);
                AssetDatabase.SaveAssets();

                Debug.Log($"[BalanceBallTransitionAuditBatchRunner] Prepared audit run {runId}. Report: {reportPath}");
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

        private static void SetupScene(string reportDirectory, string behaviourName)
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            BalanceBall balanceBall = UnityEngine.Object.FindObjectsOfType<BalanceBall>(true).FirstOrDefault(IsTargetBalanceBallAgent);
            if (balanceBall == null)
                throw new InvalidOperationException("Could not find the target BalanceBall agent in the scene.");

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

            AgentBehaviour behaviour = CreateFreshBehaviourFromAgent(balanceBall, behaviourName);
            ConfigureBehaviourForAudit(behaviour);

            balanceBall.model = behaviour;
            balanceBall.behaviourType = BehaviourType.Learn;
            balanceBall.enabled = true;

            Directory.CreateDirectory(reportDirectory);
            EditorUtility.SetDirty(balanceBall);
            EditorUtility.SetDirty(requester);
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
                fresh.inferenceDevice = Device.CPU;
                fresh.trainingDevice = Device.CPU;
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

        private static void ConfigureBehaviourForAudit(AgentBehaviour behaviour)
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
            hp.actorLearningRate = 1e-3f;
            hp.criticLearningRate = 1e-3f;
            hp.gamma = 0.99f;
            hp.LRSchedule = false;
            hp.maxNorm = 0.5f;
            hp.replayBufferSize = 1_000_000;
            hp.minibatchSize = 64;
            hp.updateInterval = 1_000_000;
            hp.updateAfter = int.MaxValue / 4;
            hp.updatesNum = 1;
            hp.alpha = 0.2f;
            hp.tau = 0.005f;
            hp.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            hp.timescale = 20f;
            hp.sacDebugMetrics = false;
        }
    }
}
#endif
