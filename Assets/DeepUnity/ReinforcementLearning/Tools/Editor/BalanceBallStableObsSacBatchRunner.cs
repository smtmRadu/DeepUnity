#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using System.Reflection;
using DeepUnity.Tutorials;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class BalanceBallStableObsSacBatchRunner
    {
        private const string ScenePath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBall.unity";

        public static void RunBalanceBallStableObsSacDenseUpdates()
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"balanceball_stable_obs_dense_updates_{runId}");
            string behaviourName = $"__BalanceBallStableObsSac_dense_updates_{runId}";

            try
            {
                SetupScene(runId, reportDirectory, behaviourName);
                Debug.Log($"[BalanceBallStableObsSacBatchRunner] Prepared dense_updates run. Report dir: {reportDirectory}");
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

        private static void SetupScene(string runId, string reportDirectory, string behaviourName)
        {
            EditorSceneManager.OpenScene(ScenePath, OpenSceneMode.Single);

            BalanceBall sourceAgent = UnityEngine.Object.FindObjectsOfType<BalanceBall>(true).FirstOrDefault(IsTargetBalanceBallAgent);
            if (sourceAgent == null)
                throw new InvalidOperationException("Could not find the MLP BalanceBall agent in the scene.");

            SerializedObject serializedSourceAgent = new SerializedObject(sourceAgent);
            Rigidbody ball = serializedSourceAgent.FindProperty("ball").objectReferenceValue as Rigidbody;
            if (ball == null)
                throw new InvalidOperationException("BalanceBall source agent is missing its ball reference.");

            foreach (Agent agent in UnityEngine.Object.FindObjectsOfType<Agent>(true))
            {
                agent.behaviourType = BehaviourType.Off;
                agent.enabled = false;
                agent.gameObject.SetActive(false);

                EditorUtility.SetDirty(agent);
            }

            GameObject clone = UnityEngine.Object.Instantiate(sourceAgent.gameObject, sourceAgent.transform.parent);
            clone.name = $"{sourceAgent.gameObject.name}_StableObs";
            clone.transform.SetLocalPositionAndRotation(sourceAgent.transform.localPosition, sourceAgent.transform.localRotation);
            clone.transform.localScale = sourceAgent.transform.localScale;

            foreach (BalanceBall component in clone.GetComponents<BalanceBall>())
                UnityEngine.Object.DestroyImmediate(component, true);

            BalanceBallStableObsAgent stableAgent = clone.AddComponent<BalanceBallStableObsAgent>();
            if (stableAgent == null)
                throw new InvalidOperationException("Failed to add BalanceBallStableObsAgent to cloned agent GameObject.");

            ConfigureStableAgentSerializedFields(stableAgent);
            stableAgent.SetBall(ball);

            DecisionRequester requester = stableAgent.GetComponent<DecisionRequester>();
            if (requester == null)
                throw new InvalidOperationException("Stable BalanceBall agent is missing DecisionRequester.");

            requester.decisionPeriod = 1;
            requester.takeActionsBetweenDecisions = true;
            requester.maxStep = 10_000;

            TrainingStatistics stats = stableAgent.GetComponent<TrainingStatistics>();
            if (stats == null)
                stats = stableAgent.gameObject.AddComponent<TrainingStatistics>();

            BalanceBallBatchPlayController playController = stableAgent.GetComponent<BalanceBallBatchPlayController>();
            if (playController == null)
                playController = stableAgent.gameObject.AddComponent<BalanceBallBatchPlayController>();

            AgentBehaviour behaviour = AgentBehaviour.CreateOrLoadAsset(
                behaviourName,
                stateSize: 6,
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

            ConfigureBehaviour(behaviour);

            stableAgent.model = behaviour;
            stableAgent.behaviourType = BehaviourType.Learn;
            stableAgent.enabled = true;
            clone.SetActive(true);
            playController.Configure(reportDirectory, runId, behaviourName, targetSteps: 5000, timeoutSeconds: 120f);

            EditorUtility.SetDirty(sourceAgent);
            EditorUtility.SetDirty(clone);
            EditorUtility.SetDirty(stableAgent);
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

        private static void ConfigureStableAgentSerializedFields(BalanceBallStableObsAgent agent)
        {
            SetAgentField(agent, "spaceSize", 6);
            SetAgentField(agent, "stackedInputs", 1);
            SetAgentField(agent, "continuousActions", 2);
            SetAgentField(agent, "discreteActions", 0);
            SetAgentField(agent, "archType", ArchitectureType.MLP);
            SetAgentField(agent, "numLayers", 2);
            SetAgentField(agent, "hidUnits", 32);
            SetAgentField(agent, "activation", NonLinearity.Tanh);
            SetAgentField(agent, "onEpisodeEnd", OnEpisodeEndType.ResetEnvironment);
            SetAgentField(agent, "useSensors", UseSensorsType.Off);
        }

        private static void SetAgentField(BalanceBallStableObsAgent agent, string fieldName, object value)
        {
            FieldInfo field = typeof(Agent).GetField(fieldName, BindingFlags.Instance | BindingFlags.NonPublic);
            if (field == null)
                throw new MissingFieldException(typeof(Agent).FullName, fieldName);

            object target = agent;
            if (target == null)
                throw new InvalidOperationException($"Stable agent target was null while setting `{fieldName}`.");

            field.SetValue(target, value);
        }

        private static void ConfigureBehaviour(AgentBehaviour behaviour)
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
            hp.actorLearningRate = 1e-3f;
            hp.criticLearningRate = 1e-3f;
            hp.gamma = 0.99f;
            hp.LRSchedule = false;
            hp.replayBufferSize = 1_000_000;
            hp.minibatchSize = 64;
            hp.updateInterval = 1;
            hp.updateAfter = 1024;
            hp.updatesNum = 1;
            hp.alpha = 0.2f;
            hp.tau = 0.005f;
            hp.timescaleAdjustment = TimescaleAdjustmentType.Constant;
            hp.timescale = 20;
            hp.debug = false;
            hp.sacDebugMetrics = false;
        }
    }
}
#endif
