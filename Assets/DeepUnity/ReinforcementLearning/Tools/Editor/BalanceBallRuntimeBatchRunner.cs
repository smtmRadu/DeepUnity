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
    public static class BalanceBallRuntimeBatchRunner
    {
        private const string ScenePath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBall.unity";

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
            public Device inferenceDevice;
            public Device trainingDevice;
            public bool disableBallNoise;
            public TrainerType trainer = TrainerType.SACDepr;
            public int decisionPeriod = 1;
        }

        public static void RunBalanceBallSacBaseline()
        {
            StartRun(new RunSpec
            {
                scenario = "baseline",
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
            });
        }

        public static void RunBalanceBallSacDenseUpdates()
        {
            StartRun(new RunSpec
            {
                scenario = "dense_updates",
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
            });
        }

        public static void RunBalanceBallSacDenseReplay()
        {
            StartRun(new RunSpec
            {
                scenario = "dense_replay",
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 50,
                updateAfter = 1024,
                updatesNum = 8,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
            });
        }

        public static void RunBalanceBallSacDenseUpdatesNoNoise()
        {
            StartRun(new RunSpec
            {
                scenario = "dense_updates_no_noise",
                targetSteps = 5000,
                timeoutSeconds = 120f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
                disableBallNoise = true,
            });
        }

        public static void RunBalanceBallSacGpuDenseUpdates()
        {
            StartRun(new RunSpec
            {
                scenario = "sacgpu_dense_updates",
                targetSteps = 20000,
                timeoutSeconds = 600f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.2f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.GPU,
                trainer = TrainerType.SAC,
            });
        }

        // Reward-scale hypothesis: BalanceBall pays 0.025/step while alpha=0.2 makes the
        // entropy term ~20x the task reward. Canonical-SAC envs pay ~1/step. Scale alpha
        // to the reward magnitude and SAC should finally see the task.
        public static void RunBalanceBallSacGpuLowAlpha()
        {
            StartRun(new RunSpec
            {
                scenario = "sacgpu_low_alpha",
                targetSteps = 20000,
                timeoutSeconds = 600f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.005f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.GPU,
                trainer = TrainerType.SAC,
            });
        }

        // Long-horizon test: with tau=0.005 the TD value needs ~horizon/tau updates to
        // propagate (~17k for an 85-step survival horizon), so 5k-20k step runs end
        // before the hockey stick. 100k steps gives the critic room to converge.
        public static void RunBalanceBallSacGpuLong()
        {
            StartRun(new RunSpec
            {
                scenario = "sacgpu_long",
                targetSteps = 100_000,
                timeoutSeconds = 2400f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.005f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.GPU,
                trainer = TrainerType.SAC,
            });
        }

        // ML-Agents 3DBall parity: decisionPeriod=5 (action repeat -> each decision moves the
        // platform ~5x, much stronger dQ/da) + reward-scaled alpha.
        // (The historical converging runs also used a -1 fall penalty; the env field was
        // removed afterwards at the user's request to keep the PPO-tuned reward.)
        public static void RunBalanceBallSacGpuMlAgentsParity()
        {
            StartRun(new RunSpec
            {
                scenario = "sacgpu_mlagents_parity",
                targetSteps = 40000, // decisions (200k env frames at decisionPeriod=5)
                timeoutSeconds = 2400f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.005f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.GPU,
                trainer = TrainerType.SAC,
                decisionPeriod = 5,
            });
        }

        // Same winning config as RunBalanceBallSacGpuMlAgentsParity but on the CPU SACTrainerDepr.
        public static void RunBalanceBallSacCpuMlAgentsParity()
        {
            StartRun(new RunSpec
            {
                scenario = "saccpu_mlagents_parity",
                targetSteps = 40000,
                timeoutSeconds = 2400f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.005f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
                trainer = TrainerType.SACDepr,
                decisionPeriod = 5,
            });
        }

        public static void RunBalanceBallSacCpuLowAlpha()
        {
            StartRun(new RunSpec
            {
                scenario = "saccpu_low_alpha",
                targetSteps = 20000,
                timeoutSeconds = 600f,
                updateInterval = 1,
                updateAfter = 1024,
                updatesNum = 1,
                minibatchSize = 64,
                replayBufferSize = 1_000_000,
                alpha = 0.005f,
                tau = 0.005f,
                actorLearningRate = 1e-3f,
                criticLearningRate = 1e-3f,
                timescale = 20,
                inferenceDevice = Device.CPU,
                trainingDevice = Device.CPU,
                trainer = TrainerType.SACDepr,
            });
        }

        private static void StartRun(RunSpec spec)
        {
            string runId = DateTime.UtcNow.ToString("yyyyMMdd_HHmmss");
            string reportDirectory = Path.Combine(Directory.GetCurrentDirectory(), "ProbeLogs", $"balanceball_runtime_{spec.scenario}_{runId}");
            string behaviourName = $"__BalanceBallRuntimeSac_{spec.scenario}_{runId}";

            try
            {
                SetupScene(spec, runId, reportDirectory, behaviourName);
                Debug.Log($"[BalanceBallRuntimeBatchRunner] Prepared {spec.scenario} run. Report dir: {reportDirectory}");
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

            requester.decisionPeriod = spec.decisionPeriod;
            requester.takeActionsBetweenDecisions = true;
            requester.maxStep = 10_000;

            if (spec.disableBallNoise)
            {
                foreach (BallNoise noise in balanceBall.GetComponentsInChildren<BallNoise>(true))
                {
                    noise.noise = 0f;
                    EditorUtility.SetDirty(noise);
                }
            }

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
            // MLP and LnMLP are both supported by the CPU trainers and the FullGPU path (GPUMLP handles RMSNorm).
            return (archType == (int)ArchitectureType.MLP || archType == (int)ArchitectureType.LnMLP)
                && stateSize == 10 && continuousActions == 2;
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
            hp.sacDebugMetrics = true;
            hp.sacDebugEveryNUpdates = 25;
        }
    }
}
#endif
