#if UNITY_EDITOR
using System;
using System.IO;
using System.Linq;
using DeepUnity.Models;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public static class SacProbeBatchRunner
    {
        private const string DefaultBalanceBallMlpBehaviourPath = "Assets/DeepUnity/Tutorials/BalanceBall/BalanceBallMLP/_BalanceBall.asset";

        public static void RunBalanceBallStochasticProbe()
        {
            try
            {
                RunStochasticProbe(DefaultBalanceBallMlpBehaviourPath, Device.CPU);
                RunStochasticProbe(DefaultBalanceBallMlpBehaviourPath, Device.GPU);
                EditorApplication.Exit(0);
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        public static void RunBalanceBallStochasticProbeCpu()
        {
            try
            {
                RunStochasticProbe(DefaultBalanceBallMlpBehaviourPath, Device.CPU);
                EditorApplication.Exit(0);
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        public static void RunBalanceBallStochasticProbeGpu()
        {
            try
            {
                RunStochasticProbe(DefaultBalanceBallMlpBehaviourPath, Device.GPU);
                EditorApplication.Exit(0);
            }
            catch (Exception ex)
            {
                Debug.LogException(ex);
                EditorApplication.Exit(1);
            }
        }

        private static void RunStochasticProbe(string behaviourPath, Device device)
        {
            AgentBehaviour behaviour = AssetDatabase.LoadAssetAtPath<AgentBehaviour>(behaviourPath);
            if (behaviour == null)
                throw new InvalidOperationException($"Behaviour not found at path: {behaviourPath}");

            RepairBehaviourReferences(behaviour);
            string reportPath = SacStochasticActorGradientProbe.RunForBehaviour(behaviour, device);
            Debug.Log($"[SacProbeBatchRunner] Stochastic SAC probe completed for {behaviour.behaviourName} on {device}. Report: {reportPath}");
        }

        private static void RepairBehaviourReferences(AgentBehaviour behaviour)
        {
            string behaviourAssetPath = AssetDatabase.GetAssetPath(behaviour);
            if (string.IsNullOrWhiteSpace(behaviourAssetPath))
                throw new InvalidOperationException("Behaviour asset path could not be resolved.");

            string directory = Path.GetDirectoryName(behaviourAssetPath)?.Replace('\\', '/');
            if (string.IsNullOrWhiteSpace(directory))
                throw new InvalidOperationException($"Could not resolve behaviour asset directory from path: {behaviourAssetPath}");

            behaviour.config = AssetDatabase.LoadAssetAtPath<Hyperparameters>($"{directory}/Config.asset") ?? behaviour.config;
            behaviour.muNetwork = ReloadNetwork(directory, "Mu", behaviour.muNetwork);
            behaviour.sigmaNetwork = ReloadNetwork(directory, "Sigma", behaviour.sigmaNetwork);
            behaviour.q1Network = ReloadNetwork(directory, "Q1", behaviour.q1Network);
            behaviour.q2Network = ReloadNetwork(directory, "Q2", behaviour.q2Network);

            var missing = behaviour.CheckForMissingAssets();
            if (missing.Count > 0)
            {
                throw new InvalidOperationException($"Behaviour still has missing assets after repair: {string.Join(", ", missing)}");
            }
        }

        private static Sequential ReloadNetwork(string directory, string assetName, Sequential current)
        {
            Sequential reloaded = AssetDatabase.LoadAssetAtPath<Sequential>($"{directory}/{assetName}.asset");
            return reloaded ?? current;
        }
    }
}
#endif
