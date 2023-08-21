using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Training Statistics")]
    public class TrainingStatistics : MonoBehaviour
    {
        [ReadOnly, Tooltip("Total seconds runned by the agent multipled by the number of parallel agents.")] public float trainingTime = 0f;
        [ReadOnly, Tooltip("Total numbers of steps runned by the agent multiplied by the number of parallel agents.")] public int trainingSteps = 0;
        [ReadOnly, Tooltip("How many policy updates were made.")] public int iterations = 0;
        [Header("Environment")]
        [Tooltip("Cumulated rewards in each episode.")] public PerformanceGraph cumulativeReward = new PerformanceGraph(100);
        [Tooltip("Steps required in each episode.")] public PerformanceGraph episodeLength = new PerformanceGraph(100);
        [Header("Losses")]
        [Tooltip("Mean values of -LCLIP")] public PerformanceGraph policyLoss = new PerformanceGraph(1000);
        public PerformanceGraph valueLoss = new PerformanceGraph(1000);
        [Header("Policy")]
        public PerformanceGraph learningRate = new PerformanceGraph(100);
        public PerformanceGraph epsilon = new PerformanceGraph(100);
    }

    [CustomEditor(typeof(TrainingStatistics)), CanEditMultipleObjects]
    class CustomAgentPerformanceTrackerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            List<string> dontDrawMe = new List<string>() { "m_Script" };


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

          
            serializedObject.ApplyModifiedProperties();
        }
    }

}


