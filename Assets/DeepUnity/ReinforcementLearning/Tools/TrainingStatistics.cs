using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Training Statistics")]
    public class TrainingStatistics : MonoBehaviour
    {
        [ReadOnly] public int episodesCompleted = 0;
        [Header("Environment")]
        [Tooltip("Cumulated rewards in each episode.")] public PerformanceGraph cumulativeReward = new PerformanceGraph(100);
        [Tooltip("Steps required in each episode.")] public PerformanceGraph episodeLength = new PerformanceGraph(100);
        [Header("Losses")]
        [Tooltip("Mean values of LCLIP")] public PerformanceGraph objectiveFunction = new PerformanceGraph(1000);
        public PerformanceGraph valueLoss = new PerformanceGraph(1000);
        [Header("Policy")]
        public PerformanceGraph learningRate = new PerformanceGraph(100);
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

