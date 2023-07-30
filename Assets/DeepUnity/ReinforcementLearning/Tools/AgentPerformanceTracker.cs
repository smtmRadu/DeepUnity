using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Performance Track")]
    public class AgentPerformanceTracker : MonoBehaviour
    {
        [ReadOnly] public int episodesCompleted = 0;
        [Tooltip("Cumulated rewards in each episode.")]public PerformanceGraph rewards = new PerformanceGraph(100);
        [Tooltip("Steps required in each episode.")]public PerformanceGraph steps = new PerformanceGraph(100);
        public PerformanceGraph criticLoss = new PerformanceGraph(1000);
    }

    [CustomEditor(typeof(AgentPerformanceTracker)), CanEditMultipleObjects]
    class CustomAgentPerformanceTrackerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            // EditorGUILayout.HelpBox("Training information for each episode.", MessageType.None);
            List<string> dontDrawMe = new List<string>() { "m_Script" };


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

          
            serializedObject.ApplyModifiedProperties();
        }
    }

}


