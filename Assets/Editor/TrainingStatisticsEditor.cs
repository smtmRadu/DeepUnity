using System.Collections.Generic;
using UnityEditor;

namespace DeepUnity
{
    [CustomEditor(typeof(TrainingStatistics)), CanEditMultipleObjects]
    class CustomAgentPerformanceTrackerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            string[] dontDrawMe = new string[] { "m_Script" };

            /*if(EditorApplication.isPlaying)
            {
                float sessionProgress = script.stepCount / ((float)PPOTrainer.SessionMaxSteps) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Progress [");
                sb.Append(script.stepCount);
                sb.Append(" / ");
                sb.Append(PPOTrainer.SessionMaxSteps);
                sb.Append($"] \n[");
                for (float i = 1.25f; i <= 100f; i += 1.25f)
                {
                    if (i == 47.5f)
                        sb.Append($"{sessionProgress.ToString("00.0")}%");
                    else if (i > 47.5f && i <= 53.75f)
                        continue;
                    else if (i <= sessionProgress)
                        sb.Append("▮");
                    else
                        sb.Append("▯");
                }
                sb.Append("]");
                EditorGUILayout.HelpBox(sb.ToString(), MessageType.None);
                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            }
            */

            DrawPropertiesExcluding(serializedObject, dontDrawMe);

            // Depending on the version, Performance Graph may require or not increasingly needed RAM memory.
            if (EditorApplication.isPlaying)
            {
                EditorGUILayout.HelpBox("Training Statistics may require considerable free RAM for overnight training sessions.", MessageType.Info);
            }

            serializedObject.ApplyModifiedProperties();
        }
    }
}