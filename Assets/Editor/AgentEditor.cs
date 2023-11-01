using System.Text;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [CustomEditor(typeof(Agent), true), CanEditMultipleObjects]
    sealed class CustomAgentEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            string[] drawNow = new string[] { "m_Script" };

            var script = (Agent)target;

            // Runtime Learn displays
            if (EditorApplication.isPlaying &&
                script.behaviourType == BehaviourType.Learn
                && script.enabled
                && script.model)
            {
                // Draw Step
                int currentStep = script.EpisodeStepCount;
                string stepcount = $"Decisions [{currentStep}]";
                EditorGUILayout.HelpBox(stepcount, MessageType.None);

                // Draw Reward
                string cumReward = $"Cumulative Reward [{script.EpsiodeCumulativeReward}]";
                EditorGUILayout.HelpBox(cumReward, MessageType.None);

                // Draw buffer 
                float bufferFillPercentage = script.Memory.Count * PPOTrainer.ParallelAgentsCount / ((float)script.model.config.bufferSize) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Buffer [");
                sb.Append(script.Memory.Count * PPOTrainer.ParallelAgentsCount);
                sb.Append(" / ");
                sb.Append(script.model.config.bufferSize);
                sb.Append($"] \n[");
                for (float i = 1.25f; i <= 100f; i += 1.25f)
                {
                    if (i == 47.5f)
                        sb.Append($"{bufferFillPercentage.ToString("00.0")}%");
                    else if (i > 47.5f && i <= 53.75f)
                        continue;
                    else if (i <= bufferFillPercentage)
                        sb.Append("▮");
                    else
                        sb.Append("▯");
                }
                sb.Append("]");
                EditorGUILayout.HelpBox(sb.ToString(), MessageType.None);
                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

                // EditorGUILayout.HelpBox($"Reward [{script.EpsiodeCumulativeReward}]",
                //                         MessageType.None);
            }

            // Runtime Heuristic displays
            if (EditorApplication.isPlaying &&
                script.behaviourType == BehaviourType.Heuristic
                && script.enabled
                && script.model)
            {
                EditorGUILayout.HelpBox("Control the agent behaviour until the buffer is fullfiled.", MessageType.None);
                // Draw buffer 
                float bufferFillPercentage = script.Memory.Count / ((float)script.model.config.bufferSize) * 100f;
                StringBuilder sb = new StringBuilder();
                sb.Append("Buffer [");
                sb.Append(script.Memory.Count);
                sb.Append(" / ");
                sb.Append(script.model.config.bufferSize);
                sb.Append($"] \n[");
                for (float i = 1.25f; i <= 100f; i += 1.25f)
                {
                    if (i == 47.5f)
                        sb.Append($"{bufferFillPercentage.ToString("00.0")}%");
                    else if (i > 47.5f && i <= 53.75f)
                        continue;
                    else if (i <= bufferFillPercentage)
                        sb.Append("▮");
                    else
                        sb.Append("▯");
                }
                sb.Append("]");
                EditorGUILayout.HelpBox(sb.ToString(), MessageType.None);
                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            }


            // On model create field draw
            if (serializedObject.FindProperty("model").objectReferenceValue == null)
            {
                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);



                EditorGUILayout.BeginHorizontal();
                if (GUILayout.Button("Bake model", GUILayout.Width(EditorGUIUtility.labelWidth * 2.32f)))
                {
                    script.BakeModel();
                }

                // Create a Rect for the second field with a specific width
                Rect propertyFieldRect = GUILayoutUtility.GetRect(0, EditorGUIUtility.singleLineHeight);
                propertyFieldRect.width = 50; // Adjust the width as needed

                SerializedProperty typeProperty = serializedObject.FindProperty("archType");
                EditorGUI.BeginDisabledGroup(true);
                EditorGUI.PropertyField(propertyFieldRect, typeProperty, GUIContent.none);
                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();




                EditorGUILayout.BeginHorizontal();
                GUILayout.Label("Num Layers", GUILayout.Width(EditorGUIUtility.labelWidth / 1.08f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("numLayers"), GUIContent.none, GUILayout.Width(50f));
                GUILayout.Label("Hidden Units", GUILayout.Width(EditorGUIUtility.labelWidth / 1.08f));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("hidUnits"), GUIContent.none, GUILayout.Width(50f));
                EditorGUILayout.EndHorizontal();


                EditorGUILayout.Space();

                EditorGUILayout.LabelField("Observations");
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                EditorGUILayout.PrefixLabel("Space Size");
                EditorGUILayout.PropertyField(serializedObject.FindProperty("spaceSize"), GUIContent.none);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                if (serializedObject.FindProperty("archType").enumValueIndex == (int)ModelType.RNN)
                    EditorGUILayout.PrefixLabel("Sequence Length");
                else
                    EditorGUILayout.PrefixLabel("Stacked Inputs");
                EditorGUI.BeginDisabledGroup(true);
                EditorGUILayout.PropertyField(serializedObject.FindProperty("stackedInputs"), GUIContent.none);
                EditorGUI.EndDisabledGroup();
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.Space(5);

                EditorGUILayout.LabelField("Actions");
                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                EditorGUILayout.PrefixLabel("Continuous Actions");
                EditorGUILayout.PropertyField(serializedObject.FindProperty("continuousActions"), GUIContent.none);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.BeginHorizontal();
                EditorGUILayout.Space(20);
                EditorGUILayout.PrefixLabel("Discrete Actions");
                EditorGUILayout.PropertyField(serializedObject.FindProperty("discreteActions"), GUIContent.none);
                EditorGUILayout.EndHorizontal();

                EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            }


            EditorGUILayout.PropertyField(serializedObject.FindProperty("model"));
            EditorGUILayout.PropertyField(serializedObject.FindProperty("behaviourType"));
            if (script.behaviourType != BehaviourType.Off)
            {
                EditorGUILayout.PropertyField(serializedObject.FindProperty("onEpisodeEnd"));
                EditorGUILayout.PropertyField(serializedObject.FindProperty("useSensors"));
            }

            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);
            DrawPropertiesExcluding(serializedObject, drawNow);
            serializedObject.ApplyModifiedProperties();


        }
    }
}

