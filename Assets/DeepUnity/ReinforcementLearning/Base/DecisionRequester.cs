using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Decision Requester")]
    public class DecisionRequester : MonoBehaviour
    {
        [SerializeField] public DecisionRequestType actionRequest = DecisionRequestType.EachFrame;
        [Range(0.01f, 20f), Tooltip("The agent performs a value every X seconds.")] public float periodBetweenDecisions = 1f;
        [Tooltip("Use random actions to help the agent explore the environment.")] public bool randomAction = false;

        [HideInInspector] public float timeSinceLastAction = 0f;
        [HideInInspector] public bool performActionForced = false;

        public bool GetPermission()
        {
            switch(actionRequest)
            {
                case DecisionRequestType.EachFrame:
                    return true;
                case DecisionRequestType.OnPeriodInterval:
                    if(timeSinceLastAction >= periodBetweenDecisions || performActionForced)
                    {
                        timeSinceLastAction = 0f;
                        return true;
                    }
                    return false;
                case DecisionRequestType.Manual:
                    if(performActionForced)
                    {
                        performActionForced = false;
                        return true;
                    }    
                    return false;
                default:
                    throw new KeyNotFoundException("Unhandled action request type!");
            }
        }

        private void FixedUpdate()
        {
            if (actionRequest == DecisionRequestType.OnPeriodInterval)
                timeSinceLastAction += Time.fixedDeltaTime;
        }

    }

    [CustomEditor(typeof(DecisionRequester), true), CanEditMultipleObjects]
    sealed class CustomEditorActionRequester : Editor
    {
        public override void OnInspectorGUI()
        {
            // Get a reference to the target script
            DecisionRequester targetScript = (DecisionRequester)target;

            // Create a list of property names that you want to exclude from being drawn
            List<string> dontDrawMe = new List<string> { "m_Script" };

            // Get the serialized object for the target script
            SerializedObject serializedObject = new SerializedObject(targetScript);

            // Get the serialized property for the "actionRequest" field (change "actionRequest" to the actual field name)
            SerializedProperty actionRequestProperty = serializedObject.FindProperty("actionRequest");

            // Check the value of "actionRequest" and modify the dontDrawMe list accordingly
            if (actionRequestProperty.enumValueIndex != (int)DecisionRequestType.OnPeriodInterval)
            {
                dontDrawMe.Add("periodBetweenDecisions");
            }
            if(actionRequestProperty.enumValueIndex == (int)DecisionRequestType.Manual)
            {
                dontDrawMe.Add("actionNoise");
            }

            // Start the Inspector GUI layout
            serializedObject.Update();

            // Draw the property fields excluding the ones in the "dontDrawMe" list
            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

            // Apply any modifications to the serialized properties
            serializedObject.ApplyModifiedProperties();

            // Check if "actionRequest" is of a specific type, and if so, display a HelpBox
            if (actionRequestProperty.enumValueIndex == (int)DecisionRequestType.Manual)
            {
                EditorGUILayout.HelpBox("Actions are performed whenever RequestAction() method is called.", MessageType.None);
            }
        }
    }
}


