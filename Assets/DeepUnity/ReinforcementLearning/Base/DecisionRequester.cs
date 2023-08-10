using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Decision Requester")]
    public class DecisionRequester : MonoBehaviour
    {
        [Tooltip("When does the agent performs an action?")]
        [SerializeField] public DecisionRequestType actionRequest = DecisionRequestType.OnceEachFrame;
        [Range(0.01f, 20f), Tooltip("The agent performs a value every X seconds.")] public float periodBetweenDecisions = 1f;
        [Tooltip("Use random actions instead of predictions.")] public bool randomAction = false;

        [HideInInspector] public float timeSinceLastAction = 0f;
        [HideInInspector] public bool TakeActionThisFrame = true;

        /// <summary>
        /// Called in FixedUpdate() * X times (but runs logic only once), Update() and LateUpdate().
        /// </summary>
        /// <returns></returns>
        /// <exception cref="KeyNotFoundException"></exception>
        public bool DoITakeActionThisFrame()
        {
            if (!TakeActionThisFrame)
                return false;

            TakeActionThisFrame = false;

            switch (actionRequest)
            {
                case DecisionRequestType.OnceEachFrame:
                    return true;
                case DecisionRequestType.OnPeriodInterval:
                    return timeSinceLastAction >= periodBetweenDecisions;
                case DecisionRequestType.WhenRequested:
                    return true;
                default:
                    throw new KeyNotFoundException("Unhandled action request type!");
            }
        }

        private void FixedUpdate()
        {
            switch (actionRequest)
            {
                case DecisionRequestType.OnceEachFrame:
                    break;
                case DecisionRequestType.OnPeriodInterval:
                    timeSinceLastAction += Time.fixedDeltaTime;
                    break;
                case DecisionRequestType.WhenRequested:
                    break;
            }
        }

        private void LateUpdate()
        {
            switch(actionRequest)
            {
                case DecisionRequestType.OnceEachFrame:
                    TakeActionThisFrame = true;
                    break;

                case DecisionRequestType.OnPeriodInterval:
                    if (timeSinceLastAction >= periodBetweenDecisions)
                    {
                        timeSinceLastAction = 0f;
                        TakeActionThisFrame = true;
                    }
                    break;
                case DecisionRequestType.WhenRequested: // TakeActionThisFrame is modified by RequestAction() method
                    break;
            }
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
            if(actionRequestProperty.enumValueIndex == (int)DecisionRequestType.WhenRequested)
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
            if (actionRequestProperty.enumValueIndex == (int)DecisionRequestType.WhenRequested)
            {
                EditorGUILayout.HelpBox("Actions are performed whenever RequestAction() method is called.", MessageType.None);
            }
        }
    }
}


