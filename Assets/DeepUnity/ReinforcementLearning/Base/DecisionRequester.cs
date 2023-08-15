using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Decision Requester")]
    public class DecisionRequester : MonoBehaviour
    {
        [Tooltip("When does the agent performs an action?")]
        [SerializeField] public DecisionRequestType performAction = DecisionRequestType.OnceEachFrame;
        [Range(0.01f, 20f), Tooltip("The agent performs a value every X seconds.")] public float periodBetweenDecisions = 1f;
        [Tooltip("Use random actions instead of predictions.")] public bool randomAction = false;

        


        /// <summary>
        /// Handles to take action only once in FixedUpdate().
        /// </summary>
        private FrameState frame = FrameState.AllowedToTakeAction;


        /// <summary>
        /// Handles the OnPeriodInterval case.
        /// </summary>
        private float timeSinceLastAction = 0f;
        /// <summary>
        /// Handles the WhenRequested case.
        /// </summary>
        public bool decisionWasRequested { get; set; } = false;
    



        /// <summary>
        /// Called in FixedUpdate() * X times (but runs logic only once), Update() and LateUpdate().
        /// </summary>
        /// <returns></returns>
        public bool DoITakeActionThisFrame()
        {
            if (frame == FrameState.NotAllowedToTakeAction)
                return false;

            frame = FrameState.NotAllowedToTakeAction;

            switch (performAction)
            {
                case DecisionRequestType.OnceEachFrame:
                    return true;
                case DecisionRequestType.OnPeriodInterval:
                    return timeSinceLastAction >= periodBetweenDecisions;
                case DecisionRequestType.WhenRequested:
                    return decisionWasRequested;
                default:
                    throw new KeyNotFoundException("Unhandled action request type!");
            }
        }



        private void FixedUpdate()
        {
            switch (performAction)
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

        /// <summary>
        /// Permit the action again for the next frame.
        /// </summary>
        private void LateUpdate()
        {
            frame = FrameState.AllowedToTakeAction;

            switch (performAction)
            {
                case DecisionRequestType.OnceEachFrame:         
                    break;

                case DecisionRequestType.OnPeriodInterval:
                    timeSinceLastAction = timeSinceLastAction >= periodBetweenDecisions ? 0f : timeSinceLastAction;
                    break;
                case DecisionRequestType.WhenRequested: // TakeActionThisFrame is modified by RequestAction() method
                    decisionWasRequested = false;
                    break;
            }
        }




        private enum FrameState
        {
            AllowedToTakeAction,
            NotAllowedToTakeAction
        }

    }

    [CustomEditor(typeof(DecisionRequester), true), CanEditMultipleObjects]
    sealed class CustomEditorActionRequester : Editor
    {
        public override void OnInspectorGUI()
        {
            DecisionRequester targetScript = (DecisionRequester)target;
            List<string> dontDrawMe = new List<string> { "m_Script" };
            SerializedObject serializedObject = new SerializedObject(targetScript);
            SerializedProperty performActProperty = serializedObject.FindProperty("performAction");


            if (performActProperty.enumValueIndex != (int)DecisionRequestType.OnPeriodInterval)
            {
                dontDrawMe.Add("periodBetweenDecisions");
            }


            serializedObject.Update();


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

            serializedObject.ApplyModifiedProperties();

            if (performActProperty.enumValueIndex == (int)DecisionRequestType.WhenRequested)
            {
                EditorGUILayout.HelpBox("Actions are performed whenever RequestAction() method is called.", MessageType.None);
            }
        }
    }
}


