using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Decision Requester")]
    public class DecisionRequester : MonoBehaviour
    {
        
        [Tooltip("The maximum length of an agent's episode. Set to a positive integer to limit the episode length to that many steps. Set to 0 for unlimited episode length.")]
        [Min(1)] public int maxStep = 1000;

        [Tooltip("When does the agent performs an action?")]
        [SerializeField] public DecisionRequestType performAction = DecisionRequestType.OnceEachFrame;
        [Range(0.01f, 20f), Tooltip("The agent performs an action every X seconds.")] public float periodBetweenDecisions = 0.5f;

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

            targetScript.maxStep = EditorGUILayout.IntField("Max Step", targetScript.maxStep);
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            dontDrawMe.Add("maxStep");
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


