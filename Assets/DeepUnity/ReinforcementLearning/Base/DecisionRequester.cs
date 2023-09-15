using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Decision Requester")]
    public class DecisionRequester : MonoBehaviour
    {
        
        [Tooltip("The maximum length of an agent's episode. Set to a positive integer to limit the episode length to that many steps. Set to 0 for unlimited episode length.")]
        [Min(0)] public int maxStep = 1000;

        [Range(1, 50), Tooltip("The agent performs an action every X (fixed) frames.")] public int decisionPeriod = 1;

        /// <summary>
        /// Handles to take action only once in FixedUpdate().
        /// </summary>
        private FrameState frame = FrameState.AllowedToTakeAction;

        public bool IsLastFrameBeforeNextAction(int fixedFramesCount)
        {
            return fixedFramesCount % decisionPeriod == decisionPeriod - 1;
        }
        /// <summary>
        /// Called in FixedUpdate() * X times (but runs logic only once), Update() and LateUpdate().
        /// </summary>
        /// <returns></returns>
        public bool TryRequestDecision(int fixedFramesCount)
        {
            if (frame == FrameState.NotAllowedToTakeAction)
                return false;

            frame = FrameState.NotAllowedToTakeAction;

            return fixedFramesCount % decisionPeriod == 0;
        }


        private void LateUpdate()
        {
            frame = FrameState.AllowedToTakeAction;
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
            SerializedProperty decisionPeriod = serializedObject.FindProperty("decisionPeriod");
            if (targetScript.maxStep == 0)
            {
                EditorGUILayout.HelpBox("Episode's steps are unlimited.", MessageType.None);
            }

            

            targetScript.maxStep = EditorGUILayout.IntField("Max Step", targetScript.maxStep);
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            string seconds = (decisionPeriod.intValue * Time.fixedDeltaTime).ToString("0.00");
            EditorGUILayout.HelpBox($"The agent performs an action once every {seconds} seconds.", MessageType.None);

            dontDrawMe.Add("maxStep");

            serializedObject.Update();
            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


