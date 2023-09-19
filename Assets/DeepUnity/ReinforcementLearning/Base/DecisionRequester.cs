using System.Collections.Generic;
using System.Security.Policy;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// [Check DecisionRequester image in Documentation]
    /// 
    /// What are decisions?
    /// Is said that an agent takes a decision in a fixed frame 'i' when the state[i] is propagated through the neural networks, obtaining action[i].
    /// Each decision is followed by an action immediately, in the same frame.
    /// 
    /// What are actions?
    /// An action is represented by a call of OnActionReceived(). An action uses values selected by a decision. An action is performed always after a decision.
    /// If takeActionsBetweenDecisions is true, OnActionReceived() is called every single frame, using the same decision values until a new decision is made.
    /// 
    /// Note: Trainer copies first agent Decision Requester's data to other parallel agents.
    /// </summary>
    [DisallowMultipleComponent, AddComponentMenu("DeepUnity/Decision Requester")]
    public class DecisionRequester : MonoBehaviour
    {
        
        [Tooltip("The maximum decisions/steps in one agent's episode. Set to a positive integer to limit the episode length to that many steps. Set to 0 for unlimited episode length.")]
        [Min(0)] public int maxStep = 1000;

        [Tooltip("The agents makes a decision once in this frames interval - ActionBuffer actions are resampled. An action is performed immediately afterwards - OnActionReceved() is called.")] 
        [Range(1, 50)] public int decisionPeriod = 1;

        [Tooltip("If true, OnActionReceived() is called every single frame, using the last decision. Otherwise, it is called only after each decision. Has no effect when Decision Period is 1 or in Heuristic Mode.")]
        public bool takeActionsBetweenDecisions = false;

        public bool IsFrameBeforeDecisionFrame(int fixedFramesCount)
        {
            return fixedFramesCount % decisionPeriod == decisionPeriod - 1;
        }
        /// <summary>
        /// Called in FixedUpdate() * X times (but runs logic only once).
        /// </summary>
        /// <returns></returns>
        public AgentEvent RequestEvent(int fixedFramesCount)
        {
            if (fixedFramesCount % decisionPeriod == 0)
                return AgentEvent.DecisionAndAction;

            if (takeActionsBetweenDecisions)
                return AgentEvent.Action;

            return AgentEvent.None;
        }
        public enum AgentEvent
        {
            None,
            Action,
            DecisionAndAction
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

            if(decisionPeriod.intValue == 1)
            {
                dontDrawMe.Add("takeActionsBetweenDecisions");
            }
            

            targetScript.maxStep = EditorGUILayout.IntField("Max Step", targetScript.maxStep);
            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            string decPeriodSec = (decisionPeriod.intValue * Time.fixedDeltaTime).ToString("0.00");
            EditorGUILayout.HelpBox($"The agent takes a decision once every {decPeriodSec} seconds.", MessageType.None);

            dontDrawMe.Add("maxStep");

            serializedObject.Update();
            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}


