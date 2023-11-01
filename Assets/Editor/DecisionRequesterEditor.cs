using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    [CustomEditor(typeof(DecisionRequester), true), CanEditMultipleObjects]
    sealed class DecisionRequesterEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            DecisionRequester targetScript = (DecisionRequester)target;
            List<string> dontDrawMe = new List<string> { "m_Script", "maxStep" };
            SerializedObject serializedObject = new SerializedObject(targetScript);

            targetScript.maxStep = EditorGUILayout.IntField("Max Step", targetScript.maxStep);

            if (targetScript.maxStep == 0)
            {
                EditorGUILayout.HelpBox("Episode's steps are unlimited.", MessageType.None);
            }

            if (targetScript.maxStep < 0)
            {
                targetScript.maxStep = 0;
            }

            if (serializedObject.FindProperty("decisionPeriod").intValue == 1)
            {
                dontDrawMe.Add("takeActionsBetweenDecisions");
            }


            if (GUI.changed)
            {
                EditorUtility.SetDirty(targetScript);
            }

            EditorGUILayout.LabelField("", GUI.skin.horizontalSlider);

            serializedObject.Update();
            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());

            serializedObject.ApplyModifiedProperties();
        }
    }
}
