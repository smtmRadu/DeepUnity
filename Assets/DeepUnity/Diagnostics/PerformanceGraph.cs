using System;
using System.Collections.Generic;
using System.Security.Cryptography;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A board to keep track of the evolution of loss or accuracy.
    /// </summary>
    [Serializable]
    public class PerformanceGraph
    {
        [ReadOnly, SerializeField] AnimationCurve graph;
        [ReadOnly, SerializeField, Tooltip("The total number of appends.")] int steps;
        [ReadOnly, SerializeField, Tooltip("The value of the last item appended.")] float current;
        [ReadOnly, SerializeField, Tooltip("The mean of all values.")] float mean;

        float time_step_size = 0.1f;
        int next_squash = 10;

        /// <summary>
        /// A board to keep track of the evolution of loss or accuracy.
        /// </summary>
        /// <param name="animationCurve"></param>
        /// <param name="resolution">The number of dots displayed in the animation curve.</param>
        public PerformanceGraph()
        {
            steps = 0;
            graph = new AnimationCurve();
        }
        public void Append(float value)
        {
           
            current = value;
            graph.AddKey(time_step_size * steps, value);

            // when reaches 1, squash them to half.
            if (steps % next_squash == 0)
            {
                next_squash *= 2;
                time_step_size = 1f / next_squash;


                Keyframe[] keys = graph.keys;
                for (int i = 0; i < keys.Length; i++)
                {
                    keys[i].time = time_step_size * i;
                }
                graph.keys = keys;
                
            }

            
            steps++;

            // set mean value
            mean = mean * (steps - 1f) / steps + value / steps; 
        }
        public void Clear()
        {
            current = 0f;
            steps = 0;
            mean = 0f;
        }
        public Keyframe[] Keys { get => graph.keys; }
    }




    [CustomPropertyDrawer(typeof(PerformanceGraph))]
    public class PerformanceGraphDrawer : PropertyDrawer
    {
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            int numFields = 4; // Change this to the number of fields to display (curve, count, current, mean)
            float lineHeight = EditorGUIUtility.singleLineHeight;
            float spacing = EditorGUIUtility.standardVerticalSpacing;
            return EditorGUIUtility.singleLineHeight + (numFields * lineHeight) + ((numFields - 1) * spacing);
        }

        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
        {
            EditorGUI.BeginProperty(position, label, property);

            SerializedProperty parentProperty = property.serializedObject.FindProperty(property.propertyPath);

            Rect labelRect = new Rect(position.x, position.y, EditorGUIUtility.labelWidth, EditorGUIUtility.singleLineHeight);
            EditorGUI.LabelField(labelRect, parentProperty.displayName);

            Rect curveRect = new Rect(position.x + EditorGUIUtility.labelWidth, position.y + EditorGUIUtility.singleLineHeight, position.width - EditorGUIUtility.labelWidth, EditorGUIUtility.singleLineHeight);
            Rect currentRect = new Rect(position.x + EditorGUIUtility.labelWidth, position.y + 2 * EditorGUIUtility.singleLineHeight + EditorGUIUtility.standardVerticalSpacing, position.width - EditorGUIUtility.labelWidth, EditorGUIUtility.singleLineHeight);
            Rect countRect = new Rect(position.x + EditorGUIUtility.labelWidth, position.y + 3 * EditorGUIUtility.singleLineHeight + 2 * EditorGUIUtility.standardVerticalSpacing, position.width - EditorGUIUtility.labelWidth, EditorGUIUtility.singleLineHeight);
            Rect meanRect = new Rect(position.x + EditorGUIUtility.labelWidth, position.y + 4 * EditorGUIUtility.singleLineHeight + 3 * EditorGUIUtility.standardVerticalSpacing, position.width - EditorGUIUtility.labelWidth, EditorGUIUtility.singleLineHeight);
            
            SerializedProperty curveProperty = property.FindPropertyRelative("graph");
            SerializedProperty currentProperty = property.FindPropertyRelative("current");
            SerializedProperty countProperty = property.FindPropertyRelative("steps");
            SerializedProperty meanProperty = property.FindPropertyRelative("mean");

            if (curveProperty.animationCurveValue == null)
            {
                EditorGUI.LabelField(curveRect, new GUIContent("Here the graph will be displayed."));
            }
            else
            {
                EditorGUI.CurveField(curveRect, curveProperty, Color.green, new Rect());
            }

            EditorGUI.PropertyField(currentRect, currentProperty);
            EditorGUI.PropertyField(countRect, countProperty);
            EditorGUI.PropertyField(meanRect, meanProperty);
            EditorGUI.EndProperty();
        }
    }

}

