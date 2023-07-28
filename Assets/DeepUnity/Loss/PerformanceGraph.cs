using System;
using System.Collections.Generic;
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
        [SerializeField] AnimationCurve graph;
        [SerializeField, Tooltip("The total number of appends.")] int steps;
        [SerializeField, Tooltip("The last value appended.")] float current;


        LinkedList<float> nodes;
        int resolution;

        /// <summary>
        /// A board to keep track of the evolution of loss or accuracy.
        /// </summary>
        /// <param name="animationCurve"></param>
        /// <param name="resolution">The number of dots displayed in the animation curve.</param>
        public PerformanceGraph(int resolution = 1000)
        {
            steps = 0;
            graph = new AnimationCurve();
            this.resolution = resolution;
            nodes = new LinkedList<float>();
        }
        public void Append(float value)
        {
            steps++;
            current = value;
            nodes.AddLast(value);

            if (nodes.Count == resolution)
            {
                LinkedList<float> newList = new LinkedList<float>();

                int index = 0;
                float lastNode = -1f;
                foreach (var node in nodes)
                {
                    if (index % 2 == 0)
                        lastNode = node;
                    else
                        newList.AddLast((lastNode + node) / 2f);


                    index++;
                }

                nodes = newList;
            }

            graph.ClearKeys();


            float time_index = 1f;
            foreach (var node in nodes)
            {
                graph.AddKey(time_index / nodes.Count, node);
                time_index += 1f;
            }
        }
        public void Clear()
        {
            current = 0f;
            steps = 0;
            nodes.Clear();        
        }
    }




    [CustomPropertyDrawer(typeof(PerformanceGraph))]
    public class PerformanceGraphDrawer : PropertyDrawer
    {
        public override float GetPropertyHeight(SerializedProperty property, GUIContent label)
        {
            int numFields = 3; // Change this to the number of fields to display (curve, count, current)
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

            SerializedProperty curveProperty = property.FindPropertyRelative("graph");
            SerializedProperty currentProperty = property.FindPropertyRelative("current");
            SerializedProperty countProperty = property.FindPropertyRelative("steps");

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

            EditorGUI.EndProperty();
        }
    }

}

