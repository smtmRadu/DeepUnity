using System;
using System.Collections.Generic;
using System.Linq;
using Unity.VisualScripting;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A board to keep track of the evolution of loss or accuracy.
    /// </summary>
    [Serializable] // must be serializable...
    public class PerformanceGraph
    {
        [ReadOnly, SerializeField] AnimationCurve graph;
        [ReadOnly, SerializeField, Tooltip("The total number of appends.")] int steps;
        [ReadOnly, SerializeField, Tooltip("The value of the last item appended.")] float current;
        [ReadOnly, SerializeField, Tooltip("The mean of all values.")] float mean;

        private float time_step_size = 0.1f;
        private int next_squash = 10;        


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

        /// <summary>
        /// Appends a value to the graph.
        /// </summary>
        /// <param name="graph"></param>
        /// <param name="value"></param>
        /// <returns></returns>
        public static PerformanceGraph operator+(PerformanceGraph graph, float value)
        {
            graph.Append(value);
            return graph;
        }
        /// <summary>
        /// Appends a value to the graph.
        /// </summary>
        /// <param name="value"></param>
        public void Append(float value)
        {         
            current = value;
            graph.AddKey(time_step_size * steps, value);

            // when reaches 1 on X axis, squash them to half.
            if (steps % next_squash == 0)
            {
                // Squash the lengths on X
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

        /// <summary>
        /// Smooths out the keyframes by computing pairwise means of neighbour nodes. 
        /// The nodes are reduced to smooth_factor * total_keyframes number.
        /// </summary>
        /// <param name="smooth_factor">a value in range (0, 1)</param>
        public void Smooth(float smooth_factor)
        {
            if (smooth_factor >= 1f || smooth_factor <= 0.01f)
                throw new ArgumentException($"Smooth factor must be in range (0.01, 1). Received {smooth_factor}.");
            
            var keys = graph.keys.ToList();

            var smoothed_out_keys = new List<Keyframe>();

            for (int i = 0; i < keys.Count; i += (int) (1f / smooth_factor))
            {
                var mean_value = 0f;
                var last_time = 0f;
                int no_in_batch = 0;
                for (int j = 0; j < (int) (1f / smooth_factor); j++)
                {
                    if (i + j == keys.Count)
                        break;

                    no_in_batch++;
                    mean_value += keys[i + j].value;
                    last_time = keys[i + j].time;
                }
                mean_value /= no_in_batch;
                smoothed_out_keys.Add(new Keyframe(last_time, mean_value));
            }

            graph.keys = smoothed_out_keys.ToArray();



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

