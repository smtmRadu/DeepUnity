using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// An attribute to keep a value in a specific range, without slider.
    /// </summary>
    public class MinMaxAttribute : PropertyAttribute
    {
        public float minValue;
        public float maxValue;

        public MinMaxAttribute(float minValue, float maxValue)
        {
            this.minValue = minValue;
            this.maxValue = maxValue;
        }
    }

#if UNITY_EDITOR
    [UnityEditor.CustomPropertyDrawer(typeof(MinMaxAttribute))]
    public class MinMaxDrawer : UnityEditor.PropertyDrawer
    {
        public override void OnGUI(Rect position,
                                   UnityEditor.SerializedProperty property,
                                   GUIContent label)
        {
            MinMaxAttribute minMaxAttribute = attribute as MinMaxAttribute;

            if (property.propertyType == SerializedPropertyType.Float)
            {
                property.floatValue = Mathf.Clamp(property.floatValue, minMaxAttribute.minValue, minMaxAttribute.maxValue);
            }
            else if (property.propertyType == SerializedPropertyType.Integer)
            {
                property.intValue = Mathf.Clamp(property.intValue, Mathf.FloorToInt(minMaxAttribute.minValue), Mathf.FloorToInt(minMaxAttribute.maxValue));
            }
            else
            {
                EditorGUI.LabelField(position, label.text, "Use MinMaxAttribute with float or int.");
            }

            EditorGUI.PropertyField(position, property, label, true);
        }
    }
#endif


}
