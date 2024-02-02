using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Displays the variable in inspector restricting the ability to modify it.
    /// </summary>
    public class ViewOnlyAttribute : PropertyAttribute
    {

    }

#if UNITY_EDITOR
    [UnityEditor.CustomPropertyDrawer(typeof(ViewOnlyAttribute))]
    public class ReadOnlyDrawer : UnityEditor.PropertyDrawer
    {
        public override float GetPropertyHeight(UnityEditor.SerializedProperty property,
                                                GUIContent label)
        {
            return UnityEditor.EditorGUI.GetPropertyHeight(property, label, true);
        }

        public override void OnGUI(Rect position,
                                   UnityEditor.SerializedProperty property,
                                   GUIContent label)
        {
            GUI.enabled = false;
            UnityEditor.EditorGUI.PropertyField(position, property, label, true);
            GUI.enabled = true;
        }
    }
#endif
}
