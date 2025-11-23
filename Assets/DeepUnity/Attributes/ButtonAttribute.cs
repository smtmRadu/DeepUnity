using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Displays a button inspector that executes a public method. 
    /// Usage: Place [Button("Name_of_the_Method</Name>")] attribute over any field of the class.
    /// </summary>
    public class ButtonAttribute : PropertyAttribute
    {
        public string methodToInvoke;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="methodName">The name of the method to be called on click.</param>
        public ButtonAttribute(string methodName)
        {
            methodToInvoke = methodName;
        }
    }
#if UNITY_EDITOR
    [UnityEditor.CustomPropertyDrawer(typeof(ButtonAttribute))]
    public class ButtonAttributeDrawer : UnityEditor.PropertyDrawer
    {
        public override void OnGUI(Rect position, UnityEditor.SerializedProperty property, GUIContent label)
        {

            ButtonAttribute buttonAttribute = attribute as ButtonAttribute;

            if (GUI.Button(position, buttonAttribute.methodToInvoke))
            {
                MonoBehaviour script = property.serializedObject.targetObject as MonoBehaviour;

                if (script != null)
                {
                    System.Reflection.MethodInfo method = script.GetType().GetMethod(buttonAttribute.methodToInvoke);
                    if (method != null)
                    {
                        method.Invoke(script, null);
                    }
                    else
                    {
                        Debug.LogError($"Public method '{buttonAttribute.methodToInvoke}' not found on {script.name} script.");
                    }
                }
            }
            UnityEditor.EditorGUILayout.PropertyField(property);
        }
    }
#endif
}
