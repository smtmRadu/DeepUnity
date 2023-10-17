using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Displays a button inspector that executes a public method. 
    /// Usage: Place [Button("_Name of the Method_")] attribute over any field of the class.
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

    [CustomPropertyDrawer(typeof(ButtonAttribute))]
    public class ButtonAttributeDrawer : PropertyDrawer
    {
        public override void OnGUI(Rect position, SerializedProperty property, GUIContent label)
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
            EditorGUILayout.PropertyField(property);
        }
    }

}


