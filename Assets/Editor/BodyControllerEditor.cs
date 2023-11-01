using DeepUnityTutorials;
using UnityEditor;

namespace DeepUnity
{
    [CustomEditor(typeof(BodyController), true), CanEditMultipleObjects]
    sealed class BodyControllerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            string[] dontDrawMe = new string[] { "m_Script" };

            serializedObject.Update();
            DrawPropertiesExcluding(serializedObject, dontDrawMe);

            serializedObject.ApplyModifiedProperties();
        }
    }
}


