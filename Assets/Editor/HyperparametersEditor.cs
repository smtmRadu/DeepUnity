using System.Collections.Generic;
using UnityEditor;

namespace DeepUnity
{
    [CustomEditor(typeof(Hyperparameters), true), CanEditMultipleObjects]
    class ScriptlessHP : Editor
    {
        public override void OnInspectorGUI()
        {
            serializedObject.Update();
            List<string> dontDrawMe = new List<string>() { "m_Script" };

            SerializedProperty lrs = serializedObject.FindProperty("LRSchedule");
            if (!lrs.boolValue)
            {
                dontDrawMe.Add("schedulerStepSize");
                dontDrawMe.Add("schedulerDecay");
            }

            if (serializedObject.FindProperty("KLDivergence").enumValueIndex == (int)KLType.Off)
                dontDrawMe.Add("targetKL");



            // if (EditorApplication.isPlaying)
            //     EditorGUILayout.HelpBox("Hyperparameters values can be modified at runtime. Config file has no effect but when the agent is learning.", MessageType.Info);
            // DO not modify the values at runtime (lr will not change, may appear bugs when changing the buffer size to a smaller size when is already filled).


            DrawPropertiesExcluding(serializedObject, dontDrawMe.ToArray());
            serializedObject.ApplyModifiedProperties();
        }
    }
}
