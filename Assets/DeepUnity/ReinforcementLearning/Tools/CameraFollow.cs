using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public class CameraFollow : MonoBehaviour
    {
        public Transform objectToFollow;
        [MinMax(0.001f, 0.999f)]public float smoothness = 0.3f;
        private Vector3 offset;
        private Vector3 speed;
        private void Awake() // It seems that it spawns behind somehow if in start
        {
            if (objectToFollow != null)
                offset = transform.position - objectToFollow.position;
            else
                ConsoleMessage.Warning("Camera cannot follow an object because the target was not assigned");
        }
        private void LateUpdate()
        {
            if (objectToFollow != null)
                transform.position = Vector3.SmoothDamp(transform.position, objectToFollow.position + offset, ref speed, smoothness);
        }
    }
#if UNITY_EDITOR
    [CustomEditor(typeof(CameraFollow), true), CanEditMultipleObjects]
    sealed class CameraFollowEditor : Editor
    {
        string[] drawNow = new string[] { "m_Script" };
        public override void OnInspectorGUI()
        {        
            DrawPropertiesExcluding(serializedObject, drawNow);
            serializedObject.ApplyModifiedProperties();
            serializedObject.Update();
        }
    }
#endif
}


