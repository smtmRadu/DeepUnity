using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public class CameraFollow : MonoBehaviour
    {
        public Transform objectToFollow;
        public float smoothness = 0.3f;
        private Vector3 offset;
        private Vector3 speed;
        private void Start()
        {
            offset = transform.position - objectToFollow.position;
        }
        private void LateUpdate()
        {
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


