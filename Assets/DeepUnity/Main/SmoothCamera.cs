using UnityEngine;

namespace DeepUnity
{
    public class SmoothCamera : MonoBehaviour
    {
        public Transform toFollow;
        public float smoothness = 0.3f;
        private Vector3 offset;
        private Vector3 speed;
        private void Start()
        {
            offset = transform.position - toFollow.position;
        }
        private void LateUpdate()
        {
            transform.position = Vector3.SmoothDamp(transform.position, toFollow.position + offset, ref speed, smoothness);
        }
    }
}


