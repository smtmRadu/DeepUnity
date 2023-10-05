using UnityEngine;

namespace DeepUnityTutorials
{
 
    public class IsGroundedScript : MonoBehaviour
    {
        public bool IsGrounded { get; private set; } = false;

        private void OnCollisionEnter(Collision collision)
        {
            IsGrounded = true;
        }

        private void OnCollisionExit(Collision collision)
        {
            IsGrounded = false;
        }
    }
}



