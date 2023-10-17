using UnityEngine;

namespace DeepUnityTutorials
{
 
    public class GroundContact : MonoBehaviour
    {
        public bool IsGrounded { get; private set; } = false;

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Floor"))
                IsGrounded = true;
        }

        private void OnCollisionExit(Collision collision)
        {
            if (collision.collider.CompareTag("Floor"))
                IsGrounded = false;
        }
    }
}



