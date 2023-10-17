using UnityEngine;

namespace DeepUnityTutorials
{

    public class GroundContact2D : MonoBehaviour
    {
        // [("Floor must have the Tag Floor. 1 bool value only.")]
        public bool IsGrounded { get; private set; } = false;

        private void OnCollisionEnter2D(Collision2D collision)
        {
            if(collision.collider.CompareTag("Floor"))
                IsGrounded = true;
        }

        private void OnCollisionExit2D(Collision2D collision)
        {
            if (collision.collider.CompareTag("Floor"))
                IsGrounded = false;
        }
    }
}
