using System;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent]
    public class ColliderContact : MonoBehaviour
    {
        /// <summary>
        /// Implement these in order to assign logic to the entrance. The Collision is the object hit.
        /// </summary>


        public Action<Collision> OnEnter;
        public Action<Collision> OnStay;
        public Action<Collision> OnExit;

        /// <summary>
        /// Does this body part touches a GameObject with "Ground" tag?
        /// </summary>
        public bool IsGrounded { get; private set; } = false;


        private void OnCollisionEnter(Collision collision)
        {
            OnEnter?.Invoke(collision);
            if (collision.collider.tag == "Ground")
                IsGrounded = true;
        }
        private void OnCollisionStay(Collision collision)
        {
            OnStay?.Invoke(collision);
        }
        private void OnCollisionExit(Collision collision)
        {
            OnExit?.Invoke(collision);
            if (collision.collider.tag == "Ground")
                IsGrounded = false;
        }
    }
}