using System;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// Triggers only for trigger colliders with Target tag.
    /// </summary>
    [DisallowMultipleComponent]
    public class TriggerContact : MonoBehaviour
    {
        /// <summary>
        /// Implement this in order to assign logic to the entrance. The collider is the object hit.
        /// </summary>
        public Action<Collider> OnEnter;
        public Action<Collider> OnStay;
        public Action<Collider> OnExit;


        private void OnTriggerEnter(Collider other)
        {
            OnEnter?.Invoke(other);
        }
        private void OnTriggerStay(Collider other)
        {
            OnStay?.Invoke(other);
        }
        private void OnTriggerExit(Collider other)
        {
            OnExit?.Invoke(other);
        }

    }
}