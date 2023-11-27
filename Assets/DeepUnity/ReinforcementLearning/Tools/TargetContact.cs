using DeepUnity;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// Triggers only for colliders with Target tag. It doesn't apply for triggers.
    /// </summary>
    [DisallowMultipleComponent]
    public class TargetContact : MonoBehaviour
    {
        private const string targetTag = "Target"; // Tag of target object.

        public Agent agent;
        public bool IsTouchingTarget { get; private set; }

        public bool endEpisodeOnContact = false;
        public float rewardOnContact = 0f;

        void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(targetTag))
            {
                IsTouchingTarget = true;
                agent.AddReward(rewardOnContact);
                if (endEpisodeOnContact)
                {
                    agent.EndEpisode();
                }
            }
        }
        void OnCollisionExit(Collision other)
        {
            if (other.transform.CompareTag(targetTag))
            {
                IsTouchingTarget = false;
            }
        }
    }
}