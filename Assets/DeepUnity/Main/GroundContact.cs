using DeepUnity;
using UnityEngine;

namespace DeepUnity
{
    [DisallowMultipleComponent]
    public class GroundContact : MonoBehaviour
    {
        private const string groundTag = "Ground"; // Tag of ground object.

        public Agent agent;
        public bool IsGrounded { get; private set; }

        public bool endEpisodeOnContact = false;
        public float rewardOnContact = 0f;

       
        void OnCollisionEnter(Collision col)
        {
            if (col.transform.CompareTag(groundTag))
            {
                IsGrounded = true;
                agent.AddReward(rewardOnContact);
                

                if (endEpisodeOnContact)
                {
                    agent.EndEpisode();
                }
            }
        }

        void OnCollisionExit(Collision other)
        {
            if (other.transform.CompareTag(groundTag))
            {
                IsGrounded = false;
            }
        }
    }
}