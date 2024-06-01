using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class SoccerballScript : MonoBehaviour
    {
        public Agent agent;

        private void FixedUpdate()
        {
            if (transform.position.y < 0)
                agent.EndEpisode();
        }
        private void OnTriggerEnter(Collider other)
        {
            if(other.CompareTag("Target"))
            {
                agent.AddReward(1f);
                agent.EndEpisode();
            }
        }
    }

}


