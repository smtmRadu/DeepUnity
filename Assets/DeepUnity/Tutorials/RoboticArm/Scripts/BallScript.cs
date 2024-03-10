using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class BallScript : MonoBehaviour
    {
        public Agent agent;

        private void FixedUpdate()
        {
            if (transform.position.y < -1)
                agent.EndEpisode();
        }


        private void OnTriggerExit(Collider other)
        {
            if (other.CompareTag("0"))
                agent.AddReward(1f);
        }

        private void OnTriggerEnter(Collider other)
        {
            if(other.CompareTag("1"))
            {
                agent.AddReward(1f);
                agent.EndEpisode();
            }
        }
    }
}



