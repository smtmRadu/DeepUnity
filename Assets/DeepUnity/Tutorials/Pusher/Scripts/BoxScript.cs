using DeepUnity.ReinforcementLearning;
using UnityEngine;


namespace DeepUnity.Tutorials
{
    public class BoxScript : MonoBehaviour
    {
        public Agent partnerAgent;

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Target"))
                partnerAgent.AddReward(1f);

            partnerAgent.EndEpisode();
        }
    }

}


