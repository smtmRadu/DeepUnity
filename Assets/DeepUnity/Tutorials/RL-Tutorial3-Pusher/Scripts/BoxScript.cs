using DeepUnity;
using UnityEngine;


namespace DeepUnityTutorials
{
    public class BoxScript : MonoBehaviour
    {
        public Agent partnerAgent;

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Terminal"))
                partnerAgent.AddReward(1f);

            partnerAgent.EndEpisode();
        }
    }

}


