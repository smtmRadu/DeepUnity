using UnityEngine;

namespace DeepUnityTutorials
{
    public class SensitiveBodyPart : MonoBehaviour
    {
        [Header("When this gameobject's collider get hit (typically by the ground), the agent episode ends.")]
        public DeepUnity.Agent agent;

        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Floor"))
                agent.EndEpisode();          
        }
    }
}




