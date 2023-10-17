using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class SensitiveBodyPart2D : MonoBehaviour
    {
        [SerializeField] Agent agent;

        private void OnCollisionEnter2D(Collision2D collision)
        {
            if(collision.collider.CompareTag("Floor"))
                 agent.EndEpisode();
        }
    }

}




