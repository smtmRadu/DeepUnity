using UnityEngine;

namespace DeepUnityTutorials
{
    public class WeightScript : MonoBehaviour
    {
        [SerializeField] PoleScript poleAgent;

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Floor"))
                poleAgent.EndEpisode();
        }
    }
}



