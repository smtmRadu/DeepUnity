using DeepUnity.Tutorials;
using UnityEngine;
using UnityEngine.AI;

namespace DeepUnity
{
    public class ZombieScript : MonoBehaviour
    {
        public float health = 1f;
        [HideInInspector] public SurvivorScript survivor;
        public AudioClip breezeOfBlood;
        NavMeshAgent agent;
        AudioSource audioSource;

        private void Awake()
        {
            agent = GetComponent<NavMeshAgent>();
            audioSource = GetComponent<AudioSource>();
        }
        private void FixedUpdate()
        {
            agent.SetDestination(survivor.transform.position);
        }
        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Bullet"))
            {
                audioSource.clip = breezeOfBlood;
                audioSource.Play();
                health -= 1f;

                if(health <= 0f)
                {
                    Destroy(this.gameObject);
                    survivor.AddReward(+0.25f);
                }
            }
        }
    }

}


