using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class BreakoutBallScript : MonoBehaviour
    {
        public Breakout agent;
        private Rigidbody rb;
        
        private void Awake()
        {
            rb = GetComponent<Rigidbody>();
        }

        private void FixedUpdate()
        {
            rb.velocity = rb.velocity.normalized *agent.ballSpeed;
        }
        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Box"))
            {
                agent.AddReward(+1);
                collision.collider.gameObject.SetActive(false);
            }
            else if (collision.collider.CompareTag("Ground"))
            {
                agent.AddReward(-1);
                agent.EndEpisode();
            }
            else if (collision.collider.CompareTag("Wall"))
            {
                rb.velocity = new Vector3(rb.velocity.x / 1.5f, rb.velocity.y);
            }
            else if(collision.collider.CompareTag("Empty")) // i mean ceil here but i added no other tag
            {
                rb.velocity = new Vector3(rb.velocity.x, rb.velocity.y / 1.5f); // add some sort of lateral movement when touching the ceil
            }
        }
    }

}


