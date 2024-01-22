using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class Jumper : Agent
    {
        public float speed = 2.5f;
        public float rotationSpeed = 2.5f;
        public float jumpPower = 4.5f;
        Rigidbody rb;

        public Transform target;
        public Transform box;
        public GameObject bonus;

        public Vector2 agent_spawn_x;
        public Vector2 agent_spawn_z;
        public Vector2 box_spawn_x;
        public Vector2 box_spawn_z;
        public Vector2 target_spawn_x;
        public Vector2 target_spawn_z;
        bool isGrounded = false;

        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
        }

        public override void CollectObservations(StateVector stateBuffer)
        {
            stateBuffer.AddObservation(isGrounded);
        }

        public override void OnEpisodeBegin()
        {

            transform.position = new Vector3(Utils.Random.Range(agent_spawn_x.x, agent_spawn_x.y), transform.position.y, Utils.Random.Range(agent_spawn_z.x, agent_spawn_z.y));
            target.position = new Vector3(Utils.Random.Range(target_spawn_x.x, target_spawn_x.y), target.position.y, Utils.Random.Range(target_spawn_z.x, target_spawn_z.y));
            box.position = new Vector3(Utils.Random.Range(box_spawn_x.x, box_spawn_x.y), box.position.y, Utils.Random.Range(box_spawn_x.x, box_spawn_x.y));

            bonus?.SetActive(true);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            Vector3 movement;
            switch(actionBuffer.DiscreteAction)
            {
                case 0:
                    break; //No Action
                case 1:
                    movement = transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 2:
                    movement = -transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 3:
                    movement = transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 4:
                    movement = -transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z); break;
                case 5:
                    if(isGrounded)
                    {
                        rb.velocity = new Vector3(rb.velocity.x, jumpPower, rb.velocity.z);
                        isGrounded = false;
                        AddReward(-0.005f);
                    }         
                    break;
                case 6:
                    transform.Rotate(0, rotationSpeed, 0);break;
                case 7:
                    transform.Rotate(0, -rotationSpeed, 0); break;

            }

            AddReward(-0.001f); // Existential Reward
        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            if (Input.GetKey(KeyCode.A))
                actionOut.DiscreteAction = 4;
            else if (Input.GetKey(KeyCode.D))
                actionOut.DiscreteAction = 3;
            else if (Input.GetKey(KeyCode.W))
                actionOut.DiscreteAction = 2;
            else if (Input.GetKey(KeyCode.S))
                actionOut.DiscreteAction = 1;
            else if (Input.GetKey(KeyCode.Space))
                actionOut.DiscreteAction = 5;
            else if (Input.GetKey(KeyCode.E))
                actionOut.DiscreteAction = 6;
            else if (Input.GetKey(KeyCode.Q))
                actionOut.DiscreteAction = 7;
            else 
                actionOut.DiscreteAction = 0;

        }


        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Wall"))
            {
                AddReward(-1f);
                EndEpisode();
            }
            if (collision.collider.CompareTag("Ground"))
            {
                isGrounded = true;
            }
        }

        private void OnCollisionStay(Collision collision)
        {
            if (collision.collider.CompareTag("Ground"))
            {
                isGrounded = true;
            }
        }
        private void OnCollisionExit(Collision collision)
        {
            if (collision.collider.CompareTag("Ground"))
            {
                isGrounded = false;
            }
        }

        private void OnTriggerEnter(Collider other)
        {
            if(other.CompareTag("Target"))
            {
                AddReward(1f);
                EndEpisode();
            }
            else if (other.name == "Bonus")
            {
                AddReward(0.5f);
                other.gameObject.SetActive(false);
            }
        }
    }
}



