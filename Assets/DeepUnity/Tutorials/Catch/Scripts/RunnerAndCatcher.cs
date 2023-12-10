
using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
   
    public class RunnerAndCatcher : Agent
    {
        public float speed = 9f;
        public float rotationSpeed = 4f;
        public float jumpPower = 9f;
        public Agent opponent;
        [SerializeField] private CatcherOrRunner type;
        private Rigidbody rb;
        private bool isGrounded = false;
        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
        }
        public override void CollectObservations(StateVector stateVector)
        {
            // 5
            Vector3 distance_between = (transform.position - opponent.transform.position) / 50f;
            stateVector.AddObservation(distance_between);
            stateVector.AddObservation((int)type);
            stateVector.AddObservation(isGrounded);
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            Vector3 movement;
            switch (actionBuffer.DiscreteAction)
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
                    if (isGrounded)
                    {
                        rb.velocity = new Vector3(rb.velocity.x, jumpPower, rb.velocity.z);
                        isGrounded = false;
                        AddReward(-0.005f);
                    }
                    break;
                case 6:
                    transform.Rotate(0, rotationSpeed, 0); break;
                case 7:
                    transform.Rotate(0, -rotationSpeed, 0); break;
            }

            if (type == CatcherOrRunner.Runner)
                AddReward(+0.001f);
            else
                AddReward(-0.001f);
        }


        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Ground"))
            {
                isGrounded = true;
            }
            if(collision.collider.CompareTag("Agent"))
            {
                if(type == CatcherOrRunner.Runner)
                {
                    AddReward(-2f);
                    EndEpisode();
                }
                else
                {
                    AddReward(+2f);
                    EndEpisode();
                }
            }
        }





        private enum CatcherOrRunner
        {
            Runner,
            Catcher
        }

    }

    


}

