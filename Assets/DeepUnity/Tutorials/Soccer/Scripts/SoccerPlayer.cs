using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class SoccerPlayer : Agent
    {
        [Header("Plus 5 additional observations")]
        [SerializeField] public Transform ball;
        [SerializeField] public float speed = 20f;
        [SerializeField] public float rotationSpeed = 10f;
        [SerializeField] public PlayerType type;
        [SerializeField] public PlayerTeam team;

        private Rigidbody rb;

        public override void Awake()
        {
            base.Awake();
            rb = transform.GetComponent<Rigidbody>();
        }
        public override void CollectObservations(StateVector sensorBuffer)
        {
            // Rays are responsible for positioning in the arena (can see goals of each types and walls only)
            // Grid is responsible for other agents and ball
            // 53 + 8 = 61
            sensorBuffer.AddObservation((int)type);
            sensorBuffer.AddObservation((int)team);
            sensorBuffer.AddObservation((transform.position - ball.position).normalized);
            sensorBuffer.AddObservation(transform.forward.normalized); // even if it is not exactly forward it doesn t matter (is the actual right rotation).
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
                    transform.Rotate(0, rotationSpeed, 0); break;
                case 6:
                    transform.Rotate(0, -rotationSpeed, 0); break;

            }

            if (type == PlayerType.Goalie)
                AddReward(0.001f); // Existential bonus
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Ball"))
                AddReward(0.001f);
        }
        public enum PlayerType
        {
            Striker,
            Goalie
        }
        public enum PlayerTeam
        {
            Pink,
            Blue
        }
    }
}


