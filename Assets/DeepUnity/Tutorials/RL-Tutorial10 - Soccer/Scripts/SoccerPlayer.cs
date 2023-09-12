using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class SoccerPlayer : Agent
    {
        [Header("Properties")]
        [SerializeField] public float speed = 5000f;
        [SerializeField] public float rotationSpeed = 5000f;
        [SerializeField] private SoccerPlayer teammate;
        [SerializeField] public PlayerType type;
        [SerializeField] public PlayerTeam team;


        private Rigidbody rb;
        private SoccerEnvironmentScript environment;

        public override void Awake()
        {
            base.Awake();
            environment = transform.parent.GetComponent<SoccerEnvironmentScript>();
            rb = transform.GetComponent<Rigidbody>();
        }
        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            // Total = 25 + 13 + 3 + 9 = 
            // RaySensor info +25
            // We add info about himself +13
            sensorBuffer.AddObservation((int)type);
            sensorBuffer.AddObservation(transform.localPosition.x / 30f);
            sensorBuffer.AddObservation(transform.localPosition.z / 15f);
            sensorBuffer.AddObservation(transform.rotation.normalized);
            sensorBuffer.AddObservation(rb.velocity.normalized);
            sensorBuffer.AddObservation(rb.angularVelocity.normalized);

            // About his teammate +3
            sensorBuffer.AddObservation(teammate.transform.localPosition);

            // About the ball +9
            sensorBuffer.AddObservationRange(environment.GetBallInfo());
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            rb.AddForce(new Vector3(actionBuffer.ContinuousActions[0], 0, actionBuffer.ContinuousActions[1]) * speed);
            rb.AddTorque(new Vector3(0, actionBuffer.ContinuousActions[2] * rotationSpeed, 0));
        }
        public override void Heuristic(ActionBuffer actionOut)
        {
            actionOut.ContinuousActions[0] = Input.GetAxis("Horizontal");
            actionOut.ContinuousActions[1] = Input.GetAxis("Vertical");
            if (Input.GetKey(KeyCode.E))
                actionOut.ContinuousActions[2] = 1f;
            else if (Input.GetKey(KeyCode.Q))
                actionOut.ContinuousActions[2] = -1f;
        }
        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Ball"))
                AddReward(0.0025f);
            else if (collision.collider.CompareTag("Agent"))
                AddReward(-0.0025f);
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


