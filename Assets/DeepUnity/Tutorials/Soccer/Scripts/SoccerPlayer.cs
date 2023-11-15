using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class SoccerPlayer : Agent
    {
        [SerializeField] public float speed = 2.5f;
        [SerializeField] public float rotationSpeed = 2.5f;
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
        public override void CollectObservations(StateBuffer sensorBuffer)
        {
            // rayinfo: 140 + 2 = 142
            sensorBuffer.AddObservation((int)type);
            sensorBuffer.AddObservation((int)team);
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

            if (type == PlayerType.Striker)
                AddReward(-0.0025f);
        }
        private void OnCollisionEnter(Collision collision)
        {
            if (type == PlayerType.Striker && collision.collider.CompareTag("Ball"))
                AddReward(0.0025f);
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

        private void OnCollisionStay(Collision collision)
        {
            if(type == PlayerType.Goalie)
            {
                if (team == PlayerTeam.Pink && collision.collider.name == "PinkGoalArea")
                    AddReward(+0.0025f);
                else if (team == PlayerTeam.Blue && collision.collider.name == "BlueGoalArea")
                    AddReward(+0.0025f);
            }
        }
    }
}


