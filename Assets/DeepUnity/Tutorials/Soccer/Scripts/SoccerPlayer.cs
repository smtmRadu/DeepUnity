using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class SoccerPlayer : Agent
    {
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
            // rayinfo: 192 + 2 = 194
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

            if (type == PlayerType.Goalie)
                AddReward(0.001f); // Existential bonus
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

        const float goalie_inZone_bonus = 1e-4f;
        public void OnTriggerStay(Collider other)
        {
            if(this.type == PlayerType.Goalie)
            {
                if (this.team == PlayerTeam.Pink && other.name == "PinkGoalArea")
                    AddReward(goalie_inZone_bonus);

                if (this.team == PlayerTeam.Blue && other.name == "BlueGoalArea")
                    AddReward(goalie_inZone_bonus);

            }
        }
    }
}


