using DeepUnity;
using UnityEditor.SceneManagement;
using UnityEngine;


namespace DeepUnityTutorials
{
    public class ShooterScript : Agent
    {
        [Button("GenerateNetworks")]
        [SerializeField] private ShooterScript opponent;
        
        [SerializeField] private GunScript gun;
        [SerializeField] private float speed;
        [SerializeField] private float rot_speed;

   
        private Rigidbody rb;
        private CameraSensor camSensor;

        public void GenerateNetworks()
        {
            // 3 64 64
            var net = new NeuralNetwork(
                    new Conv2D((3, 64, 64), 1, 3),// 1 62 62
                    new MaxPool2D(2), // 1 30 30
                    new LeakyReLU(),
                    new Flatten(),
                    new Dense(961, 128),
                    new LeakyReLU(),
                    new Dense(128, 9),
                    new Softmax()).CreateAsset("shooter_discreteHead");

            net = new NeuralNetwork(
                    new Conv2D((3, 64, 64), 1, 3),// 1 62 62
                    new MaxPool2D(2), // 1 31 31
                    new LeakyReLU(),
                    new Flatten(),
                    new Dense(961, 128),
                    new LeakyReLU(),
                    new Dense(128, 1)).CreateAsset("shooter_critic");
        }
        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
            camSensor = GetComponent<CameraSensor>();
        }

        public override void CollectObservations(StateVector stateBuffer)
        {
            stateBuffer.AddObservation(1);
            var view = camSensor.GetObservationPixels();
            Tensor tensorView = Tensor.Constant(view, (3, 64, 64));
            // stateBuffer.State = tensorView;
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            Vector3 movement;
            switch (actionBuffer.DiscreteAction)
            {
                case 0:
                    movement = transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 1:
                    movement = transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 2:
                    movement = -transform.right * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 3:
                    movement = -transform.forward * speed;
                    rb.velocity = new Vector3(movement.x, rb.velocity.y, movement.z);
                    break;
                case 4:
                    transform.Rotate(0, -rot_speed, 0);
                    break;
                case 5:
                    transform.Rotate(0, rot_speed, 0);
                    break;
                case 6: //Interact
                    gun.Fire();
                    break;
                case 7:
                    gun.Reload();
                    break;
                case 8:
                    break;

            }
        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            int action;

            if (Input.GetKey(KeyCode.A))
                action = 0;
            else if (Input.GetKey(KeyCode.S))
                action = 1;
            else if (Input.GetKey(KeyCode.D))
                action = 2;
            else if (Input.GetKey(KeyCode.W))
                action = 3;
            else if (Input.GetKey(KeyCode.Q))
                action = 4;
            else if (Input.GetKey(KeyCode.E))
                action = 5;
            else if (Input.GetKey(KeyCode.R))
                action = 7;
            else if (Input.GetMouseButton(0))
                action = 6;
            else
                action = 8;

            actionOut.DiscreteAction = action;

        }

        private void OnCollisionEnter(Collision collision)
        {
            // if(collision.collider.CompareTag("Enemy"))
            // {
            //     opponent.AddReward(1);
            //     AddReward(-1);
            //     opponent.EndEpisode();
            //     EndEpisode();
            // }
        }
    }

}


