using DeepUnity;
using UnityEngine;


namespace DeepUnityTutorials

{
    public class Pusher : Agent
    {
        [Header("Properties")]
        public Transform box;
        public float speed = 3000f;
        public float rotationSpeed = 100f;
        private Rigidbody rb;


        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
        }

        public override void OnEpisodeBegin()
        {
            // Replace box;
            float randx = Utils.Random.Range(6f, 52f);
            float randz = Utils.Random.Range(15f, 48f);
            box.localPosition = new Vector3(randx, box.localPosition.y, randz);
            box.transform.Rotate(0, Utils.Random.Range(0f, 360f), 0);

            // Reposition agent
            randx = Utils.Random.Range(6f, 52f);
            randz = Utils.Random.Range(15f, 48f);
            transform.localPosition = new Vector3(randx, transform.localPosition.y, randz);
            transform.Rotate(0, Utils.Random.Range(0f, 360f), 0);

        }
        public override void CollectObservations(StateBuffer sensorBuffer)
        {
            // + 60
            sensorBuffer.AddObservation(rb.velocity.normalized); // + 3
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            rb.AddForce(new Vector3(actionBuffer.ContinuousActions[0], 0, actionBuffer.ContinuousActions[1]) * speed);
            rb.AddTorque(new Vector3(0, actionBuffer.ContinuousActions[2], 0) * rotationSpeed, ForceMode.Impulse);

            AddReward(-0.0025f);
        }
        public override void Heuristic(ActionBuffer actionBuffer)
        {
            actionBuffer.ContinuousActions[0] = Input.GetAxis("Horizontal");
            actionBuffer.ContinuousActions[1] = Input.GetAxis("Vertical");

            if (Input.GetKey(KeyCode.Q))
                actionBuffer.ContinuousActions[2] = -1f;
            else if (Input.GetKey(KeyCode.E))
                actionBuffer.ContinuousActions[2] = 1f;
            else
                actionBuffer.ContinuousActions[2] = 0f;
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Wall"))
            {
                AddReward(-1f);
                EndEpisode();
            }
        }

        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Box"))
            {
                AddReward(+0.0025f);
            }
        }
    }



}

