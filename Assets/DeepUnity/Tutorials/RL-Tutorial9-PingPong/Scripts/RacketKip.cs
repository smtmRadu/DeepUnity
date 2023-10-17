using DeepUnity;
using UnityEngine;


namespace DeepUnityTutorials
{
    public class RacketKip : Agent
    {
        [SerializeField] Rigidbody ball;
        Rigidbody selfrb;
        [SerializeField] float rotationSpeed = 100;
        [SerializeField] float moveSpeed = 100;

        [SerializeField] float outrange = 10f;
        Vector3 initialPosition;
        public override void Awake()
        {
            base.Awake();
            selfrb = GetComponent<Rigidbody>();
            initialPosition = transform.position;
        }
        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            sensorBuffer.AddObservation(transform.rotation);

            sensorBuffer.AddObservation((transform.position - ball.transform.position).normalized);
            sensorBuffer.AddObservation(ball.velocity.normalized);
            sensorBuffer.AddObservation(ball.angularVelocity.normalized);
            sensorBuffer.AddObservation(selfrb.velocity);
            sensorBuffer.AddObservation(selfrb.angularVelocity);

            // 5 * 3 + 4 = 19;

        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            float xrot = actionBuffer.ContinuousActions[0];
            float yrot = actionBuffer.ContinuousActions[1];
            float zrot = actionBuffer.ContinuousActions[2];
            selfrb.AddTorque(new Vector3(xrot, yrot, zrot) * rotationSpeed, ForceMode.Impulse);

            float xmov = actionBuffer.ContinuousActions[3];
            float ymov = actionBuffer.ContinuousActions[4];
            float zmov = actionBuffer.ContinuousActions[5];
            selfrb.AddForce(new Vector3(xmov, ymov, zmov) *  moveSpeed, ForceMode.Impulse);


            if (ball.position.y  < initialPosition.y - outrange)
            {
                AddReward(-1f);
                EndEpisode();
            }
        }


        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Ball"))
                AddReward(0.25f);
        }
    }

}


