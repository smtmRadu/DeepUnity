using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class HoldBall : Agent
    {
        [SerializeField] Rigidbody ball;
        [SerializeField] float rotationSpeed = 1f;
        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            // 10 observations
            sensorBuffer.AddObservation(transform.rotation.normalized);
            sensorBuffer.AddObservation(ball.velocity.normalized);
            sensorBuffer.AddObservation(ball.gameObject.transform.position - transform.position);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // 2 continuous actions
            float xRot = actionBuffer.ContinuousActions[0];
            float zRot = actionBuffer.ContinuousActions[1];

            // if (transform.rotation.x > -0.25f && transform.rotation.x < 0.25f)
            transform.Rotate(new Vector3(1, 0, 0), xRot * rotationSpeed);

            //if (transform.rotation.z > -0.25f && transform.rotation.z < 0.25f)
            transform.Rotate(new Vector3(0, 0, 1), zRot * rotationSpeed);

            if (ball.gameObject.transform.position.y < transform.position.y)
            {
                SetReward(-1f);

                EndEpisode();
            }
            else
            {
                SetReward(0.1f);
            }
        }

        public override void Heuristic(ActionBuffer actionBuffer)
        {
            float xRot = 0f;
            float zRot = 0f;

            if (Input.GetKey(KeyCode.D))
                xRot = 1f;
            else if (Input.GetKey(KeyCode.A))
                xRot = -1f;

            if (Input.GetKey(KeyCode.W))
                zRot = 1f;
            else if (Input.GetKey(KeyCode.S))
                zRot = -1f;

            actionBuffer.ContinuousActions[0] = xRot;
            actionBuffer.ContinuousActions[1] = zRot;
        }
    }



}
