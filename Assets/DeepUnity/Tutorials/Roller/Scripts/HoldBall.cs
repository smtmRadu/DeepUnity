using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{

    // After 10 minutes on X8 the agent must show the first big progress (tested on buffer = 2048)
    // After 1 hour on ~X12 he must master the ball.
    public class HoldBall : Agent
    {
        [SerializeField] Rigidbody ball;
        [SerializeField] const float rotationSpeed = 1f;

        public override void Awake()
        {
            base.Awake();
        }
        public override void CollectObservations(StateVector sensorBuffer)
        {
            // 10 observations
            sensorBuffer.AddObservation(transform.rotation);
            sensorBuffer.AddObservation(ball.velocity);
            sensorBuffer.AddObservation(ball.gameObject.transform.position - transform.position);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // 2 continuous actions
            float xRot = actionBuffer.ContinuousActions[0];
            float zRot = actionBuffer.ContinuousActions[1];

            transform.Rotate(new Vector3(1, 0, 0), xRot * rotationSpeed);
            transform.Rotate(new Vector3(0, 0, 1), zRot * rotationSpeed);

            if (ball.gameObject.transform.position.y < transform.position.y)
            {
                EndEpisode();
            }
            else
            {
                SetReward(0.025f);
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
