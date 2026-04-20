using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    /// <summary>
    /// Same BalanceBall control task, but with a smaller, sign-stable observation set.
    /// This is used only as a debugger environment to test whether SAC is failing on raw
    /// quaternion-based observations rather than on the off-policy trainer itself.
    /// </summary>
    public sealed class BalanceBallStableObsAgent : Agent
    {
        [SerializeField] private Rigidbody ball;
        [SerializeField] private float rotationSpeed = 1f;

        public void SetBall(Rigidbody rigidbody)
        {
            ball = rigidbody;
        }

        public override void CollectObservations(StateVector sensorBuffer)
        {
            Vector3 up = transform.up;
            Vector3 velocity = ball.velocity;
            Vector3 rel = ball.transform.position - transform.position;

            sensorBuffer.AddObservation(up.x);
            sensorBuffer.AddObservation(up.z);
            sensorBuffer.AddObservation(velocity.x);
            sensorBuffer.AddObservation(velocity.z);
            sensorBuffer.AddObservation(rel.x);
            sensorBuffer.AddObservation(rel.z);
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            float xRot = actionBuffer.ContinuousActions[0];
            float zRot = actionBuffer.ContinuousActions[1];

            transform.Rotate(new Vector3(1f, 0f, 0f), xRot * rotationSpeed);
            transform.Rotate(new Vector3(0f, 0f, 1f), zRot * rotationSpeed);

            SetReward(0.025f);
            if (ball.gameObject.transform.position.y < transform.position.y)
                EndEpisode();
        }
    }
}
