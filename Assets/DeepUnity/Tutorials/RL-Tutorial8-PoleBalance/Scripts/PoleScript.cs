using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class PoleScript : Agent
    {
        [SerializeField] Rigidbody @base;
        [SerializeField] Rigidbody weight;
        [SerializeField] private float speed = 1;
        public override void CollectObservations(StateBuffer sensorBuffer)
        {
            sensorBuffer.AddObservation(@base.velocity.normalized);
            sensorBuffer.AddObservation(@base.angularVelocity.normalized);
            sensorBuffer.AddObservation(weight.velocity.normalized);
            sensorBuffer.AddObservation(weight.angularVelocity.normalized);
            sensorBuffer.AddObservation(base.transform.localPosition.x - weight.transform.localPosition.x);

            //13 observations
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // 1 continuous action
            @base.AddForce(new Vector3(actionBuffer.ContinuousActions[0] * speed, 0, 0), ForceMode.Impulse);

            AddReward(0.0025f);
        }

        public override void Heuristic(ActionBuffer actionBuffer)
        {
            actionBuffer.ContinuousActions[0] = Input.GetAxis("Horizontal");
        }

    }
}


