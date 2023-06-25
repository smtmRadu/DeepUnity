using UnityEngine;
using DeepUnity;

namespace kbRadu
{
    public class AgentTest : Agent
    {
        [Header("Properties")]

        public float speed;
        public Transform target;

        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            sensorBuffer.AddObservation(transform.localPosition.x);
            sensorBuffer.AddObservation(transform.localPosition.z);
            sensorBuffer.AddObservation(target.transform.localPosition.x);
            sensorBuffer.AddObservation(target.transform.localPosition.y);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            float xmov = actionBuffer.ContinuousActions[0];
            float zmov = actionBuffer.ContinuousActions[1];

            transform.position += new Vector3(xmov, 0, zmov) * Time.deltaTime * speed;
        }

        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Target"))
            {
                AddReward(1f);
                EndEpisode();
            }    
            else if(collision.collider.CompareTag("Wall"))
            {
                AddReward(-1f);
                EndEpisode();
            }
        }
    }
}

