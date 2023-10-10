using UnityEngine;
using DeepUnity;

namespace DeepUnityTutoriale
{ 
    public class ChooseHighActions : Agent
    {
        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            sensorBuffer.AddObservation(Utils.Random.Range(-1f, 1f));
            sensorBuffer.AddObservation(Utils.Random.Range(-1f, 1f));
            sensorBuffer.AddObservation(Utils.Random.Range(-1f, 1f));
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            AddReward(1f / Mathf.Abs(2f - actionBuffer.ContinuousActions[0]));
        }
    }
}


