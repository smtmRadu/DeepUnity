using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{ 
    public class ChooseHighActions : Agent
    {
        public override void CollectObservations(StateBuffer sensorBuffer)
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


