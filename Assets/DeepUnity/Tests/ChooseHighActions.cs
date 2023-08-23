using UnityEngine;
using DeepUnity;
public class ChooseHighActions : Agent
{
    public override void CollectObservations(SensorBuffer sensorBuffer)
    {
        sensorBuffer.AddObservation(Utils.Random.Value);
        sensorBuffer.AddObservation(Utils.Random.Value);
        sensorBuffer.AddObservation(Utils.Random.Value);
    }

    public override void OnActionReceived(ActionBuffer actionBuffer)
    {
        AddReward(1f / (Mathf.Abs(2f - actionBuffer.ContinuousActions[0])) * 0.02f);
    }
}


