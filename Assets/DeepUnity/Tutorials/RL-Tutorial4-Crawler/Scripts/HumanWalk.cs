using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class HumanWalk : Agent
    {
        [SerializeField] private JointScript leftFoot;
        [SerializeField] private JointScript leftShin;
        [SerializeField] private JointScript leftThigh;
        [SerializeField] private JointScript rightFoot;
        [SerializeField] private JointScript rightShin;
        [SerializeField] private JointScript rightThigh;

        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            throw new System.NotImplementedException();
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // Move Feet
            leftFoot.SetTargetAngularVelocity(actionBuffer.ContinuousActions[0], 0, 0);

        }
        public override void Heuristic(ActionBuffer actionBuffer)
        {
            actionBuffer.ContinuousActions[0] = Input.GetAxis("Horizontal");
            actionBuffer.ContinuousActions[1] = 0f;
            actionBuffer.ContinuousActions[2] = 0f;
        }
    }


}

