using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnity.Tutorials
{
    public class RoboticArm : Agent
    {
        public GameObject arm1;
        public GameObject arm2;
        public GameObject arm3;
        public GameObject claw1;
        public GameObject claw2;
        public GameObject claw3;
        BodyController bodyController;

        public override void Awake()
        {
            base.Awake();
            bodyController = GetComponent<BodyController>();

            bodyController.AddBodyPart(arm1);
            bodyController.AddBodyPart(arm2);
            bodyController.AddBodyPart(arm3);
            bodyController.AddBodyPart(claw1);
            bodyController.AddBodyPart(claw2);
            bodyController.AddBodyPart(claw3);

            foreach (var item in bodyController.bodyPartsList)
            {
                item.SetJointStrength(1f);
            }
        }

        public override void CollectObservations(StateVector stateVector)
        {
            // 6 parts x 7 - 2 = 40 inputs
            foreach (var item in bodyController.bodyPartsList)
            {
                stateVector.AddObservation(item.rigidbody.velocity);
                stateVector.AddObservation(item.rigidbody.angularVelocity);

                if (item.gameObject == arm1)               
                    stateVector.AddObservation(item.CurrentEulerRotation);
                else
                    stateVector.AddObservation(item.CurrentEulerRotation.x);
            }
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            var continuousActions = actionBuffer.ContinuousActions;

            int i = 0; // 9 actions
            bodyController.bodyPartsDict[arm1].SetJointTargetRotation(continuousActions[i++], continuousActions[i++], continuousActions[i++]);
            bodyController.bodyPartsDict[arm2].SetJointTargetRotation(continuousActions[i++], 0f, 0f);
            bodyController.bodyPartsDict[arm3].SetJointTargetRotation(continuousActions[i++], 0f, 0f);
            bodyController.bodyPartsDict[claw1].SetJointTargetRotation(continuousActions[i++], 0f, 0f);
            bodyController.bodyPartsDict[claw2].SetJointTargetRotation(continuousActions[i++], 0f, 0f);
            bodyController.bodyPartsDict[claw3].SetJointTargetRotation(continuousActions[i++], 0f, 0f);
            bodyController.bodyPartsDict[arm2].SetJointTargetRotation(continuousActions[i++], 0f, 0f);
        }
    }


}

