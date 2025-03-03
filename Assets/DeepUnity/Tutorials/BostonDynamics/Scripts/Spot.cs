using DeepUnity.ReinforcementLearning;
using System;
using UnityEngine;


namespace DeepUnity.Tutorials
{
    public class Spot : Agent
    {
        [Header("Reguires observations normalization, 100 fps")]
        [SerializeField, ViewOnly] float currentXPosition;
        public GameObject chest;
        public GameObject leftArm;
        public GameObject leftForearm;
        public GameObject leftHand;
        public GameObject rightArm;
        public GameObject rightForearm;
        public GameObject rightHand;
        public GameObject leftLeg;
        public GameObject leftShin;
        public GameObject leftFoot;
        public GameObject rightLeg;
        public GameObject rightShin;
        public GameObject rightFoot;

        BodyController bodyController;

        
        public override void Awake()
        {
            base.Awake();
            bodyController = GetComponent<BodyController>();

            bodyController.AddBodyPart(chest); 
            bodyController.AddBodyPart(leftArm); // % 1
            bodyController.AddBodyPart(leftForearm); // %2
            bodyController.AddBodyPart(leftHand); // % 3
            bodyController.AddBodyPart(rightArm);
            bodyController.AddBodyPart(rightForearm);
            bodyController.AddBodyPart(rightHand);
            bodyController.AddBodyPart(leftLeg);
            bodyController.AddBodyPart(leftShin);
            bodyController.AddBodyPart(leftFoot);
            bodyController.AddBodyPart(rightLeg);
            bodyController.AddBodyPart(rightShin);
            bodyController.AddBodyPart(rightFoot);

            Action<Collision> touch_gr = (col) =>
            {
                if (col.collider.CompareTag("Ground"))
                {
                    EndEpisode();
                }
            };
            bodyController.bodyPartsDict[chest].ColliderContact.OnEnter = touch_gr;

            bodyController.bodyPartsDict[leftArm].ColliderContact.OnEnter = touch_gr;
            bodyController.bodyPartsDict[leftShin].ColliderContact.OnEnter = touch_gr;
            bodyController.bodyPartsDict[leftLeg].ColliderContact.OnEnter = touch_gr;
            bodyController.bodyPartsDict[leftForearm].ColliderContact.OnEnter = touch_gr;

            bodyController.bodyPartsDict[rightArm].ColliderContact.OnEnter = touch_gr;
            bodyController.bodyPartsDict[rightShin].ColliderContact.OnEnter = touch_gr;
            bodyController.bodyPartsDict[rightLeg].ColliderContact.OnEnter = touch_gr;
            bodyController.bodyPartsDict[rightForearm].ColliderContact.OnEnter = touch_gr;
        }

        public override void OnEpisodeBegin()
        {
            currentXPosition = transform.position.x;
        }
        public override void CollectObservations(StateVector stateVector)
        {
            BodyPart _chest = bodyController.bodyPartsList[0];

            // 76
            for (int i = 1; i < bodyController.bodyPartsList.Count; i++)
            {
                // 4 x 10
                if(i%3 == 1)
                {
                    stateVector.AddObservation(_chest.rigidbody.velocity - bodyController.bodyPartsList[i].rigidbody.velocity);
                    stateVector.AddObservation(_chest.rigidbody.angularVelocity - bodyController.bodyPartsList[i].rigidbody.angularVelocity);
                    stateVector.AddObservation(bodyController.bodyPartsList[i].CurrentNormalizedEulerRotation);
                    stateVector.AddObservation(bodyController.bodyPartsList[i].CurrentNormalizedStrength);
                }
                // 4 x 8
                else if(i%3 == 2)
                {
                    stateVector.AddObservation(_chest.rigidbody.velocity - bodyController.bodyPartsList[i].rigidbody.velocity);
                    stateVector.AddObservation(_chest.rigidbody.angularVelocity - bodyController.bodyPartsList[i].rigidbody.angularVelocity);
                    stateVector.AddObservation(bodyController.bodyPartsList[i].CurrentNormalizedEulerRotation.x);
                    stateVector.AddObservation(bodyController.bodyPartsList[i].CurrentNormalizedStrength);
                }  
                // 4 x 1
                else
                {
                    stateVector.AddObservation(bodyController.bodyPartsList[i].ColliderContact.IsGrounded);
                }
            }
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // 24 actions
            float[] actions_vector = actionBuffer.ContinuousActions;

            int a = 0;
            for (int i = 0; i < bodyController.bodyPartsList.Count; i++)
            {
                // 4 x 4
                if (i % 3 == 1)
                {
                    bodyController.bodyPartsList[i].SetJointTargetRotation(actions_vector[a++], actions_vector[a++], actions_vector[a++]);
                    bodyController.bodyPartsList[i].SetJointStrength(actions_vector[a++]);
                }
                // 4 x 2
                else if (i % 3 == 2)
                { 
                    bodyController.bodyPartsList[i].SetJointTargetRotation(actions_vector[a++], 0, 0);
                    bodyController.bodyPartsList[i].SetJointStrength(actions_vector[a++]);
                }
            }

            float reward = transform.position.x - currentXPosition;
            AddReward(0.1f * reward);
            currentXPosition += reward;

            if (transform.position.y < -10f)
                EndEpisode();
        }
    }
}



