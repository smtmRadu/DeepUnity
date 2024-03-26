using UnityEngine;
using DeepUnity.ReinforcementLearning;
using System;

namespace DeepUnityTutorials
{
    public class RobotScript : Agent
    {
        [Header("Apply normalization")]
        public Transform bonus;

        public GameObject head;
        public GameObject neck;
        public GameObject body;

        public GameObject leftEar;
        public GameObject rightEar;

        public GameObject leftThigh;
        public GameObject leftShin;
        public GameObject leftFoot;

        public GameObject rightThigh;
        public GameObject rightShin;
        public GameObject rightFoot;

        BodyController bodyController;

        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(neck);
            bodyController.AddBodyPart(body);

            bodyController.AddBodyPart(leftEar);
            bodyController.AddBodyPart(rightEar);

            bodyController.AddBodyPart(leftThigh);
            bodyController.AddBodyPart(leftShin);
            bodyController.AddBodyPart(leftFoot);

            bodyController.AddBodyPart(rightThigh);
            bodyController.AddBodyPart(rightShin);
            bodyController.AddBodyPart(rightFoot);

            Action<Collision> hitGround = (col) =>
            {
                if (col.collider.CompareTag("Ground"))
                {
                    AddReward(-1f);
                    EndEpisode();
                }
            };

            Action<Collider> hitTarget = (col) =>
            {
                if (col.CompareTag("Target"))
                {
                    AddReward(+0.5f);
                }

                col.gameObject.SetActive(false);
            };

            bodyController.bodyPartsDict[body].ColliderContact.OnEnter = hitGround;
            bodyController.bodyPartsDict[leftThigh].ColliderContact.OnEnter = hitGround;
            bodyController.bodyPartsDict[rightThigh].ColliderContact.OnEnter = hitGround;

            bodyController.bodyPartsDict[body].TriggerContact.OnEnter = hitTarget;
        }
        public override void OnEpisodeBegin()
        {
            if (bonus == null)
                return;

            for (int i = 0; i < bonus.childCount; i++)
            {
                var child = bonus.GetChild(i);
                child.gameObject.SetActive(true);
            }
        }
        public override void CollectObservations(StateVector stateVector)
        {
            var jdict = bodyController.bodyPartsDict;
            // total - 14 + 10 + 8 + 54 = 86

            // + 14
            BodyPart _body = jdict[body];
            {
                stateVector.AddObservation(_body.gameObject.transform.rotation);
                stateVector.AddObservation(_body.rigidbody.velocity);
                stateVector.AddObservation(_body.rigidbody.angularVelocity);
                stateVector.AddObservation(_body.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(_body.CurrentNormalizedStrength);
            }

            // + 10
            BodyPart _head = jdict[head];
            {
                stateVector.AddObservation(_head.rigidbody.velocity);
                stateVector.AddObservation(_head.rigidbody.angularVelocity);
                stateVector.AddObservation(_head.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(_head.CurrentNormalizedStrength);
            }

            // +8
            BodyPart _neck = jdict[head];
            {
                stateVector.AddObservation(_neck.rigidbody.velocity);
                stateVector.AddObservation(_neck.rigidbody.angularVelocity);
                stateVector.AddObservation(_neck.CurrentNormalizedEulerRotation.x);
                stateVector.AddObservation(_neck.CurrentNormalizedStrength);
            }

            // + 54
            BodyPart _legR = jdict[rightThigh];
            BodyPart _legL = jdict[leftThigh];
            BodyPart _shinR = jdict[rightShin];
            BodyPart _shinL = jdict[leftShin];
            BodyPart _footR = jdict[rightFoot];
            BodyPart _footL = jdict[leftFoot];
            {
                stateVector.AddObservation(_legR.rigidbody.velocity);
                stateVector.AddObservation(_legR.rigidbody.angularVelocity);
                stateVector.AddObservation(_legR.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(_legR.CurrentNormalizedStrength);
                     
                stateVector.AddObservation(_legL.rigidbody.velocity);
                stateVector.AddObservation(_legL.rigidbody.angularVelocity);
                stateVector.AddObservation(_legL.CurrentNormalizedEulerRotation);
                stateVector.AddObservation(_legL.CurrentNormalizedStrength);
                     
                stateVector.AddObservation(_shinR.rigidbody.velocity);
                stateVector.AddObservation(_shinR.rigidbody.angularVelocity);
                stateVector.AddObservation(_shinR.CurrentNormalizedEulerRotation.x);
                stateVector.AddObservation(_shinR.CurrentNormalizedStrength);
                 
                stateVector.AddObservation(_shinL.rigidbody.velocity);
                stateVector.AddObservation(_shinL.rigidbody.angularVelocity);
                stateVector.AddObservation(_shinL.CurrentNormalizedEulerRotation.x);
                stateVector.AddObservation(_shinL.CurrentNormalizedStrength);
                     
                stateVector.AddObservation(_footR.rigidbody.velocity);
                stateVector.AddObservation(_footR.rigidbody.angularVelocity);
                stateVector.AddObservation(_footR.CurrentNormalizedEulerRotation.x);
                stateVector.AddObservation(_footR.CurrentNormalizedStrength);
                stateVector.AddObservation(_footR.ColliderContact.IsGrounded);
                    
                stateVector.AddObservation(_footL.rigidbody.velocity);
                stateVector.AddObservation(_footL.rigidbody.angularVelocity);
                stateVector.AddObservation(_footL.CurrentNormalizedEulerRotation.x);
                stateVector.AddObservation(_footL.CurrentNormalizedStrength);
                stateVector.AddObservation(_footL.ColliderContact.IsGrounded);
            }
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;

            float[] actions_vector = actionBuffer.ContinuousActions;

            int i = 0;
            // 26 actions

            jdDict[head].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[neck].SetJointTargetRotation(actions_vector[i++], 0, 0);

            jdDict[leftEar].SetJointTargetRotation(actions_vector[i++], 0, 0);
            jdDict[rightEar].SetJointTargetRotation(actions_vector[i++], 0, 0);

            jdDict[leftThigh].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[rightThigh].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);

            jdDict[leftShin].SetJointTargetRotation(actions_vector[i++], 0, 0);
            jdDict[rightShin].SetJointTargetRotation(actions_vector[i++], 0, 0);

            jdDict[leftFoot].SetJointTargetRotation(actions_vector[i++], 0, 0);
            jdDict[rightFoot].SetJointTargetRotation(actions_vector[i++], 0, 0);


            jdDict[head].SetJointStrength(actions_vector[i++]);
            jdDict[neck].SetJointStrength(actions_vector[i++]);

            jdDict[leftEar].SetJointStrength(actions_vector[i++]);
            jdDict[rightEar].SetJointStrength(actions_vector[i++]);

            jdDict[leftThigh].SetJointStrength(actions_vector[i++]);
            jdDict[rightThigh].SetJointStrength(actions_vector[i++]);

            jdDict[leftShin].SetJointStrength(actions_vector[i++]);
            jdDict[rightShin].SetJointStrength(actions_vector[i++]);

            jdDict[leftFoot].SetJointStrength(actions_vector[i++]);
            jdDict[rightFoot].SetJointStrength(actions_vector[i++]);


            AddReward(+0.0025f);
        }
    }
}



