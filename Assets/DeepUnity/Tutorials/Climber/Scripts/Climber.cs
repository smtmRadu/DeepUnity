using DeepUnity;
using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace Climbing

{
    public class Climber : Agent
    {
        // Spring: 2500 | Damper: 100 | MaxForce: 5000

        [Header("Needs normalization")]
        public GameObject head;
        public GameObject hips;
        public GameObject chest;

        public GameObject armL;
        public GameObject forearmL;
        public GameObject handL;

        public GameObject armR;
        public GameObject forearmR;
        public GameObject handR;

        public GameObject legL;
        public GameObject shinL;
        public GameObject footL;

        public GameObject legR;
        public GameObject shinR;
        public GameObject footR;

        BodyController bodyController;

        [SerializeField, ViewOnly] FixedJoint rightHandJoint;
        [SerializeField, ViewOnly] FixedJoint leftHandJoint;
        [SerializeField, ViewOnly] Rigidbody rightHandStone;
        [SerializeField, ViewOnly] Rigidbody leftHandStone;

        float rewardScale = 0.25f;
        float maxLeftHandYpos;
        float maxRightHandYpos;

        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            // 16 body parts
            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(chest);
            bodyController.AddBodyPart(hips);

            bodyController.AddBodyPart(armL);
            bodyController.AddBodyPart(forearmL);
            bodyController.AddBodyPart(handL);

            bodyController.AddBodyPart(armR);
            bodyController.AddBodyPart(forearmR);
            bodyController.AddBodyPart(handR);

            bodyController.AddBodyPart(legL);
            bodyController.AddBodyPart(shinL);
            bodyController.AddBodyPart(footL);

            bodyController.AddBodyPart(legR);
            bodyController.AddBodyPart(shinR);
            bodyController.AddBodyPart(footR);

            bodyController.bodyPartsDict[handL].ColliderContact.OnEnter = (col) =>
            {
                if(col.collider.CompareTag("Mountain"))
                    leftHandStone = col.collider.attachedRigidbody;
            };
            bodyController.bodyPartsDict[handL].ColliderContact.OnExit = (col) =>
            {
                if (col.collider.CompareTag("Mountain"))
                    leftHandStone = null;

            };

            bodyController.bodyPartsDict[handR].ColliderContact.OnEnter = (col) =>
            {
                if (col.collider.CompareTag("Mountain"))
                    rightHandStone = col.collider.attachedRigidbody;
            };
            bodyController.bodyPartsDict[handR].ColliderContact.OnExit = (col) =>
            {
                if (col.collider.CompareTag("Mountain"))
                    rightHandStone = null;
            };
        }

        public override void OnEpisodeBegin()
        {
            maxLeftHandYpos = handL.transform.position.y;
            maxRightHandYpos = handR.transform.position.y;
        }

        // 24 sensors + 124 = 148
        public override void CollectObservations(StateVector stateBuffer)
        {
            // Hand placement relative to chest
            
            // 10
            stateBuffer.AddObservation(leftHandJoint == null);
            stateBuffer.AddObservation(rightHandJoint == null);
            stateBuffer.AddObservation(leftHandStone == null);
            stateBuffer.AddObservation(rightHandStone = null);

            stateBuffer.AddObservation(chest.transform.position - handL.transform.position);
            stateBuffer.AddObservation(chest.transform.position - handR.transform.position);

            var jdDict = bodyController.bodyPartsDict;

            // 114
            // + 10
            BodyPart _head = jdDict[head];
            {
                stateBuffer.AddObservation(_head.rigidbody.velocity);
                stateBuffer.AddObservation(_head.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_head.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_head.CurrentNormalizedStrength);
            }

            // + 6
            BodyPart _chest = jdDict[chest];
            {
                stateBuffer.AddObservation(_chest.rigidbody.velocity);
                stateBuffer.AddObservation(_chest.rigidbody.angularVelocity);
            }

            // + 10
            BodyPart _torso = jdDict[hips];
            {
                stateBuffer.AddObservation(_torso.rigidbody.velocity);
                stateBuffer.AddObservation(_torso.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_torso.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_torso.CurrentNormalizedStrength);
            }

            // 10 x 2 + 8 x 2 = 36
            BodyPart _armR = jdDict[armR];
            BodyPart _armL = jdDict[armL];
            BodyPart _faR = jdDict[forearmR];
            BodyPart _faL = jdDict[forearmL];
            {
                stateBuffer.AddObservation(_armR.rigidbody.velocity);
                stateBuffer.AddObservation(_armR.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_armR.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_armR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_armL.rigidbody.velocity);
                stateBuffer.AddObservation(_armL.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_armL.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_armL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_faR.rigidbody.velocity);
                stateBuffer.AddObservation(_faR.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_faR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_faL.rigidbody.velocity);
                stateBuffer.AddObservation(_faL.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_faL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faL.CurrentNormalizedStrength);
            }

            //10 x 2 + 8 x 4 = 52
            BodyPart _legR = jdDict[legR];
            BodyPart _legL = jdDict[legL];
            BodyPart _shinR = jdDict[shinR];
            BodyPart _shinL = jdDict[shinL];
            BodyPart _footR = jdDict[footR];
            BodyPart _footL = jdDict[footL];
            {
                stateBuffer.AddObservation(_legR.rigidbody.velocity);
                stateBuffer.AddObservation(_legR.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_legR.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_legR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_legL.rigidbody.velocity);
                stateBuffer.AddObservation(_legL.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_legL.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_legL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_shinR.rigidbody.velocity);
                stateBuffer.AddObservation(_shinR.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_shinL.rigidbody.velocity);
                stateBuffer.AddObservation(_shinL.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_footR.rigidbody.velocity);
                stateBuffer.AddObservation(_footR.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_footR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_footR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_footL.rigidbody.velocity);
                stateBuffer.AddObservation(_footL.rigidbody.angularVelocity);
                stateBuffer.AddObservation(_footL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_footL.CurrentNormalizedStrength);
            }
        }

        // 38 continuous actions
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;

            float[] actions_vector = actionBuffer.ContinuousActions;

            int i = 0;
            jdDict[head].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[hips].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);

            jdDict[armL].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[forearmL].SetJointTargetRotation(actions_vector[i++], 0f, 0f);

            jdDict[armR].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[forearmR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);

            jdDict[legL].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[shinL].SetJointTargetRotation(actions_vector[i++], 0f, 0f);
            jdDict[footL].SetJointTargetRotation(actions_vector[i++], 0f, 0f);

            jdDict[legR].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[shinR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);
            jdDict[footR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);


            jdDict[head].SetJointStrength(actions_vector[i++]);
            jdDict[hips].SetJointStrength(actions_vector[i++]);

            jdDict[armL].SetJointStrength(actions_vector[i++]);
            jdDict[forearmL].SetJointStrength(actions_vector[i++]);

            jdDict[armR].SetJointStrength(actions_vector[i++]);
            jdDict[forearmR].SetJointStrength(actions_vector[i++]);

            jdDict[legL].SetJointStrength(actions_vector[i++]);
            jdDict[shinL].SetJointStrength(actions_vector[i++]);
            jdDict[footL].SetJointStrength(actions_vector[i++]);

            jdDict[legR].SetJointStrength(actions_vector[i++]);
            jdDict[shinR].SetJointStrength(actions_vector[i++]);
            jdDict[footR].SetJointStrength(actions_vector[i++]);

            LeftHandCatch(actions_vector[i++]);
            RightHandCatch(actions_vector[i++]);

            if (leftHandJoint != null)
                AddReward(-0.001f * rewardScale);
            if (rightHandJoint != null)
                AddReward(-0.001f * rewardScale);

            if (transform.position.y < 0f)
            {
                EndEpisode();
                AddReward(-0.1f * rewardScale);
            }
        }

        void LeftHandCatch(float input)
        {
            if (input < 0f) // detach
            {
                Destroy(leftHandJoint);
                leftHandJoint = null;
            }
            else if(leftHandJoint == null && leftHandStone != null) //catch
            {
                leftHandJoint = handL.gameObject.AddComponent<FixedJoint>();
                leftHandJoint.connectedBody = leftHandStone;
                leftHandJoint.breakTorque = 9999f;
                leftHandJoint.breakForce = 9999f;


                if(handL.transform.position.y > maxLeftHandYpos)
                {
                    AddReward((handL.transform.position.y - maxLeftHandYpos) * rewardScale);
                    maxLeftHandYpos = handL.transform.position.y;
                }
            }  
        }

        void RightHandCatch(float input)
        {
            if (input < 0f) // detach
            {
                Destroy(rightHandJoint);
                rightHandJoint = null;
            }
            else if (rightHandJoint == null && rightHandStone != null)
            {
                rightHandJoint = handR.gameObject.AddComponent<FixedJoint>();
                rightHandJoint.connectedBody = rightHandStone;
                rightHandJoint.breakTorque = 9999f;


                if (handR.transform.position.y > maxRightHandYpos)
                {
                    AddReward((handR.transform.position.y - maxRightHandYpos) * rewardScale);
                    maxRightHandYpos = handR.transform.position.y;
                }
            }
        }
    }


}

