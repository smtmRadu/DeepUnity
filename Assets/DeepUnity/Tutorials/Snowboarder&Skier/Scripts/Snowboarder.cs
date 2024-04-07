using DeepUnity;
using DeepUnity.ReinforcementLearning;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class Snowboarder : Agent
    {
        [Header("Needs normalization")]
        public float survival_reward = 0.001f;
        public GameObject snowboard;
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

        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            // 16 body parts
            bodyController.AddBodyPart(snowboard);
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

            bodyController.bodyPartsList.ForEach(bodypart =>
            {
                if (
                bodypart.gameObject != snowboard.gameObject &&
                bodypart.gameObject != footL.gameObject &&
                bodypart.gameObject != footR.gameObject &&
                bodypart.gameObject != shinL.gameObject &&
                bodypart.gameObject != shinR.gameObject)
                {
                    bodypart.ColliderContact.OnEnter = (col) =>
                    {
                        if (col.collider.CompareTag("Ground"))
                        {
                            EndEpisode();
                        }
                    };
                }
            });
        }

        // 113 input variables
        public override void CollectObservations(StateVector stateBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;

            // We will add all velocities relative to the snowboard

            BodyPart _sb = jdDict[snowboard];
            stateBuffer.AddObservation(_sb.ColliderContact.IsGrounded);

            Vector3 reference_velocity = _sb.rigidbody.velocity;
            Vector3 reference_angular_velocity = _sb.rigidbody.angularVelocity;

            // + 6
            BodyPart _chest = jdDict[chest];
            stateBuffer.AddObservation(_chest.rigidbody.velocity - reference_velocity);
            stateBuffer.AddObservation(_chest.rigidbody.angularVelocity - reference_angular_velocity);

            // + 10
            BodyPart _head = jdDict[head];
            {
                stateBuffer.AddObservation(_head.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_head.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_head.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_head.CurrentNormalizedStrength);
            }

            // + 10
            BodyPart _torso = jdDict[hips];
            {
                stateBuffer.AddObservation(_torso.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_torso.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_torso.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_torso.CurrentNormalizedStrength);
            }

            // + 36
            BodyPart _armR = jdDict[armR];
            BodyPart _armL = jdDict[armL];
            BodyPart _faR = jdDict[forearmR];
            BodyPart _faL = jdDict[forearmL];
            {
                stateBuffer.AddObservation(_armR.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_armR.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_armR.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_armR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_armL.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_armL.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_armL.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_armL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_faR.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_faR.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_faR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_faL.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_faL.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_faL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faL.CurrentNormalizedStrength);
            }

            // + 52
            BodyPart _legR = jdDict[legR];
            BodyPart _legL = jdDict[legL];
            BodyPart _shinR = jdDict[shinR];
            BodyPart _shinL = jdDict[shinL];
            BodyPart _footR = jdDict[footR];
            BodyPart _footL = jdDict[footL];
            {
                // 10
                stateBuffer.AddObservation(_legR.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_legR.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_legR.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_legR.CurrentNormalizedStrength);

                // 10
                stateBuffer.AddObservation(_legL.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_legL.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_legL.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(_legL.CurrentNormalizedStrength);

                // 8
                stateBuffer.AddObservation(_shinR.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_shinR.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedStrength);

                // 8
                stateBuffer.AddObservation(_shinL.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_shinL.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedStrength);

                // 8
                stateBuffer.AddObservation(_footR.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_footR.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_footR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_footR.CurrentNormalizedStrength);

                // 8
                stateBuffer.AddObservation(_footL.rigidbody.velocity - reference_velocity);
                stateBuffer.AddObservation(_footL.rigidbody.angularVelocity - reference_angular_velocity);
                stateBuffer.AddObservation(_footL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_footL.CurrentNormalizedStrength);
            }

            // 8
            stateBuffer.AddObservation(_sb.rigidbody.rotation);
            stateBuffer.AddObservation(_chest.rigidbody.rotation);
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
            jdDict[footL].SetJointTargetRotation(actions_vector[i++], 0f, actions_vector[i++]);

            jdDict[legR].SetJointTargetRotation(actions_vector[i++], actions_vector[i++], actions_vector[i++]);
            jdDict[shinR].SetJointTargetRotation(actions_vector[i++], 0f, 0f);
            jdDict[footR].SetJointTargetRotation(actions_vector[i++], 0f, actions_vector[i++]);


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

            AddReward(survival_reward);
        }
    }


}

