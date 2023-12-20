using DeepUnity;
using UnityEngine;

namespace DeepUnityTutorials
{
    public class HumanStandup : Agent
    {
        [Header("Body Parts 16")]
        public Transform head;
        public Transform stomach;
        public Transform hips;
        public Transform chest;

        public Transform armL;
        public Transform forearmL;
        public Transform handL;

        public Transform armR;
        public Transform forearmR;
        public Transform handR;

        public Transform legL;
        public Transform shinL;
        public Transform footL;

        public Transform legR;
        public Transform shinR;
        public Transform footR;

        BodyController bodyController;

        public override void Awake()
        {
            base.Awake();

            bodyController = GetComponent<BodyController>();

            bodyController.AddBodyPart(head);
            bodyController.AddBodyPart(chest);
            bodyController.AddBodyPart(stomach);
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
        }

        // 138 observations
        public override void CollectObservations(StateVector stateBuffer)
        {
            // 13 X 10
            foreach (var bp in bodyController.bodyPartsList)
            {
                if (bp.rb.transform == chest)
                    continue;

                if (bp.rb.transform == handL)
                    continue;

                if (bp.rb.transform == handR)
                    continue;

                // 10 info
                stateBuffer.AddObservation(bp.rb.velocity);
                stateBuffer.AddObservation(bp.rb.angularVelocity);
                stateBuffer.AddObservation(bp.CurrentNormalizedRotation);
                stateBuffer.AddObservation(bp.CurrentNormalizedStrength);
            }

            // 8
            var jdDict = bodyController.bodyPartsDict;

            stateBuffer.AddObservation(jdDict[chest].rb.velocity);
            stateBuffer.AddObservation(jdDict[chest].rb.angularVelocity);

            stateBuffer.AddObservation(jdDict[footL].GroundContact.IsGrounded);
            stateBuffer.AddObservation(jdDict[footR].GroundContact.IsGrounded);
        }

        // 44 continuous actions
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;

            float[] actions_vector = actionBuffer.ContinuousActions;

            jdDict[head].SetJointTargetRotation(actions_vector[0], actions_vector[1], actions_vector[2]);
            jdDict[stomach].SetJointTargetRotation(actions_vector[3], actions_vector[4], actions_vector[5]);
            jdDict[hips].SetJointTargetRotation(actions_vector[6], actions_vector[7], actions_vector[8]);

            jdDict[armL].SetJointTargetRotation(actions_vector[9], actions_vector[10], actions_vector[11]);
            jdDict[forearmL].SetJointTargetRotation(actions_vector[12], 0f, 0f);

            jdDict[armR].SetJointTargetRotation(actions_vector[13], actions_vector[14], actions_vector[15]);
            jdDict[forearmR].SetJointTargetRotation(actions_vector[16], 0f, 0f);

            jdDict[legL].SetJointTargetRotation(actions_vector[17], actions_vector[18], actions_vector[19]);
            jdDict[shinL].SetJointTargetRotation(actions_vector[20], 0f, 0f);
            jdDict[footL].SetJointTargetRotation(actions_vector[21], actions_vector[22], actions_vector[23]);

            jdDict[legR].SetJointTargetRotation(actions_vector[24], actions_vector[25], actions_vector[26]);
            jdDict[shinR].SetJointTargetRotation(actions_vector[27], 0f, 0f);
            jdDict[footR].SetJointTargetRotation(actions_vector[28], actions_vector[29], actions_vector[30]);



            jdDict[head].SetJointStrength(actions_vector[31]);
            jdDict[stomach].SetJointStrength(actions_vector[32]);
            jdDict[hips].SetJointStrength(actions_vector[33]);

            jdDict[armL].SetJointStrength(actions_vector[34]);
            jdDict[forearmL].SetJointStrength(actions_vector[35]);

            jdDict[armR].SetJointStrength(actions_vector[36]);
            jdDict[forearmR].SetJointStrength(actions_vector[37]);

            jdDict[legL].SetJointStrength(actions_vector[38]);
            jdDict[shinL].SetJointStrength(actions_vector[39]);
            jdDict[footL].SetJointStrength(actions_vector[40]);

            jdDict[legR].SetJointStrength(actions_vector[41]);
            jdDict[shinR].SetJointStrength(actions_vector[42]);
            jdDict[footR].SetJointStrength(actions_vector[43]);

            AddReward(head.position.y * 0.01f); // Constant reward for keeping head up

            const float range = 40f;
            if (transform.position.x < -range || transform.position.x > range ||
                transform.position.z < -range || transform.position.z > range ||
                transform.position.y < -range || transform.position.y > range)
                EndEpisode();
        }
    }


}

