using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class HumanWalk : Agent
    {
        // Spring: 3000 | Damper: 100 | MaxForce: 6000

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

            // 16 body parts
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

            bodyController.bodyPartsList.ForEach(x =>
            {
                if (x.rb.transform != footR && x.rb.transform != footL &&
                    x.rb.transform != shinR && x.rb.transform != shinL)
                    x.GroundContact.endEpisodeOnContact = true;
            });


        }

        // 138 observations
        public override void CollectObservations(StateVector stateBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;
            BodyPart _faR = jdDict[forearmR];
            BodyPart _faL = jdDict[forearmL];
            BodyPart _shinR = jdDict[shinR];
            BodyPart _shinL = jdDict[shinL];

            // 4 x 8 = 32
            {
                stateBuffer.AddObservation(_faR.rb.velocity / 20f);
                stateBuffer.AddObservation(_faR.rb.angularVelocity / 20f);
                stateBuffer.AddObservation(_faR.CurrentNormalizedRotation.x);
                stateBuffer.AddObservation(_faR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_faL.rb.velocity / 20f);
                stateBuffer.AddObservation(_faL.rb.angularVelocity / 20f);
                stateBuffer.AddObservation(_faL.CurrentNormalizedRotation.x);
                stateBuffer.AddObservation(_faL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_shinR.rb.velocity / 20f);
                stateBuffer.AddObservation(_shinR.rb.angularVelocity / 20f);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedRotation.x);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_shinL.rb.velocity / 20f);
                stateBuffer.AddObservation(_shinL.rb.angularVelocity / 20f);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedRotation.x);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedStrength);
            }

            // + 6
            BodyPart _cst = jdDict[chest];
            {
                stateBuffer.AddObservation(_cst.rb.velocity / 20f);
                stateBuffer.AddObservation(_cst.rb.angularVelocity / 20f);
            }



            // 10 X 9 
            foreach (var bp in bodyController.bodyPartsList)
            {
                if (bp.rb.transform == handL || bp.rb.transform == handR)
                    continue;

                if (bp.rb.transform == chest)
                    continue;

                if (bp.rb.transform == forearmR || bp.rb.transform == forearmL)
                    continue;

                if (bp.rb.transform == shinR || bp.rb.transform == shinL)
                    continue;

                stateBuffer.AddObservation(bp.rb.velocity / 20f);
                stateBuffer.AddObservation(bp.rb.angularVelocity / 20f);
                stateBuffer.AddObservation(bp.CurrentNormalizedRotation);
                stateBuffer.AddObservation(bp.CurrentNormalizedStrength);
            }

            

            // + 2           
            stateBuffer.AddObservation(jdDict[footL].GroundContact.IsGrounded);
            stateBuffer.AddObservation(jdDict[footR].GroundContact.IsGrounded);

            // 130 inputs
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

            const float disc = 0.001f;
            AddReward(head.position.y * disc);

            AddReward(Mathf.Clamp(head.position.z * disc, -0.01f, 0.01f)); // add reward to move forward.
        }
    }

}


