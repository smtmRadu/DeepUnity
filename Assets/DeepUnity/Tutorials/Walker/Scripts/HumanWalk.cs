using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class HumanWalk : Agent
    {
        // Spring: 3000 | Damper: 100 | MaxForce: 6000
        public Transform target;

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

            bodyController.bodyPartsList.ForEach(x =>
            {
                if (x.rb.transform != footR && x.rb.transform != footL &&
                    x.rb.transform != shinR && x.rb.transform != shinL)
                    x.GroundContact.endEpisodeOnContact = true;
            });

            bodyController.bodyPartsList.ForEach(x => x.TargetContact.rewardOnContact = 10f);
        }
        public override void OnEpisodeBegin()
        {
            float random_angle = Utils.Random.Range(0f, 360f);
            const float distance = 5f;

            float random_rad = Mathf.Rad2Deg * random_angle;
            float x = distance * Mathf.Cos(random_rad);
            float z = distance * Mathf.Sin(random_rad);

            target.position = new Vector3(x, target.position.y, z);
        }
        public override void CollectObservations(StateBuffer stateBuffer)
        {
            // 12 Set direction to Target....
            Vector3 dirToTarget = target.position - bodyController.bodyPartsDict[head].rb.position;
            stateBuffer.AddObservation(dirToTarget.normalized); //3
            stateBuffer.AddObservation(bodyController.bodyPartsDict[head].rb.position); //3
            stateBuffer.AddObservation(head.forward); //3
            stateBuffer.AddObservation(head.up); //3
      
            // 13 X 13
            foreach (var bp in bodyController.bodyPartsList)
            {
                if (bp.rb.transform == chest)
                    continue;

                if (bp.rb.transform == handL)
                    continue;

                if (bp.rb.transform == handR)
                    continue;

                // 9 info
                stateBuffer.AddObservation(bp.rb.velocity);
                stateBuffer.AddObservation(bp.rb.angularVelocity);
                Vector3 localPosRelToHips = hips.InverseTransformPoint(bp.rb.position);
                stateBuffer.AddObservation(localPosRelToHips);

                // 4 info
                stateBuffer.AddObservation(bp.CurrentNormalizedRotation);
                stateBuffer.AddObservation(bp.CurrentNormalizedStrength);                
            }

            // 8
            var jdDict = bodyController.bodyPartsDict;

            stateBuffer.AddObservation(jdDict[footL].GroundContact.IsGrounded);
            stateBuffer.AddObservation(jdDict[footR].GroundContact.IsGrounded);
            stateBuffer.AddObservation(jdDict[chest].rb.velocity);
            stateBuffer.AddObservation(jdDict[chest].rb.angularVelocity);

            // 189 total
        }
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
            jdDict[footL].SetJointTargetRotation(actions_vector[21], actions_vector[22], 0f);

            jdDict[legR].SetJointTargetRotation(actions_vector[23], actions_vector[24], actions_vector[25]);
            jdDict[shinR].SetJointTargetRotation(actions_vector[26], 0f, 0f);
            jdDict[footR].SetJointTargetRotation(actions_vector[27], actions_vector[28], 0f);



            jdDict[head].SetJointStrength(actions_vector[27]);
            jdDict[stomach].SetJointStrength(actions_vector[28]);
            jdDict[hips].SetJointStrength(actions_vector[29]);

            jdDict[armL].SetJointStrength(actions_vector[30]);
            jdDict[forearmL].SetJointStrength(actions_vector[31]);

            jdDict[armR].SetJointStrength(actions_vector[32]);
            jdDict[forearmR].SetJointStrength(actions_vector[33]);

            jdDict[legL].SetJointStrength(actions_vector[34]);
            jdDict[shinL].SetJointStrength(actions_vector[35]);
            jdDict[footL].SetJointStrength(actions_vector[36]);

            jdDict[legR].SetJointStrength(actions_vector[37]);
            jdDict[shinR].SetJointStrength(actions_vector[38]);
            jdDict[footR].SetJointStrength(actions_vector[39]);


            AddReward(head.position.y / 100f); // Constant reward for keeping head up

            const float range = 50f;
            if (transform.position.x < -range || transform.position.x > range ||
                transform.position.z < -range || transform.position.z > range ||
                transform.position.y < -range)
                EndEpisode();
        }
    }

}


