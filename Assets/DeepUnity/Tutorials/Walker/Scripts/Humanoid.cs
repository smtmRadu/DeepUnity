using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    /// <summary>
    /// This script was made for the tall mode. This is very hard to train due to it's nature cause balancing is the first problem.
    /// </summary>
    public class Humanoid : Agent
    {
        // Spring: 3000 | Damper: 100 | MaxForce: 6000

        [Header("Body Parts 16")]
        public GameObject head;
        public GameObject stomach;
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

        [ViewOnly] public float stepReward;


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

            bodyController.bodyPartsList.ForEach(bodypart =>
            {
                if (bodypart.gameObject != footL.gameObject && bodypart.gameObject != footR.gameObject && bodypart.gameObject != shinL.gameObject && bodypart.gameObject != shinR.gameObject)
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
        public override void CollectObservations(StateVector stateBuffer)
        {
            var jdDict = bodyController.bodyPartsDict;
            BodyPart _faR = jdDict[forearmR];
            BodyPart _faL = jdDict[forearmL];
            BodyPart _shinR = jdDict[shinR];
            BodyPart _shinL = jdDict[shinL];

            const float velocity_norm = 30f;
            // 4 x 8 = 32
            {
                stateBuffer.AddObservation(_faR.rigidbody.velocity / velocity_norm);
                stateBuffer.AddObservation(_faR.rigidbody.angularVelocity / velocity_norm);
                stateBuffer.AddObservation(_faR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_faL.rigidbody.velocity / velocity_norm);
                stateBuffer.AddObservation(_faL.rigidbody.angularVelocity / velocity_norm);
                stateBuffer.AddObservation(_faL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_faL.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_shinR.rigidbody.velocity / velocity_norm);
                stateBuffer.AddObservation(_shinR.rigidbody.angularVelocity / velocity_norm);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinR.CurrentNormalizedStrength);

                stateBuffer.AddObservation(_shinL.rigidbody.velocity / velocity_norm);
                stateBuffer.AddObservation(_shinL.rigidbody.angularVelocity / velocity_norm);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedEulerRotation.x);
                stateBuffer.AddObservation(_shinL.CurrentNormalizedStrength);
            }

            // + 6
            BodyPart _cst = jdDict[chest];
            {
                stateBuffer.AddObservation(_cst.rigidbody.velocity / velocity_norm);
                stateBuffer.AddObservation(_cst.rigidbody.angularVelocity / velocity_norm);
            }

            // 10 X 9 
            foreach (var bp in bodyController.bodyPartsList)
            {
                if (bp.rigidbody.transform == handL || bp.rigidbody.transform == handR)
                    continue;

                if (bp.rigidbody.transform == chest)
                    continue;

                if (bp.rigidbody.transform == forearmR || bp.rigidbody.transform == forearmL)
                    continue;

                if (bp.rigidbody.transform == shinR || bp.rigidbody.transform == shinL)
                    continue;

                stateBuffer.AddObservation(bp.rigidbody.velocity / velocity_norm);
                stateBuffer.AddObservation(bp.rigidbody.angularVelocity / velocity_norm);
                stateBuffer.AddObservation(bp.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(bp.CurrentNormalizedStrength);
            }

            // + 2           
            stateBuffer.AddObservation(bodyController.bodyPartsDict[footL].ColliderContact.IsGrounded);
            stateBuffer.AddObservation(bodyController.bodyPartsDict[footR].ColliderContact.IsGrounded);

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

            // /// Normal reward
            Vector3 stom_head_dif = stomach.transform.position - head.transform.position;
            float head_stomach_alignment = 0.002f * (1 - new Vector2(stom_head_dif.x, stom_head_dif.z).magnitude);   
            float orientation_reward = 0.003f * (1f - Vector3.Angle(-Vector3.forward, -head.transform.forward) % 360f / 360f);
            float position_reward = 0.01f * (-stomach.transform.position.z);
            float alive_reward = 0.005f;
            stepReward = head_stomach_alignment + orientation_reward + alive_reward + position_reward;
            stepReward /= 10f;
            AddReward(stepReward);
            // /// Geometric reward
            // float head_stomach_alignment = 1f - Mathf.Abs(stomach.transform.position.z - head.transform.position.z);
            // float orientation_reward = 1f - Vector3.Angle(-Vector3.forward, -head.transform.forward) % 360f / 360f;
            // float position_reward = -stomach.transform.position.z;
            // float head_reward = 1;
            // stepReward = head_stomach_alignment * orientation_reward * position_reward * head_reward;
            // stepReward /= 75f;
            // stepReward = Mathf.Clamp(stepReward, -0.1f, 0.1f);
            // AddReward(stepReward);
        }
    }

}


