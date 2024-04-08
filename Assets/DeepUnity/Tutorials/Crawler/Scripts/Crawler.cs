using UnityEngine;
using DeepUnity;
using System;
using DeepUnity.ReinforcementLearning;

namespace DeepUnity.Tutorials
{
    public class Crawler : Agent
    {
        [Header("Requires Normalization")]
        [SerializeField] GameObject bonus;

        [SerializeField] GameObject thigh1;
        [SerializeField] GameObject thigh2;
        [SerializeField] GameObject thigh3;
        [SerializeField] GameObject thigh4;
                     
        [SerializeField] GameObject shin1;
        [SerializeField] GameObject shin2;
        [SerializeField] GameObject shin3;
        [SerializeField] GameObject shin4;
                       
        [SerializeField] GameObject foot1;
        [SerializeField] GameObject foot2;
        [SerializeField] GameObject foot3;
        [SerializeField] GameObject foot4;

        [ViewOnly] public bool foot1_grounded = false;
        [ViewOnly] public bool foot2_grounded = false;
        [ViewOnly] public bool foot3_grounded = false;
        [ViewOnly] public bool foot4_grounded = false;

        BodyController controller;


        public override void OnEpisodeBegin()
        {
            if (bonus == null)
                return;

            for (int i = 0; i < bonus.transform.childCount; i++)
            {
                Transform x = bonus.transform.GetChild(i);
                x.gameObject.SetActive(true);
            }
        }

        public override void Awake()
        {
            base.Awake();

            controller = GetComponent<BodyController>();

            controller.AddBodyPart(this.gameObject);
            controller.AddBodyPart(thigh1);
            controller.AddBodyPart(thigh2);
            controller.AddBodyPart(thigh3);
            controller.AddBodyPart(thigh4);
            controller.AddBodyPart(shin1);
            controller.AddBodyPart(shin2);
            controller.AddBodyPart(shin3);
            controller.AddBodyPart(shin4);
            controller.AddBodyPart(foot1);
            controller.AddBodyPart(foot2);
            controller.AddBodyPart(foot3);
            controller.AddBodyPart(foot4);


            Action<Collision> end = (col) =>
            {
                if (col.collider.CompareTag("Ground"))
                {
                    EndEpisode();
                }
            };

            Action<Collider> rewards = (col) =>
            {
                if (col.CompareTag("Target"))
                {
                    AddReward(+0.1f);
                    col.gameObject.SetActive(false);
                }
            };


            controller.bodyPartsDict[this.gameObject].ColliderContact.OnEnter = end;
            controller.bodyPartsDict[this.gameObject].TriggerContact.OnEnter = rewards;
            controller.bodyPartsDict[thigh1].ColliderContact.OnEnter = end;
            controller.bodyPartsDict[thigh2].ColliderContact.OnEnter = end;
            controller.bodyPartsDict[thigh3].ColliderContact.OnEnter = end;
            controller.bodyPartsDict[thigh4].ColliderContact.OnEnter = end;

            controller.bodyPartsDict[foot1].ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground")) foot1_grounded = true; };
            controller.bodyPartsDict[foot2].ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground")) foot2_grounded = true; };
            controller.bodyPartsDict[foot3].ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground")) foot3_grounded = true; };
            controller.bodyPartsDict[foot4].ColliderContact.OnEnter = (col) => { if (col.collider.CompareTag("Ground")) foot4_grounded = true; };
                                                                                                                                              
            controller.bodyPartsDict[foot1].ColliderContact.OnExit = (col) => { if (col.collider.CompareTag("Ground")) foot1_grounded = false; };
            controller.bodyPartsDict[foot2].ColliderContact.OnExit = (col) => { if (col.collider.CompareTag("Ground")) foot2_grounded = false; };
            controller.bodyPartsDict[foot3].ColliderContact.OnExit = (col) => { if (col.collider.CompareTag("Ground")) foot3_grounded = false; };
            controller.bodyPartsDict[foot4].ColliderContact.OnExit = (col) => { if (col.collider.CompareTag("Ground")) foot4_grounded = false; };
        }

        public override void CollectObservations(StateVector stateBuffer)
        {
            // Total 94

            // + 10
            stateBuffer.AddObservation(transform.rotation); 
            stateBuffer.AddObservation(controller.bodyPartsDict[this.gameObject].rigidbody.velocity);
            stateBuffer.AddObservation(controller.bodyPartsDict[this.gameObject].rigidbody.angularVelocity);

            // + 10 x 8
            foreach (var bp in controller.bodyPartsList)
            {
                if (bp.gameObject == foot1 || bp.gameObject == foot2 || bp.gameObject == foot3 || bp.gameObject == foot4)
                    continue;

                if (bp.rigidbody.transform == transform)
                    continue;
              

                // 10 info
                stateBuffer.AddObservation(bp.rigidbody.velocity);
                stateBuffer.AddObservation(bp.rigidbody.angularVelocity);
                stateBuffer.AddObservation(bp.CurrentNormalizedEulerRotation);
                stateBuffer.AddObservation(bp.CurrentNormalizedStrength);
            }

            // 4
            stateBuffer.AddObservation(foot1_grounded);
            stateBuffer.AddObservation(foot2_grounded);
            stateBuffer.AddObservation(foot3_grounded);
            stateBuffer.AddObservation(foot4_grounded);
        }

        public void OnDrawGizmos()
        {
            Gizmos.color = Color.red; // You can change the color as desired

            // Calculate the rotation (90 degrees around Y-axis)
            Quaternion rotation = Quaternion.Euler(0f, -90f, 0f);

            // Apply the rotation to the forward vector
            Vector3 rotatedForward = rotation * transform.forward;

            // Calculate the end point of the rotated line
            Vector3 endPoint = transform.position + rotatedForward * 5f;

            // Draw the line using Gizmos.DrawLine
            Gizmos.DrawLine(transform.position, endPoint);
        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            // Total 20


            float[] act_vec = actionBuffer.ContinuousActions;

            controller.bodyPartsDict[thigh1].SetJointTargetRotation(act_vec[0], act_vec[1], 0);
            controller.bodyPartsDict[thigh2].SetJointTargetRotation(act_vec[2], act_vec[3], 0);
            controller.bodyPartsDict[thigh3].SetJointTargetRotation(act_vec[4], act_vec[5], 0);
            controller.bodyPartsDict[thigh4].SetJointTargetRotation(act_vec[6], act_vec[7], 0);

            controller.bodyPartsDict[shin1].SetJointTargetRotation(act_vec[8], 0, 0);
            controller.bodyPartsDict[shin2].SetJointTargetRotation(act_vec[9], 0, 0);
            controller.bodyPartsDict[shin3].SetJointTargetRotation(act_vec[10], 0, 0);
            controller.bodyPartsDict[shin4].SetJointTargetRotation(act_vec[11], 0, 0);

            controller.bodyPartsDict[thigh1].SetJointStrength(act_vec[12]);
            controller.bodyPartsDict[thigh2].SetJointStrength(act_vec[13]);
            controller.bodyPartsDict[thigh3].SetJointStrength(act_vec[14]);
            controller.bodyPartsDict[thigh4].SetJointStrength(act_vec[15]);

            controller.bodyPartsDict[shin1].SetJointStrength(act_vec[16]);
            controller.bodyPartsDict[shin2].SetJointStrength(act_vec[17]);
            controller.bodyPartsDict[shin3].SetJointStrength(act_vec[18]);
            controller.bodyPartsDict[shin4].SetJointStrength(act_vec[19]);

            // Point the arrow towards the target
            // directionArrow.rotation = Quaternion.LookRotation(target.position - transform.position) * Quaternion.Euler(0, 90f, 0);

            // float orientation_reward = (1f - Vector3.Angle(-Vector3.forward, Quaternion.Euler(0, 90f, 0f) * -transform.forward) % 360f / 360f);
            // float position_reward = -transform.position.z;
            // float reward = 0.005f * position_reward + 0.001f * orientation_reward;
            // AddReward(Mathf.Clamp(reward, -0.05f, 0.05f));

            AddReward(+0.00001f);
        }
    }


}
