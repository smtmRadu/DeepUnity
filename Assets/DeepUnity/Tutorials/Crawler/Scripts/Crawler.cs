using UnityEngine;
using DeepUnity;

namespace DeepUnityTutorials
{
    public class Crawler : Agent
    {
        [SerializeField] Transform target;
        [SerializeField] Transform directionArrow;

        [Header("Body Parts")]
        [SerializeField] Transform thigh1;
        [SerializeField] Transform thigh2;
        [SerializeField] Transform thigh3;
        [SerializeField] Transform thigh4;

        [SerializeField] Transform shin1;
        [SerializeField] Transform shin2;
        [SerializeField] Transform shin3;
        [SerializeField] Transform shin4;

        [SerializeField] Transform foot1;
        [SerializeField] Transform foot2;
        [SerializeField] Transform foot3;
        [SerializeField] Transform foot4;

        BodyController controller;
        Vector3 dirToTarget;
        public override void Awake()
        {
            base.Awake();

            controller = GetComponent<BodyController>();

            controller.AddBodyPart(this.transform);
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

            controller.bodyPartsDict[this.transform].GroundContact.endEpisodeOnContact = true;
            controller.bodyPartsDict[this.transform].GroundContact.rewardOnContact = -1f;
            controller.bodyPartsDict[thigh1].GroundContact.endEpisodeOnContact = true;
            controller.bodyPartsDict[thigh2].GroundContact.endEpisodeOnContact = true;
            controller.bodyPartsDict[thigh3].GroundContact.endEpisodeOnContact = true;
            controller.bodyPartsDict[thigh4].GroundContact.endEpisodeOnContact = true;
        }
        public override void OnEpisodeBegin()
        {
            float random_angle = Utils.Random.Range(0f, 360f);
            const float distance = 3f;

            float random_rad = Mathf.Rad2Deg * random_angle;
            float x = distance * Mathf.Cos(random_rad);
            float z = distance * Mathf.Sin(random_rad);

            target.localPosition = new Vector3(x, target.localPosition.y, z);
        }


        public override void CollectObservations(StateVector stateBuffer)
        {

            // Total 126

            // + 12
            dirToTarget = target.position - controller.bodyPartsDict[transform].rb.position;
            stateBuffer.AddObservation(dirToTarget.normalized); //3
            stateBuffer.AddObservation(controller.bodyPartsDict[transform].rb.position); //3
            stateBuffer.AddObservation(transform.forward); //3
            stateBuffer.AddObservation(transform.up); //3


            // + 13 x 8 + 4 + 6
            foreach (var bp in controller.bodyPartsList)
            {
                if (bp.rb.transform == transform)
                {
                    stateBuffer.AddObservation(bp.rb.velocity / 30f);
                    stateBuffer.AddObservation(bp.rb.angularVelocity / 30f);
                    continue;
                }
                if (bp.rb.transform == foot1 || bp.rb.transform == foot2 || bp.rb.transform == foot3 || bp.rb.transform == foot4)
                {
                    stateBuffer.AddObservation(bp.GroundContact.IsGrounded);
                    continue;
                }

                // 9 info
                stateBuffer.AddObservation(bp.rb.velocity / 30f);
                stateBuffer.AddObservation(bp.rb.angularVelocity / 30f);
                Vector3 localPosRelToHead = transform.InverseTransformPoint(bp.rb.position);
                stateBuffer.AddObservation(localPosRelToHead);

                // 4 info
                stateBuffer.AddObservation(bp.CurrentNormalizedRotation);
                stateBuffer.AddObservation(bp.CurrentNormalizedStrength);
            }

          
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

            AddReward(0.005f);
            AddReward(0.001f * Vector3.Dot(dirToTarget.normalized,  Quaternion.Euler(0f, -90f, 0f) * transform.forward)); // Reward for looking at the target
            AddReward(Mathf.Clamp(0.001f / Vector3.Distance(transform.position, target.position), 0, 0.001f)); // Reward for getting close to the target

            // Point the arrow towards the target
            directionArrow.rotation = Quaternion.LookRotation(target.position - transform.position) * Quaternion.Euler(0, 90f, 0);

            if (transform.position.y < -1)
                EndEpisode();
        }

        private void OnTriggerEnter(Collider other)
        {
            if (other.CompareTag("Target"))
            {
                AddReward(+1);
                OnEpisodeBegin();
            }
        }
    }


}
