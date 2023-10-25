using UnityEngine;
using DeepUnity;


namespace DeepUnityTutorials
{
    public class HumanWalk : Agent
    {
        public float highStrengthJointsForce = 40;
        public float lowStrengthJointsForce = 20;


        [Header("Body")]
        public JointScript headJoint;
        public JointScript stomachJoint;
        public JointScript torsoJoint;
        public Transform chest;
        public Rigidbody chestRigidbody;
   
        [Header("Left arm")]
        public JointScript leftArmJoint;
        public JointScript leftForearmJoint;
       
        [Header("Right arm")] 
        public JointScript rightArmJoint;
        public JointScript rightForearmJoint;

        [Header("Left leg")]
        public JointScript leftLegJoint;
        public JointScript leftShinJoint;
        public JointScript leftFootJoint;
        public GroundContact leftFootIsGrounded;

        [Header("Right leg")]
        public JointScript rightLegJoint;      
        public JointScript rightShinJoint;       
        public JointScript rightFootJoint;
        public GroundContact rightFootIsGrounded;

        public override void CollectObservations(StateBuffer stateBuffer)
        {
            // 10 info per joint
   
            // Mid body - 40
            stateBuffer.AddObservation(headJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(headJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(headJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(stomachJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(stomachJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(stomachJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(torsoJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(torsoJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(torsoJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(chest.transform.rotation.normalized);
            stateBuffer.AddObservation(chestRigidbody.velocity.normalized);
            stateBuffer.AddObservation(chestRigidbody.angularVelocity.normalized);

            // Left arm - 20

            stateBuffer.AddObservation(leftArmJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(leftArmJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(leftArmJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(leftForearmJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(leftForearmJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(leftForearmJoint.rb.angularVelocity.normalized);


            // Right arm - 20

            stateBuffer.AddObservation(rightArmJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(rightArmJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(rightArmJoint.rb.angularVelocity.normalized);
                                       
            stateBuffer.AddObservation(rightForearmJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(rightForearmJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(rightForearmJoint.rb.angularVelocity.normalized);


            // Left leg - 31


            stateBuffer.AddObservation(leftLegJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(leftLegJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(leftLegJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(leftShinJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(leftShinJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(leftShinJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(leftFootJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(leftFootJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(leftFootJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(leftFootIsGrounded.IsGrounded);



            // Right Leg - 31

            stateBuffer.AddObservation(rightLegJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(rightLegJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(rightLegJoint.rb.angularVelocity.normalized);
                                       
            stateBuffer.AddObservation(rightShinJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(rightShinJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(rightShinJoint.rb.angularVelocity.normalized);
                                      
            stateBuffer.AddObservation(rightFootJoint.transform.rotation.normalized);
            stateBuffer.AddObservation(rightFootJoint.rb.velocity.normalized);
            stateBuffer.AddObservation(rightFootJoint.rb.angularVelocity.normalized);

            stateBuffer.AddObservation(rightFootIsGrounded.IsGrounded);


            // 80 + 62 = 142
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {

            headJoint.SetAngularVelocity(actionBuffer.ContinuousActions[0] * lowStrengthJointsForce, actionBuffer.ContinuousActions[1] * lowStrengthJointsForce, actionBuffer.ContinuousActions[2] * lowStrengthJointsForce);
            stomachJoint.SetAngularVelocity(actionBuffer.ContinuousActions[3] * lowStrengthJointsForce, actionBuffer.ContinuousActions[4] * lowStrengthJointsForce, actionBuffer.ContinuousActions[5] * lowStrengthJointsForce);
            torsoJoint.SetAngularVelocity(actionBuffer.ContinuousActions[6] * lowStrengthJointsForce, actionBuffer.ContinuousActions[7] * lowStrengthJointsForce, actionBuffer.ContinuousActions[8] * lowStrengthJointsForce);
            
            leftArmJoint.SetAngularVelocity(actionBuffer.ContinuousActions[9] * lowStrengthJointsForce, actionBuffer.ContinuousActions[10] * lowStrengthJointsForce, actionBuffer.ContinuousActions[11] * lowStrengthJointsForce);
            leftForearmJoint.SetAngularVelocity(actionBuffer.ContinuousActions[12] * lowStrengthJointsForce, 0f, 0f);

            rightArmJoint.SetAngularVelocity(actionBuffer.ContinuousActions[13] * lowStrengthJointsForce, actionBuffer.ContinuousActions[14] * lowStrengthJointsForce, actionBuffer.ContinuousActions[15] * lowStrengthJointsForce);
            rightForearmJoint.SetAngularVelocity(actionBuffer.ContinuousActions[16] * lowStrengthJointsForce, 0f, 0f);

            leftLegJoint.SetAngularVelocity(actionBuffer.ContinuousActions[17] * highStrengthJointsForce, actionBuffer.ContinuousActions[18] * highStrengthJointsForce, actionBuffer.ContinuousActions[19] * highStrengthJointsForce);
            leftShinJoint.SetAngularVelocity(actionBuffer.ContinuousActions[20] * highStrengthJointsForce, 0f, 0f);
            leftFootJoint.SetAngularVelocity(actionBuffer.ContinuousActions[21] * lowStrengthJointsForce, actionBuffer.ContinuousActions[22] * lowStrengthJointsForce, 0f);
            
            rightLegJoint.SetAngularVelocity(actionBuffer.ContinuousActions[23] * highStrengthJointsForce, actionBuffer.ContinuousActions[24] * highStrengthJointsForce, actionBuffer.ContinuousActions[25] * highStrengthJointsForce);
            rightShinJoint.SetAngularVelocity(actionBuffer.ContinuousActions[26] * highStrengthJointsForce, 0f, 0f);
            rightFootJoint.SetAngularVelocity(actionBuffer.ContinuousActions[27] * lowStrengthJointsForce, actionBuffer.ContinuousActions[28] * lowStrengthJointsForce, 0f);

            AddReward(+0.025f); // Constant existential reward
            AddReward(headJoint.transform.position.y / 100f); // reward for keeping the head up
        }
    }

}


