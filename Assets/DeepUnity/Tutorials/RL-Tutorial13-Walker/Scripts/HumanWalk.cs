using UnityEngine;

using DeepUnity;


namespace DeepUnityTutorials
{
    public class HumanWalk : Agent
    {
        public float highStrengthJointsForce = 1000;
        public float lowStrengthJointsForce = 600;


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
        public IsGroundedScript leftFootIsGrounded;

        [Header("Right leg")]
        public JointScript rightLegJoint;      
        public JointScript rightShinJoint;       
        public JointScript rightFootJoint;
        public IsGroundedScript rightFootIsGrounded;

        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            // 10 info per joint
   
            // Mid body - 40
            sensorBuffer.AddObservation(headJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(headJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(headJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(stomachJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(stomachJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(stomachJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(torsoJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(torsoJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(torsoJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(chest.transform.rotation.normalized);
            sensorBuffer.AddObservation(chestRigidbody.velocity.normalized);
            sensorBuffer.AddObservation(chestRigidbody.angularVelocity.normalized);

            // Left arm - 20

            sensorBuffer.AddObservation(leftArmJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(leftArmJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(leftArmJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(leftForearmJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(leftForearmJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(leftForearmJoint.rb.angularVelocity.normalized);


            // Right arm - 20

            sensorBuffer.AddObservation(rightArmJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(rightArmJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(rightArmJoint.rb.angularVelocity.normalized);
                                       
            sensorBuffer.AddObservation(rightForearmJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(rightForearmJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(rightForearmJoint.rb.angularVelocity.normalized);


            // Left leg - 31


            sensorBuffer.AddObservation(leftLegJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(leftLegJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(leftLegJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(leftShinJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(leftShinJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(leftShinJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(leftFootJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(leftFootJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(leftFootJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(leftFootIsGrounded.IsGrounded);



            // Right Leg - 31

            sensorBuffer.AddObservation(rightLegJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(rightLegJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(rightLegJoint.rb.angularVelocity.normalized);
                                       
            sensorBuffer.AddObservation(rightShinJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(rightShinJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(rightShinJoint.rb.angularVelocity.normalized);
                                      
            sensorBuffer.AddObservation(rightFootJoint.transform.rotation.normalized);
            sensorBuffer.AddObservation(rightFootJoint.rb.velocity.normalized);
            sensorBuffer.AddObservation(rightFootJoint.rb.angularVelocity.normalized);

            sensorBuffer.AddObservation(rightFootIsGrounded.IsGrounded);


            // 80 + 62 = 142
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {

            headJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[0] * lowStrengthJointsForce, actionBuffer.ContinuousActions[1] * lowStrengthJointsForce, actionBuffer.ContinuousActions[2] * lowStrengthJointsForce);
            stomachJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[3] * lowStrengthJointsForce, actionBuffer.ContinuousActions[4] * lowStrengthJointsForce, actionBuffer.ContinuousActions[5] * lowStrengthJointsForce);
            torsoJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[6] * lowStrengthJointsForce, actionBuffer.ContinuousActions[7] * lowStrengthJointsForce, actionBuffer.ContinuousActions[8] * lowStrengthJointsForce);
            
            leftArmJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[9] * lowStrengthJointsForce, actionBuffer.ContinuousActions[10] * lowStrengthJointsForce, actionBuffer.ContinuousActions[11] * lowStrengthJointsForce);
            leftForearmJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[12] * lowStrengthJointsForce, 0f, 0f);

            rightArmJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[13] * lowStrengthJointsForce, actionBuffer.ContinuousActions[14] * lowStrengthJointsForce, actionBuffer.ContinuousActions[15] * lowStrengthJointsForce);
            rightForearmJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[16] * lowStrengthJointsForce, 0f, 0f);

            leftLegJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[17] * highStrengthJointsForce, actionBuffer.ContinuousActions[18] * highStrengthJointsForce, actionBuffer.ContinuousActions[19] * highStrengthJointsForce);
            leftShinJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[20] * highStrengthJointsForce, 0f, 0f);
            leftFootJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[21] * lowStrengthJointsForce, actionBuffer.ContinuousActions[22] * lowStrengthJointsForce, 0f);
            
            rightLegJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[23] * highStrengthJointsForce, actionBuffer.ContinuousActions[24] * highStrengthJointsForce, actionBuffer.ContinuousActions[25] * highStrengthJointsForce);
            rightShinJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[26] * highStrengthJointsForce, 0f, 0f);
            rightFootJoint.SetTargetAngularVelocity(actionBuffer.ContinuousActions[27] * lowStrengthJointsForce, actionBuffer.ContinuousActions[28] * lowStrengthJointsForce, 0f);

            AddReward(+0.0025f); // Constant existential reward
            AddReward(headJoint.transform.position.y / 1000f); // reward for keeping the head up
        }
    }

}


