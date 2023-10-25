using UnityEngine;
using DeepUnity;
using System.Collections.Generic;

namespace DeepUnityTutorials
{
    public class CheetahRun : Agent
    {
        [Header("Properties")]
        public float jointsSpeed = 2000f;
        [SerializeField] List<HingeJoint2D> joints;
        public GroundContact2D contact1;
        public GroundContact2D contact2;

        public override void CollectObservations(StateBuffer sensorBuffer)
        {
            // 15 info from raysensor
            // 14 info from joints
            // 2 bool values
            foreach (var item in joints)
            {
                sensorBuffer.AddObservation(item.jointAngle / item.limits.max);
                sensorBuffer.AddObservation(item.jointSpeed / jointsSpeed);
            }

            sensorBuffer.AddObservation(contact1.IsGrounded);
            sensorBuffer.AddObservation(contact2.IsGrounded);

        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            for (int i = 0; i < joints.Count; i++)
            {
                var motor = joints[i].motor;
                motor.motorSpeed = actionBuffer.ContinuousActions[i] * jointsSpeed;
                joints[i].motor = motor;
            }

            float reward = joints[0].transform.position.x;
            AddReward(reward / 100f);
        }



        public override void Heuristic(ActionBuffer actionBuffer)
        {
            float hor = Input.GetAxis("Horizontal");
            float vert = Input.GetAxis("Vertical");

            actionBuffer.ContinuousActions[0] = hor;
            actionBuffer.ContinuousActions[1] = vert;
        }
    }



}
