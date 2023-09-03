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


        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            // 15 info from raysensor
            // 14 info from joints
            foreach (var item in joints)
            {
                sensorBuffer.AddObservation(item.jointAngle / item.limits.max);
                sensorBuffer.AddObservation(item.jointSpeed / jointsSpeed);
            }


        }
        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            for (int i = 0; i < joints.Count; i++)
            {
                var motor = joints[i].motor;
                motor.motorSpeed = actionBuffer.ContinuousActions[i] * jointsSpeed;
                joints[i].motor = motor;
            }
            AddReward(joints[0].transform.position.x / 100);
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
