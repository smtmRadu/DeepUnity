using DeepUnity.ReinforcementLearning;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.Tutorials
{

    public class Driver : Agent
    {

      
        private CarController carController;

        List<GameObject> checkPoints = new List<GameObject>();

        [Header("Attributes - No additional observations, no normalization.")]
        public float maxMotorTorque = 1000f;
        public float maxSteerAngle = 35f;
        public float breakPower = 1000000f;

        private Rigidbody rb;
        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
            carController = GetComponent<CarController>();
        }
        public override void OnEpisodeBegin()
        {
            foreach (var item in checkPoints)
            {
                item.SetActive(true);
            }
            checkPoints.Clear();
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            carController.Accelerate(actionBuffer.ContinuousActions[0]);
            carController.Steer(actionBuffer.ContinuousActions[1]);
            carController.Break(actionBuffer.ContinuousActions[2] == 1f);
        }
        public override void Heuristic(ActionBuffer actionOut)
        {
            actionOut.ContinuousActions[0] = Input.GetAxis("Vertical");
            actionOut.ContinuousActions[1] = Input.GetAxis("Horizontal");
            actionOut.ContinuousActions[2] = Input.GetKey(KeyCode.Space) ? 1f : 0f;
        }


        private static void CorrelateColliderAndTransform(WheelCollider wheel_collider, Transform wheel_transform, bool yaxis180rot = false)
        {
            var pos = Vector3.zero;
            var rot = Quaternion.identity;

            wheel_collider.GetWorldPose(out pos, out rot);
            wheel_transform.position = pos;
            wheel_transform.rotation = rot * Quaternion.Euler(0, yaxis180rot ? 180f : 0f, 0f);
        }
        private void OnCollisionEnter(Collision collision)
        {
            if (collision.collider.CompareTag("Wall"))
            {
                AddReward(-1f);
                EndEpisode();
            }

        }

        private void OnTriggerEnter(Collider other)
        {
            if(other.CompareTag("Goal"))
            {
                AddReward(+0.1f);

                checkPoints.Add(other.gameObject);
                other.gameObject.SetActive(false);
            }
            else if(other.CompareTag("Target"))
            {
                AddReward(+0.15f);
                EndEpisode();
            }
        }
    }


}

