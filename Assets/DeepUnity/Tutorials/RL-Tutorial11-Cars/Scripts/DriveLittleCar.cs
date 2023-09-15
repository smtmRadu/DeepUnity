using DeepUnity;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnityTutorials
{

    public class DriveLittleCar : Agent
    {

        [Header("Attributes")]
        public WheelCollider rf_collider;
        public WheelCollider rb_collider;
        public WheelCollider lf_collider;
        public WheelCollider lb_collider;

        public Transform rf_transform;
        public Transform rb_transform;
        public Transform lf_transform;
        public Transform lb_transform;

        public Transform center_of_mass;

        List<GameObject> checkPoints = new List<GameObject>();


        public float maxMotorTorque = 1000f;
        public float maxSteerAngle = 35f;
        public float breakPower = 1000f;

        private Rigidbody rb;
        public override void Awake()
        {
            base.Awake();
            rb = GetComponent<Rigidbody>();
            rb.centerOfMass = center_of_mass.transform.localPosition;
        }

        public override void OnEpisodeBegin()
        {
            foreach (var item in checkPoints)
            {
                item.SetActive(true);
            }
            checkPoints.Clear();

            rf_collider.motorTorque = 0f;
            rb_collider.motorTorque = 0f;
            lf_collider.motorTorque = 0f;
            lb_collider.motorTorque = 0f;
        }
        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            // +9 RaySensor
            // +4
            sensorBuffer.AddObservation((rf_collider.steerAngle + lf_collider.steerAngle) / (2 * maxSteerAngle));
            sensorBuffer.AddObservation(rb.velocity);
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            Accelerate(actionBuffer.ContinuousActions[0]);
            Steer(actionBuffer.ContinuousActions[1]);
            Break(actionBuffer.ContinuousActions[2] == 1f);

            CorrelateColliderAndTransform(lb_collider, lb_transform);
            CorrelateColliderAndTransform(lf_collider, lf_transform);
            CorrelateColliderAndTransform(rb_collider, rb_transform, true);
            CorrelateColliderAndTransform(rf_collider, rf_transform, true);

            AddReward(-0.0025f);
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
        private void Accelerate(float strength)
        {
            rb_collider.motorTorque = maxMotorTorque * strength;
            lb_collider.motorTorque = maxMotorTorque * strength;
        }
        private void Steer(float strength)
        {
            rf_collider.steerAngle = maxSteerAngle * strength;
            lf_collider.steerAngle = maxSteerAngle * strength;
        }
        private void Break(bool doBreak)
        {
            if (doBreak)
            {
                rb_collider.brakeTorque = breakPower;
                lb_collider.brakeTorque = breakPower;
            }
            else
            {
                rf_collider.brakeTorque = 0f;
                lf_collider.brakeTorque = 0f;
                rb_collider.brakeTorque = 0f;
                lb_collider.brakeTorque = 0f;
            }
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
            AddReward(+1f);

            checkPoints.Add(other.gameObject);
            other.gameObject.SetActive(false);
        }
    }


}

