using DeepUnity;
using System.Collections.Generic;
using UnityEngine;


namespace DeepUnityTutorials
{
    public class ParkingCar : Agent
    {
        public List<Transform> initialStates = new List<Transform>();
        public float maxMotorTorque = 1000f;
        public float maxSteerAngle = 35f;
        public float breakPower = 1000f;

        public Transform centerOfMass;
        [Header("Wheels")]
        public Transform lf_transform;
        public Transform rf_transform;
        public Transform lb_transform;
        public Transform rb_transform;
        public WheelCollider lf_collider;
        public WheelCollider rf_collider;
        public WheelCollider lb_collider;
        public WheelCollider rb_collider;

        public MeshRenderer brakelightleft;
        public MeshRenderer brakelightright;

        private Rigidbody rb;
        public override void Awake()
        {
            base.Awake();
            Physics.gravity = new Vector3(0f, -40f, 0f);
            rb = GetComponent<Rigidbody>();
            rb.centerOfMass = centerOfMass.transform.localPosition;
        }

        public override void OnEpisodeBegin()
        {
            // Transform randInit = Utils.Random.Sample(initialStates);
            // transform.position = randInit.position;
            // transform.rotation = randInit.rotation;
            // this.rb.velocity = Vector3.zero;
            // this.rb.angularVelocity = Vector3.zero;
        }

        public override void CollectObservations(SensorBuffer sensorBuffer)
        {
            sensorBuffer.AddObservation(rb.velocity);
        }

        public override void OnActionReceived(ActionBuffer actionBuffer)
        {
            CorrelateColliderAndTransform(lb_collider, lb_transform);
            CorrelateColliderAndTransform(lf_collider, lf_transform);
            CorrelateColliderAndTransform(rb_collider, rb_transform, true);
            CorrelateColliderAndTransform(rf_collider, rf_transform, true);

            Accelerate(actionBuffer.ContinuousActions[0]);
            Steer(actionBuffer.ContinuousActions[1]);
            Break(actionBuffer.ContinuousActions[2] > 0f);

        }

        public override void Heuristic(ActionBuffer actionOut)
        {
            Accelerate(Input.GetAxis("Vertical"));
            Steer(Input.GetAxis("Horizontal"));
            Break(Input.GetKey(KeyCode.Space));
        }

        /// <summary>
        /// Strength is a value in range [-1, 1]
        /// </summary>
        /// <param name="strength"></param>
        public void Accelerate(float strength)
        {
            rb_collider.motorTorque = maxMotorTorque * strength;
            lb_collider.motorTorque = maxMotorTorque * strength;
        }
        /// <summary>
        /// Strength is a value in range [-1, 1]
        /// </summary>
        /// <param name="strength"></param>
        public void Steer(float strength)
        {
            rf_collider.steerAngle = maxSteerAngle * strength;
            lf_collider.steerAngle = maxSteerAngle * strength;
        }
        public void Break(bool doBreak)
        {
            if (doBreak)
            {
                // rf_collider.brakeTorque = breakPower;
                // lf_collider.brakeTorque = breakPower;
                rb_collider.brakeTorque = breakPower;
                lb_collider.brakeTorque = breakPower;

                brakelightleft.enabled = true;
                brakelightright.enabled = true;
            }
            else
            {
                rf_collider.brakeTorque = 0f;
                lf_collider.brakeTorque = 0f;
                rb_collider.brakeTorque = 0f;
                lb_collider.brakeTorque = 0f;

                brakelightleft.enabled = false;
                brakelightright.enabled = false;
            }
        }

        private static void CorrelateColliderAndTransform(WheelCollider wheel_collider, Transform wheel_transform, bool yaxis180rot = false)
        {
            var pos = Vector3.zero;
            var rot = Quaternion.identity;

            wheel_collider.GetWorldPose(out pos, out rot);
            wheel_transform.position = pos;
            wheel_transform.rotation = rot * Quaternion.Euler(0f, yaxis180rot ? 180f : 0f, 0f);
        }

    }


}

