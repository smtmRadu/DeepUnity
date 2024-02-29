using DeepUnity;
using DeepUnity.ReinforcementLearning;
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
        public Transform target;
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
            Transform randInit = Utils.Random.Sample(initialStates);
            transform.position = randInit.position;
            transform.rotation = randInit.rotation;
            this.rb.velocity = Vector3.zero;
            this.rb.angularVelocity = Vector3.zero;
        }

        public override void CollectObservations(StateVector sensorBuffer)
        {
            // + 10
            sensorBuffer.AddObservation(rb.velocity / 30f);
            sensorBuffer.AddObservation(transform.rotation.x % 360 / 360f);
            sensorBuffer.AddObservation(transform.rotation.y % 360 / 360f);
            sensorBuffer.AddObservation(transform.rotation.z % 360 / 360f);
            sensorBuffer.AddObservation(transform.rotation.w % 360 / 360f);
            sensorBuffer.AddObservation((transform.position - target.position).normalized);
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
            actionOut.ContinuousActions[0] = Input.GetAxis("Vertical");
            actionOut.ContinuousActions[1] = Input.GetAxis("Horizontal");
            actionOut.ContinuousActions[2] = Input.GetKey(KeyCode.Space) ? 1 : 0;
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


        private void OnCollisionEnter(Collision collision)
        {
            if(collision.collider.CompareTag("Wall"))
            {
                AddReward(-0.1f * (transform.position - target.position).sqrMagnitude);
                EndEpisode();
            }
            else if(collision.collider.CompareTag("Target"))
            {
                AddReward(1f);
                EndEpisode();
            }
        }

    }


}

