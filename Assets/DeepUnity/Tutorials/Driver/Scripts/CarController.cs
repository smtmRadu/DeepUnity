using UnityEngine;

namespace DeepUnityTutorials
{
 
    public class CarController : MonoBehaviour
    {
        public float maxMotorTorque = 1000f;
        public float maxSteerAngle = 35f;
        public float breakPower = 100_000f;

        [Header("Wheels")]
        public WheelCollider rf_collider;
        public WheelCollider rb_collider;
        public WheelCollider lf_collider;
        public WheelCollider lb_collider;

        public Transform rf_transform;
        public Transform rb_transform;
        public Transform lf_transform;
        public Transform lb_transform;

        public Transform center_of_mass;

        private Rigidbody rb;


        private void Awake()
        {
            rb = GetComponent<Rigidbody>();
            rb.centerOfMass = center_of_mass.transform.localPosition;
        }
        private void Start()
        {
            rf_collider.motorTorque = 0f;
            rb_collider.motorTorque = 0f;
            lf_collider.motorTorque = 0f;
            lb_collider.motorTorque = 0f;
        }

        private void FixedUpdate()
        {
            CorrelateColliderAndTransform(lb_collider, lb_transform);
            CorrelateColliderAndTransform(lf_collider, lf_transform);
            CorrelateColliderAndTransform(rb_collider, rb_transform, true);
            CorrelateColliderAndTransform(rf_collider, rf_transform, true);
        }

        public void Accelerate(float strength)
        {
            rb_collider.motorTorque = maxMotorTorque * strength;
            lb_collider.motorTorque = maxMotorTorque * strength;
        }
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
            }
            else
            {
                rf_collider.brakeTorque = 0f;
                lf_collider.brakeTorque = 0f;
                rb_collider.brakeTorque = 0f;
                lb_collider.brakeTorque = 0f;
            }
        }
        private static void CorrelateColliderAndTransform(WheelCollider wheel_collider, Transform wheel_transform, bool yaxis180rot = false)
        {
            var pos = Vector3.zero;
            var rot = Quaternion.identity;

            wheel_collider.GetWorldPose(out pos, out rot);
            wheel_transform.position = pos;
            wheel_transform.rotation = rot * Quaternion.Euler(0, yaxis180rot ? 180f : 0f, 0f);
        }
    }
}



