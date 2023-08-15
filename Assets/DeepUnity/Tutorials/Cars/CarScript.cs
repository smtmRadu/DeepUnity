using Unity.VisualScripting;
using UnityEngine;
using UnityEngine.Video;
using DeepUnity;
using UnityEngine.UI;

public class CarScript : MonoBehaviour
{
	public WheelCollider rf_collider;
	public WheelCollider rb_collider;
	public WheelCollider lf_collider;
	public WheelCollider lb_collider;

	public Transform rf_transform;
	public Transform rb_transform;
	public Transform lf_transform;
	public Transform lb_transform;


    public Transform centerOfMass;
    private Rigidbody rb;
    private AudioSource audioSource;
    public CamSensor camSensor;
    public TMPro.TMP_Text speedLabel;
    public RawImage cameraView;

    public float maxSpeed = 100f;
    public float maxSteerAngle = 35f;

    [SerializeField, ReadOnly] private float motorTorque = 0f;
    [SerializeField, ReadOnly] private float steerAngle = 0f;
    

    private void Start()
    {
        rb = GetComponent<Rigidbody>();
        rb.centerOfMass = centerOfMass.transform.localPosition;
    }
    private void FixedUpdate()
    {
        speedLabel.text = $"Speed: {(rb.velocity.magnitude * 4f).ToString("0.0")}km/h";
    }
    private void Update()
    {
        CorrelateColliderAndTransform(lb_collider, lb_transform);
        CorrelateColliderAndTransform(lf_collider, lf_transform);
        CorrelateColliderAndTransform(rb_collider, rb_transform, true);
        CorrelateColliderAndTransform(rf_collider, rf_transform, true);

        lb_collider.motorTorque = motorTorque;
        rb_collider.motorTorque = motorTorque;
        lf_collider.steerAngle = steerAngle;
        rf_collider.steerAngle = steerAngle;

        if(cameraView.enabled)
        {
            if (cameraView.texture != null)
                Destroy(cameraView.texture);
            cameraView.texture = camSensor.Capture();
        }
        
    }

    /// <summary>
    /// Strength is a value in range [-1, 1]
    /// </summary>
    /// <param name="strength"></param>
    public void Accelerate(float strength)
    {
        motorTorque = maxSpeed * strength;


    }
    /// <summary>
    /// Strength is a value in range [-1, 1]
    /// </summary>
    /// <param name="strength"></param>
    public void Steer(float strength)
    {
        steerAngle = maxSteerAngle * strength;
    }

    public void Break(bool doBreak)
    {
        if(doBreak)
            motorTorque = 0f;
    }

    private static void CorrelateColliderAndTransform(WheelCollider wheel_collider, Transform wheel_transform, bool yaxis180rot = false)
    {
        var pos = Vector3.zero;
        var rot = Quaternion.identity;

        wheel_collider.GetWorldPose(out pos, out rot);
        wheel_transform.position = pos;
        wheel_transform.rotation = rot * Quaternion.Euler(0, yaxis180rot ? 180f : 0f, 90f);
    }
}


