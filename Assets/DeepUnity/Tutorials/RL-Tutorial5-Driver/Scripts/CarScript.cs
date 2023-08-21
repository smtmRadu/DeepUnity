using UnityEngine;
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
    public CameraSensor camSensor;
    public Text speedLabel;
    public RawImage cameraView;

    public float maxMotorTorque = 1000f;
    public float maxSteerAngle = 35f;
    public float breakPower = 1000f;


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

        if(cameraView.enabled)
        {
            if(cameraView.texture == null)
                cameraView.texture = new Texture2D(camSensor.Width, camSensor.Height);

            Texture2D camViewTex2D = (Texture2D) cameraView.texture;
            camViewTex2D.SetPixels(camSensor.GetObservationPixels());
            camViewTex2D.Apply();
        }
        
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
        wheel_transform.rotation = rot * Quaternion.Euler(0, yaxis180rot ? 180f : 0f, 90f);
    }
}


