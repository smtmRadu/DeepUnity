using UnityEngine;

public class TestJoint : MonoBehaviour
{
	public float rotationSpeed = 1f;
	public HingeJoint joint;
    public CharacterJoint characterJoint;
    public JointMotor jointMotor;
    public ConfigurableJoint configurableJoint;
    private void Update()
    {
        configurableJoint.targetRotation = new Quaternion();
    }

}


