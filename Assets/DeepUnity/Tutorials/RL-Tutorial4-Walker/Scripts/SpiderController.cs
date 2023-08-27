using UnityEngine;

public class SpiderController : MonoBehaviour
{
	[SerializeField] JointScript shin1;
	[SerializeField] JointScript shin2;
	[SerializeField] JointScript shin3;
	[SerializeField] JointScript shin4;

	[SerializeField] JointScript thigh1;
	[SerializeField] JointScript thigh2;
	[SerializeField] JointScript thigh3;
	[SerializeField] JointScript thigh4;

	[SerializeField] float speed = 5f;


    public void Update()
    {
		float hor = Input.GetAxis("Horizontal") * 10f;
		float vert = Input.GetAxis("Vertical") * 10f;

		shin1.SetTargetAngularVelocity(hor, 0, 0);
		shin2.SetTargetAngularVelocity(hor, 0, 0);
		shin3.SetTargetAngularVelocity(hor, 0, 0);
		shin4.SetTargetAngularVelocity(hor, 0, 0);

		thigh1.SetTargetAngularVelocity(vert, 0, 0);
		thigh2.SetTargetAngularVelocity(vert, 0, 0);
        thigh3.SetTargetAngularVelocity(vert, 0, 0);
        thigh4.SetTargetAngularVelocity(vert, 0, 0);
    }
}


