using UnityEngine;

namespace DeepUnityTutorials
{
    public class JointScript : MonoBehaviour
    {
        public ConfigurableJoint joint { get; private set; }
        public Rigidbody rb { get; private set; }
        private void Awake()
        {
            joint = GetComponent<ConfigurableJoint>();
            rb = GetComponent<Rigidbody>();

            var jd = new JointDrive() { positionDamper = 1000, maximumForce = 3.402823e+38f };
            joint.angularXDrive = jd;

            var jd2 = new JointDrive() { positionDamper = 1000, maximumForce = 3.402823e+38f };
            joint.angularYZDrive = jd2;
        }
        public void SetAngularVelocity(float x, float y, float z)
        {
            joint.targetAngularVelocity = new Vector3(x, y, z);
        }
    }

}


