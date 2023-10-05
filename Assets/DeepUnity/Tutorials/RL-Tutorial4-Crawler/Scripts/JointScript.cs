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
        }
        public void SetTargetAngularVelocity(float x, float y, float z)
        {
            joint.targetAngularVelocity = new Vector3(x, y, z);
        }
    }

}



