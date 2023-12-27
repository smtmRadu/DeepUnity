using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity
{
    public class BodyPart
    {
        /// <summary>
        /// The GameObject that is the bodypart.
        /// </summary>
        public GameObject gameObject;
        /// <summary>
        /// The Rigidbody attached to this BodyPart GameObject (if is any).
        /// </summary>
        public Rigidbody rb { get; set; }
        /// <summary>
        /// The ConfigurableJoint attached to this GameObject (if is any).
        /// </summary>
        public ConfigurableJoint Joint { get; set; }

        /// <summary>
        /// Automatically added if missing.
        /// </summary>
        public GroundContact GroundContact { get; set; }
        /// <summary>
        /// Automatically added if missing.
        /// </summary>
        public TargetContact TargetContact { get; set; }
        /// <summary>
        /// The BodyController that controls this BodyPart.
        /// </summary>
        [HideInInspector] public BodyController Controller { get; set; }

        public Vector3 CurrentEulerRotation { get; set; } = Vector3.zero;
        public Vector3 CurrentNormalizedRotation { get; set; } = Vector3.zero;
        public float CurrentStrength { get; set; } = 0f;
        public float CurrentNormalizedStrength { get; set; } = 0f;

        /// <summary>
        /// X, Y, Z : values in range [-1, 1]
        /// </summary>
        public void SetJointTargetRotation(float x, float y, float z)
        {
            if (float.IsNaN(x))
                x = 0;
            if (float.IsNaN(y))
                y = 0;
            if (float.IsNaN(z))
                z = 0;

            x = (x + 1f) / 2f;
            y = (y + 1f) / 2f;
            z = (z + 1f) / 2f;

            CurrentEulerRotation = new Vector3(
                Mathf.Lerp(Joint.lowAngularXLimit.limit, Joint.highAngularXLimit.limit, x),
                Mathf.Lerp(-Joint.angularYLimit.limit, Joint.angularYLimit.limit, y),
                Mathf.Lerp(-Joint.angularZLimit.limit, Joint.angularZLimit.limit, z));

            Joint.targetRotation = Quaternion.Euler(CurrentEulerRotation);
  
            CurrentNormalizedRotation = new Vector3(
                Mathf.InverseLerp(Joint.lowAngularXLimit.limit, Joint.highAngularXLimit.limit, CurrentEulerRotation.x),
                Mathf.InverseLerp(-Joint.angularYLimit.limit, Joint.angularYLimit.limit, CurrentEulerRotation.y),
                Mathf.InverseLerp(-Joint.angularZLimit.limit, Joint.angularZLimit.limit, CurrentEulerRotation.z));
        }
        /// <summary>
            /// Strength : value in range [-1, 1]
            /// </summary>
            /// <param name="strength"></param>
        public void SetJointStrength(float strength)
        {
            if (float.IsNaN(strength))
                strength = 0f;
                var rawVal = (strength + 1f) * 0.5f * Controller.maxJointForce;
                var jd = new JointDrive
                {
                    positionSpring = Controller.maxJointSpring,
                    positionDamper = Controller.jointDamper,
                    maximumForce = rawVal
                };

                Joint.slerpDrive = jd;

                CurrentStrength = jd.maximumForce;
                CurrentNormalizedStrength = CurrentStrength / Controller.maxJointForce;
        }
    }

    /// <summary>
    /// Setup:
    /// 1. Twitch settings.
    /// 2. Add each GameObject/Transform body part of the character.
    /// 3. To each body part is automatically added a groundContact and targetContact script. 
    ///    For custom contacts, either add them for each body part in the inspector and modify them, or through code control.
    /// </summary>

    public class BodyController : MonoBehaviour
    {
        public float maxJointSpring = 3000f; // 10_000f;
        public float jointDamper = 100f; // 500f;
        public float maxJointForce = 6000f; // 25_000f;
    
        [HideInInspector] public Dictionary<Transform, BodyPart> bodyPartsDict = new Dictionary<Transform, BodyPart>();
        [HideInInspector] public List<BodyPart> bodyPartsList = new List<BodyPart>();

        public void AddBodyPart(Transform bodyPart)
        {
            BodyPart bp = new BodyPart
            {
                gameObject = bodyPart.gameObject,
                rb = bodyPart.GetComponent<Rigidbody>(),
                Joint = bodyPart.GetComponent<ConfigurableJoint>(),
            };
            if(bp.Joint)
                bp.Joint.rotationDriveMode = RotationDriveMode.Slerp;
            bp.rb.maxAngularVelocity = 100;

            // Add & setup the ground contact script
            bp.GroundContact = bodyPart.GetComponent<GroundContact>();
            if (!bp.GroundContact)
                bp.GroundContact = bodyPart.gameObject.AddComponent<GroundContact>();
            bp.GroundContact.agent = gameObject.GetComponent<Agent>();


            // Add & setup the target contact script
            bp.TargetContact = bodyPart.GetComponent<TargetContact>();
            if (!bp.TargetContact)
                bp.TargetContact = bodyPart.gameObject.AddComponent<TargetContact>();
            bp.TargetContact.agent = gameObject.GetComponent<Agent> ();

            bp.Controller = this;
            bodyPartsDict.Add(bodyPart, bp);
            bodyPartsList.Add(bp);
        }
        public void AddBodyPart(GameObject bodyPart) => AddBodyPart(bodyPart.transform);
    }


#if UNITY_EDITOR

    [CustomEditor(typeof(BodyController), true), CanEditMultipleObjects]
    sealed class BodyControllerEditor : Editor
    {
        public override void OnInspectorGUI()
        {
            string[] dontDrawMe = new string[] { "m_Script" };

            serializedObject.Update();
            DrawPropertiesExcluding(serializedObject, dontDrawMe);

            serializedObject.ApplyModifiedProperties();
        }
    }

#endif
}