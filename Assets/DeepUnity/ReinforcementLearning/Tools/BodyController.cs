using System.Collections.Generic;
using UnityEditor;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    public class BodyPart
    {
        /// <summary>
        /// The GameObject that is the bodypart.
        /// </summary>
        public GameObject gameObject { get; set; }
        /// <summary>
        /// The Rigidbody attached to this BodyPart GameObject (if is any).
        /// </summary>
        public Rigidbody rigidbody { get; set; }
        /// <summary>
        /// The ConfigurableJoint attached to this GameObject (if is any).
        /// </summary>
        public ConfigurableJoint Joint { get; set; }
        /// <summary>
        /// Allows events to happen when this bodypart enters in contact with a collider. Also checks for grounding. Automatically added to the BodyPart gameObject.
        /// </summary>
        public ColliderContact ColliderContact { get; set; }
        /// <summary>
        /// Allows events to happen when this bodypart enters in contact with a trigger. Automatically added to the BodyPart gameObject.
        /// </summary>
        public TriggerContact TriggerContact { get; set; }
        /// <summary>
        /// The BodyController that controls this BodyPart.
        /// </summary>
        [HideInInspector] public BodyController Controller { get; set; }

        public Vector3 CurrentEulerRotation { get; set; } = Vector3.zero;
        public Vector3 CurrentNormalizedEulerRotation { get; set; } = Vector3.zero;
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

            CurrentNormalizedEulerRotation = new Vector3(
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
}

namespace DeepUnity.ReinforcementLearning
{
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
        /// <summary>
        /// Returns the bodyPart object created from the given GameObject
        /// </summary>
        /// <param name="bodyPart"></param>
        /// <returns></returns>
        public BodyPart this[GameObject bodyPart] { get => bodyPartsDict[bodyPart]; }
        [HideInInspector] public Dictionary<GameObject, BodyPart> bodyPartsDict = new Dictionary<GameObject, BodyPart>();
        [HideInInspector] public List<BodyPart> bodyPartsList = new List<BodyPart>();

        private void AddBodyPart(Transform bodyPart)
        {
            BodyPart bp = new BodyPart
            {
                gameObject = bodyPart.gameObject,
                rigidbody = bodyPart.GetComponent<Rigidbody>(),
                Joint = bodyPart.GetComponent<ConfigurableJoint>(),
            };
            if (bp.Joint)
                bp.Joint.rotationDriveMode = RotationDriveMode.Slerp;
            bp.rigidbody.maxAngularVelocity = 100;

            // Add & setup the collision methods
            bp.ColliderContact = bodyPart.GetComponent<ColliderContact>();
            if (!bp.ColliderContact)
                bp.ColliderContact = bodyPart.gameObject.AddComponent<ColliderContact>();

            // Add & setup the trigger methods
            bp.TriggerContact = bodyPart.GetComponent<TriggerContact>();
            if (!bp.TriggerContact)
                bp.TriggerContact = bodyPart.gameObject.AddComponent<TriggerContact>();


            bp.Controller = this;
            bodyPartsDict.Add(bodyPart.gameObject, bp);
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