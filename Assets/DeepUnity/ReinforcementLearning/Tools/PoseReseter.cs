using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    /// <summary>
    /// A tool used to reset all Transform, RigidBody[2D] and BodyController components of a GameObject, including all it's children GameObjects of any degree.<br />
    /// </summary>
    public class PoseReseter
    {
        private static GameObject Instance;
        private Transform parent;
        private List<Transform> initialTransformsCopies;
        private List<Rigidbody> rigidBodies;
        private List<Rigidbody2D> rigidBodies2D;
        private BodyController bodyController;

        public PoseReseter(Transform parent)
        {
            if (Instance == null)
            {
                Instance = new GameObject("[DeepUnity] Initial Transform References");
            }

            this.parent = parent;
            initialTransformsCopies = new List<Transform>();
            rigidBodies = new List<Rigidbody>();
            rigidBodies2D = new List<Rigidbody2D>();
            bodyController = parent.GetComponent<BodyController>();
            GetAllTransforms(parent);
            GetAllRigidBodies(parent);
            GetAllRigidBodies2D(parent);
        }
        public void Reset()
        {
            int transformsStart = 0;
            ResetAllTransforms(parent, ref transformsStart);
            ResetAllRigidBodies();
            ResetAllRigidBodies2D();
            bodyController?.bodyPartsList.ForEach(part =>
            {
                part.CurrentStrength = 0f;
                part.CurrentNormalizedStrength = 0f;
                part.CurrentEulerRotation = Vector3.zero;
                part.CurrentNormalizedEulerRotation = Vector3.zero;
            });
        }

        private void GetAllTransforms(Transform parent)
        {
            Transform transformClone = new GameObject($"[DeepUnity] To GameObject: {parent.gameObject.name}").transform;

            // warning: do not assign the parent otherwise infinite loop
            transformClone.parent = Instance.transform;
            transformClone.position = parent.position;
            transformClone.rotation = parent.rotation;
            transformClone.localScale = parent.localScale;
            transformClone.localRotation = parent.localRotation;
            transformClone.localEulerAngles = parent.localEulerAngles;

            initialTransformsCopies.Add(transformClone);


            foreach (Transform child in parent)
            {
                GetAllTransforms(child);
            }
        }
        private void GetAllRigidBodies(Transform parent)
        {
            Rigidbody[] rbs = parent.transform.GetComponents<Rigidbody>();
            rigidBodies.AddRange(rbs);
            foreach (Transform child in parent)
            {
                GetAllRigidBodies(child);
            }
        }
        private void GetAllRigidBodies2D(Transform parent)
        {
            Rigidbody2D[] rbs2D = parent.transform.GetComponents<Rigidbody2D>();
            rigidBodies2D.AddRange(rbs2D);
            foreach (Transform child in parent)
            {
                GetAllRigidBodies2D(child);
            }
        }

        private void ResetAllTransforms(Transform parent, ref int index)
        {
            Transform initialTransform = initialTransformsCopies[index++];

            parent.position = initialTransform.position;
            parent.rotation = initialTransform.rotation;
            parent.localScale = initialTransform.localScale;
            parent.localRotation = initialTransform.localRotation;
            parent.localEulerAngles = initialTransform.localEulerAngles;

            foreach (Transform child in parent)
            {
                ResetAllTransforms(child, ref index);
            }

        }
        private void ResetAllRigidBodies()
        {
            foreach (var rb in rigidBodies)
            {
                if (rb.isKinematic)
                    continue;
                rb.velocity = Vector3.zero;
                rb.angularVelocity = Vector3.zero;
            }
        }
        private void ResetAllRigidBodies2D()
        {
            foreach (var rb2d in rigidBodies2D)
            {
                if (rb2d.isKinematic)
                    continue;
                rb2d.velocity = Vector2.zero;
                rb2d.angularVelocity = 0f;
            }
        }

    }
}

