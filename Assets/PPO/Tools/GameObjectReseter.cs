using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    /// <summary>
    /// A tool used to reset all Transform and RigidBody[2D] components of a GameObject, including all it's children GameObjects of any degree.<br />
    /// </summary>
    public class GameObjectReseter
    {
        private Transform parent;
        private List<Transform> initialTransformsCopies;
        private List<Rigidbody> rigidBodies;
        private List<Rigidbody2D> rigidBodies2D;

        public GameObjectReseter(Transform parent)
        {
            this.parent = parent;
            this.initialTransformsCopies = new List<Transform>();
            this.rigidBodies = new List<Rigidbody>();
            this.rigidBodies2D = new List<Rigidbody2D>();
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
        }

        private void GetAllTransforms(Transform parent)
        {
            Transform clone = new GameObject($"DeepUnity - InitialTransformReference {parent.GetInstanceID()}").transform;

            // warning: do not assign the parent otherwise infinite loop
            clone.position = parent.position;
            clone.rotation = parent.rotation;
            clone.localScale = parent.localScale;
            clone.localRotation = parent.localRotation;
            clone.localEulerAngles = parent.localEulerAngles;

            initialTransformsCopies.Add(clone);


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
                    GetAllTransforms(child);
                }
            }
        private void GetAllRigidBodies2D(Transform parent)
            {
                Rigidbody2D[] rbs2D = parent.transform.GetComponents<Rigidbody2D>();
                rigidBodies2D.AddRange(rbs2D);
                foreach (Transform child in parent)
                {
                    GetAllTransforms(child);
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
                    rb.velocity = Vector3.zero;
                    rb.angularVelocity = Vector3.zero;                    
                }
            }
        private void ResetAllRigidBodies2D()
            {
                foreach (var rb2d in rigidBodies2D)
                {
                    rb2d.velocity = Vector2.zero;
                    rb2d.angularVelocity = 0f;
                }
            }

    }
}

