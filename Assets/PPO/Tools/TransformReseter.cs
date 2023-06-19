using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    namespace NeuroForge
    {
        public class TransformReseter
        {
            private Transform parent;
            private List<Transform> initialTransforms;

            public TransformReseter(Transform parent)
            {
                this.parent = parent;
                this.initialTransforms = new List<Transform>();
                FindAllTransforms(parent);
            }
            public void Reset()
            {
                int transformsStart = 0;
                ResetAllTransforms(parent, ref transformsStart);
            }

            private void FindAllTransforms(Transform parent)
            {
                foreach (Transform child in parent)
                {
                    Transform clone = new GameObject("NeuroForge - InitialTransformReference").transform;

                    // warning: do not assign the parent otherwise infinite loop
                    clone.position = child.position;
                    clone.rotation = child.rotation;
                    clone.localScale = child.localScale;
                    clone.localRotation = child.localRotation;
                    clone.localEulerAngles = child.localEulerAngles;

                    initialTransforms.Add(clone);
                    FindAllTransforms(child);
                }
            }
            private void ResetAllTransforms(Transform parent, ref int index)
            {
                for (int i = 0; i < parent.transform.childCount; i++)
                {
                    Transform child = parent.transform.GetChild(i);
                    Transform initialTransform = initialTransforms[index++];

                    child.position = initialTransform.position;
                    child.rotation = initialTransform.rotation;
                    child.localScale = initialTransform.localScale;
                    child.localRotation = initialTransform.localRotation;
                    child.localEulerAngles = initialTransform.localEulerAngles;

                    ResetAllTransforms(child, ref index);
                }
            }
        }

    }

}

