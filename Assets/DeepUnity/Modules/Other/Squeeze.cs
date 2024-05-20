using UnityEngine;
using System;

namespace DeepUnity.Modules
{
    [Serializable]
    public class Squeeze : IModule
    {
        [SerializeField] private int axis;

        private int[] InputShape { get; set; } = null;
        public Squeeze(int axis)
        {
            if (axis >= 0)
                throw new ArgumentException("Squeeze layer allows only negative axis to avoid batched/non-batched input coincidence");

            throw new System.NotSupportedException("Squeeze layer is deprecated due to axis-batch problem. Use reshape instead");
            this.axis = axis;
        }

        public Tensor Predict(Tensor input)
        {
            return input.Squeeze(axis);
        }
        public Tensor Forward(Tensor input)
        {
            InputShape = input.Shape;
            return input.Squeeze(axis);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY.Reshape(InputShape);
        }
        public object Clone()
        {
            return new Squeeze(axis);
        }
    }

}


