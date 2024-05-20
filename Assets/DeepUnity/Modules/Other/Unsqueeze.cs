using UnityEngine;
using System;

namespace DeepUnity.Modules
{
    [Serializable]
    public class Unsqueeze : IModule
    {
        [SerializeField] private int axis;

        private int[] InputShape { get; set; } = null;

        public Unsqueeze(int axis)
        {
            throw new System.NotSupportedException("Unsqueeze layer is deprecated due to axis-batch problem. Use reshape instead");
            this.axis = axis;
        }

        public Tensor Predict(Tensor input)
        {      
            return input.Unsqueeze(axis);
        }
        public Tensor Forward(Tensor input)
        {
            InputShape = input.Shape;
            return input.Unsqueeze(axis);
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY.Reshape(InputShape);
        }
        public object Clone()
        {
            return new Unsqueeze(axis);
        }
    }

}


