using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {
        private int[] InputShapeCache { get; set; }

        [SerializeField] private int startAxis;
        [SerializeField] private int endAxis;

        /// <summary>
        /// Use <b>negative</b> axis in order to avoid batch collision.
        /// </summary>
        /// <param name="startAxis"></param>
        /// <param name="endAxis"></param>
        public Flatten(int startAxis, int endAxis)
        {
            this.startAxis = startAxis;
            this.endAxis = endAxis;
        }

        public Tensor Predict(Tensor input)
        {
            return input.Flatten(startAxis, endAxis);

        }
        public Tensor Forward(Tensor input)
        {
            InputShapeCache = input.Shape;
            return input.Flatten(startAxis, endAxis);           
        }
        public Tensor Backward(Tensor loss)
        {
            return loss.Reshape(InputShapeCache);
        }

    }

}