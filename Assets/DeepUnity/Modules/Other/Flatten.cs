using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {

        [SerializeField] TDim startDim;
        [SerializeField] TDim endDim;
        [SerializeField] int startAxis;
        [SerializeField] int endAxis;

        public Flatten(TDim startDim, TDim endDim)
        {

        }
        public Flatten(int startAxis, int endAxis)
        {

        }
        public Tensor Predict(Tensor input)
        {
            return null;
        }
        public Tensor Forward(Tensor input)
        {
            return null;
        }
        public Tensor Backward(Tensor loss)
        {
            return null;
        }

    }

}