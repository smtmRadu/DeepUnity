using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {

        [SerializeField] private Dim startDim;
        [SerializeField] private Dim endDim;
        [SerializeField] private int startAxis;
        [SerializeField] private int endAxis;

        public Flatten(Dim startDim, Dim endDim)
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