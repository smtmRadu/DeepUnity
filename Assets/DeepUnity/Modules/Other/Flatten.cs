using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class Flatten : IModule
    {
        [SerializeField] private int startAxis;
        [SerializeField] private int endAxis;

        public Flatten(int startAxis, int endAxis)
        {
            throw new NotImplementedException();
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