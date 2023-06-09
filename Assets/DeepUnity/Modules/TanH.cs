using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class TanH : ActivationBase, IModule
    {
        protected override void Activation(Tensor x)
        {
            x.ForEach(x =>
            {
                float e2x = Mathf.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return tanh;
            });
        }
        protected override void Derivative(Tensor x)
        {
            x.ForEach(x =>
            {
                float e2x = Mathf.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return 1f - tanh * tanh;
            });
        }
    }

}