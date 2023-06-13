using System;
using UnityEngine;

namespace DeepUnity
{
    [Serializable]
    public class TanH : ActivationBase
    {
        protected override void Activation(NDArray x)
        {
            x.ForEach(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return tanh;
            });
        }
        protected override void Derivative(NDArray x)
        {
            x.ForEach(x =>
            {
                float e2x = MathF.Exp(2f * x);
                float tanh = (e2x - 1f) / (e2x + 1f);
                return 1f - tanh * tanh;
            });
        }
    }

}