using DeepUnity.Modules;
using System;
using UnityEngine;

namespace DeepUnity.Activations
{
    /// <summary>
    /// Applies the randomized leaky rectified linear unit activation function. The slope is 
    /// extracted from the uniform distribution U(<paramref name="lower"/>, <paramref name="upper"/>) on training,
    /// and on inference the slope is (<paramref name="lower"/> + <paramref name="upper"/>) / 2.
    /// </summary>
    [Serializable]
    public class RReLU : IModule, IActivation
    {
        [SerializeField] private float lower = 0.1249999999776482582f;
        [SerializeField] private float upper = 0.329999999776482582f;

        /// <summary>
        /// Applies the randomized leaky rectified linear unit activation function. The slope is 
        /// extracted from the uniform distribution U(<paramref name="lower"/>, <paramref name="upper"/>) on training,
        /// and on inference the slope is (<paramref name="lower"/> + <paramref name="upper"/>) / 2.
        /// </summary>
        public RReLU(float lower = 0.125f, float upper = 1f/3f)
        {
            this.lower = lower;
            this.upper = upper;
        }

        private Tensor InputCache { get; set; }
        private Tensor ACache { get; set; }
        public Tensor Predict(Tensor input)
        {
            float a = (lower + upper) / 2f;
            return input.Select(x =>
            {
                if (x >= 0)
                    return x;
                else
                    return a * x;
            });
        }

        public Tensor Forward(Tensor input)
        {
            ACache = Tensor.RandomRange((lower, upper), input.Shape);

            return input.Zip(ACache, (x, a) =>
            {
                if (x >= 0)
                    return x;
                else
                    return a * x;
            });
        }

        public Tensor Backward(Tensor dLdY)
        {
            return dLdY * InputCache.Zip(ACache, (x, a) =>
            {
                if (x > 0f)
                    return 1;
                else
                    return a;
            });
        }

        public object Clone() => new RReLU(lower, upper);
    }

}


