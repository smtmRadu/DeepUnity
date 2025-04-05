using System;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning
{
    [Serializable]
    public class OUNoise 
    {
        [SerializeField] public float mu=0f, theta=0.15f, sigma=0.1f, dt = 1e-2f;
        [SerializeField] public float? x0 = null;
        [SerializeField] int size=1;
        [SerializeField] Tensor xt;

        /// <summary>
        /// 
        /// </summary>
        /// <param name="size"></param>
        /// <param name="mu"></param>
        /// <param name="sigma"></param>
        /// <param name="theta"></param>
        /// <param name="dt"></param>
        /// <param name="x0">Initial state of the OU process.</param>
        public OUNoise(int size=1, float mu=0f, float sigma = 0.1f, float theta=0.15f, float dt = 1e-2f, float? x0 = null)
        {
            this.size = size;
            this.mu = mu;
            this.theta = theta;
            this.sigma = sigma;
            this.dt = dt;
            this.x0 = x0;
            Reset();

        }
        public void Reset() => xt = this.x0 == null ? Tensor.Zeros(this.size) : Tensor.Fill(x0.Value, this.size);
        

        public Tensor Sample()
        {
            var dxt = theta * (mu - xt) * dt + MathF.Sqrt(dt) * sigma * Tensor.RandomNormal(this.size);
            xt += dxt;
            return xt;
        }
    }
}

