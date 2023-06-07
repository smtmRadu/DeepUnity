using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper

    [Serializable]
    public class AdaMax : IOptimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float alpha;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;


        public AdaMax(float learningRate = 0.002f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f)
        {
            this.t = 0;
            this.alpha = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            t++;

            System.Threading.Tasks.Parallel.ForEach(layers, L =>
            {
                // Update biased first momentum estimate
                L.m_W = beta1 * L.m_W + (1f - beta1) * L.g_W;
                L.m_B = beta1 * L.m_B + (1f - beta1) * L.g_W;
                
                // Update the exponentially weighted infinity norm
                L.v_W = Tensor.Max(beta2 * L.v_W, Tensor.Abs(L.g_W));
                L.v_B = Tensor.Max(beta2 * L.v_B, Tensor.Abs(L.g_B));

                // Update parameters
                L.t_W = L.t_W * (1f - weightDecay) - (alpha / (1f - MathF.Pow(beta1, t)) * L.m_W / L.v_W);
                L.t_B = L.t_B - (alpha /  (1f - MathF.Pow(beta1, t)) * L.m_B / L.v_B);

            });

        }
    }

}