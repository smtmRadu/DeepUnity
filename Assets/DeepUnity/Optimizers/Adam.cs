using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper

    [Serializable]
    public class Adam : IOptimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float alpha;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;

        
        public Adam(float learningRate = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f)
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
                Tensor mHat;
                Tensor vHat;

                // Update biased first momentum estimate
                L.m_W = beta1 * L.m_W + (1f - beta1) * L.g_W;

                // Update biased second raw momentum estimate
                L.v_W = beta2 * L.v_W + (1f - beta2) * Tensor.Pow(L.g_W, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = L.m_W / (1f - MathF.Pow(beta1, t));

                // Compute bias-corrected second raw momentum estimate
                vHat = L.v_W / (1f - MathF.Pow(beta2, t));

                // Update parameters
                L.t_W = L.t_W * (1f - weightDecay) - alpha * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);




                // Update biased first momentum estimate
                L.m_B = beta1 * L.m_B + (1f - beta1) * L.g_B;

                // Update biased second raw momentum estimate
                L.v_B = beta2 * L.v_B + (1f - beta2) * Tensor.Pow(L.g_B, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = L.m_B / (1f - MathF.Pow(beta1, t));

                // Compute bias-corrected second raw momentum estimate
                vHat = L.v_B / (1f - MathF.Pow(beta2, t));

                // Update parameters 
                L.t_B = L.t_B - alpha * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);

            });

        }
    }

}