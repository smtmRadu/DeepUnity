using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
    [Serializable]
    public class Adagrad : IOptimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float learningRate;
        [SerializeField] private float learningRateDecay;
        [SerializeField] private float weightDecay;


        public Adagrad(float learningRate = 0.01f, float learningRateDecay = 0f, float weightDecay = 0f)
        {
            this.t = 0;
            this.learningRate = learningRate;
            this.learningRateDecay = learningRateDecay;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            t++;

            System.Threading.Tasks.Parallel.ForEach(layers, L =>
            {
                var gammaBar = learningRate / (1 + (t - 1) * learningRateDecay);

                if(weightDecay != 0f)
                {
                    L.g_W = L.g_W + weightDecay * L.g_W;
                }

                // state_sum is hold in m
                L.m_W = L.m_W + Tensor.Pow(L.g_W, 2f);
                L.m_B = L.m_B + Tensor.Pow(L.g_B, 2f);

                L.t_W = L.t_W - gammaBar * (L.g_W / (Tensor.Sqrt(L.m_W) + 1e-10f));
                L.t_B = L.t_B - gammaBar * (L.g_B / (Tensor.Sqrt(L.m_B) + 1e-10f));

            });
        }
    }
}

