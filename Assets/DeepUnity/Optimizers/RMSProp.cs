using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    [Serializable]
    public sealed class RMSProp : IOptimizer
    {
        [SerializeField] private float learningRate;
        [SerializeField] private float alpha;
        [SerializeField] private float momentum;
        [SerializeField] private float weightDecay;

        public RMSProp(float learningRate = 0.01f, float alpha = 0.99f, float momentum = 0.9f, float weightDecay = 1e-5f)
        {
            this.learningRate = learningRate;
            this.alpha = alpha;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
        }

        public void Step(Dense[] layers)
        {
            int channels = layers[0].InputCache.Shape[1];

            
            System.Threading.Tasks.Parallel.ForEach(layers, L =>
            {
                if (weightDecay != 0)
                {
                    L.g_W += L.t_W * weightDecay;
                    // we do not apply decay on biases
                }

                L.v_W = alpha * L.v_W + (1f - alpha) * Tensor.Pow(L.g_W, 2);
                L.v_B = alpha * L.v_B + (1f - alpha) * Tensor.Pow(L.g_B, 2);

                // centered?... let's say pass

                if(momentum > 0)
                {
                    // In RMSProp i use v as momentums, and m as buffers. (m and v are from dense)
                    L.m_W = momentum * L.m_W + L.g_W / (Tensor.Sqrt(L.v_W) + Utils.EPSILON);
                    L.m_B = momentum * L.m_B + L.g_B / (Tensor.Sqrt(L.v_B) + Utils.EPSILON);

                    L.t_W = L.t_W - learningRate * L.m_W;
                    L.t_B = L.t_B - learningRate * L.m_B;
                }
                else
                {
                    L.t_W = L.t_W - learningRate * L.g_W / (Tensor.Sqrt(L.v_W) + Utils.EPSILON);
                    L.t_B = L.t_B - learningRate * L.g_B / (Tensor.Sqrt(L.v_B) + Utils.EPSILON);
                }

            });
        }
    }

}