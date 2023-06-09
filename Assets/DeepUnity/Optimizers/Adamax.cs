using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper, does not work idk why

    [Serializable]
    public class AdaMax : IOptimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float alpha;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;

        // 1st momentum buffer
        [NonSerialized] public Tensor[] m_W;
        [NonSerialized] public Tensor[] m_B;

        // exponentially weighted infinity norm
        [NonSerialized] public Tensor[] u_W;
        [NonSerialized] public Tensor[] u_B;


        public AdaMax(float learningRate = 0.002f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f)
        {
            throw new Exception("AdaMax not working...");
            this.t = 0;
            this.alpha = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;
        }

        public void Initialize(IModule[] modules)
        {
            m_W = new Tensor[modules.Length];
            m_B = new Tensor[modules.Length];

            u_W = new Tensor[modules.Length];
            u_B = new Tensor[modules.Length];

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense D)
                {
                    int inputs = D.param_W.Shape[1];
                    int outputs = D.param_W.Shape[0];

                    m_W[i] = Tensor.Zeros(outputs, inputs);
                    m_B[i] = Tensor.Zeros(outputs);

                    u_W[i] = Tensor.Zeros(outputs, inputs);
                    u_B[i] = Tensor.Zeros(outputs);

                }
            }
        }


        public void Step(IModule[] modules)
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, modules.Length, i =>
            {
                if (modules[i] is Dense L)
                {
                    // Update biased first momentum estimate
                    m_W[i] = beta1 * m_W[i] + (1f - beta1) * L.grad_W;
                    m_B[i] = beta1 * m_B[i] + (1f - beta1) * L.grad_B;

                    // Update the exponentially weighted infinity norm
                    u_W[i] = Tensor.Max(beta2 * u_W[i], Tensor.Abs(L.grad_W));
                    u_B[i] = Tensor.Max(beta2 * u_B[i], Tensor.Abs(L.grad_B));

                    // Update parameters
                    L.param_W = L.param_W * (1f - weightDecay) - alpha / (1f - MathF.Pow(beta1, t)) * m_W[i] / u_W[i];
                    L.param_B = L.param_B - alpha / (1f - MathF.Pow(beta1, t)) * m_B[i] / u_B[i];
                }
            });


        }
    }

}