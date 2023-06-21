using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper, does not work idk why

    public class AdaMax : Optimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;

        // 1st momentum buffer
        [NonSerialized] public Tensor[] m_W;
        [NonSerialized] public Tensor[] m_B;

        // exponentially weighted infinity norm
        [NonSerialized] public Tensor[] u_W;
        [NonSerialized] public Tensor[] u_B;


        public AdaMax(Learnable[] parameters, float learningRate = 0.002f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f)
        {
            this.t = 0;
            this.learningRate = learningRate;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;



            this.parameters = parameters;

            m_W = new Tensor[parameters.Length];
            m_B = new Tensor[parameters.Length];

            u_W = new Tensor[parameters.Length];
            u_B = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is Learnable P)
                {
                    m_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    m_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                    u_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    u_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                }
            }
        }


        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is Learnable P)
                {
                    // Update biased first momentum estimate
                    m_W[i] = beta1 * m_W[i] + (1f - beta1) * P.gradGamma;
                    m_B[i] = beta1 * m_B[i] + (1f - beta1) * P.gradBeta;

                    // Update the exponentially weighted infinity norm
                    u_W[i] = Tensor.Max(beta2 * u_W[i], Tensor.Abs(P.gradGamma));
                    u_B[i] = Tensor.Max(beta2 * u_B[i], Tensor.Abs(P.gradBeta));

                    // Update parameters
                    P.gamma = P.gamma * (1f - weightDecay) - (learningRate / (1f - MathF.Pow(beta1, t))) * m_W[i] / u_W[i];
                    P.beta = P.beta - (learningRate / (1f - MathF.Pow(beta1, t))) * m_B[i] / u_B[i];
                }
            });


        }
    }

}