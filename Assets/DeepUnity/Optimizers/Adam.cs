using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper

    public sealed class Adam : Optimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;

        // 1st momentum buffer
        [NonSerialized] public Tensor[] m_W;
        [NonSerialized] public Tensor[] m_B;

        // 2nd momentum buffer 
        [NonSerialized] public Tensor[] v_W;
        [NonSerialized] public Tensor[] v_B;


        public Adam(Learnable[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f)
        {
            this.t = 0;
            this.learningRate = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;



            this.parameters = parameters;

            m_W = new Tensor[parameters.Length];
            m_B = new Tensor[parameters.Length];

            v_W = new Tensor[parameters.Length];
            v_B = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                Learnable P = parameters[i];


                m_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                m_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                v_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                v_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                Learnable P = parameters[i];

                Tensor mHat;
                Tensor vHat;

                // Update biased first momentum estimate
                m_W[i] = beta1 * m_W[i] + (1f - beta1) * P.gradGamma;

                // Update biased second raw momentum estimate
                v_W[i] = beta2 * v_W[i] + (1f - beta2) * Tensor.Pow(P.gradGamma, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = m_W[i] / (1f - MathF.Pow(beta1, t));

                // Compute bias-corrected second raw momentum estimate
                vHat = v_W[i] / (1f - MathF.Pow(beta2, t));

                // Update parameters
                P.gamma = P.gamma * (1f - weightDecay) - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);




                // Update biased first momentum estimate
                m_B[i] = beta1 * m_B[i] + (1f - beta1) * P.gradBeta;

                // Update biased second raw momentum estimate
                v_B[i] = beta2 * v_B[i] + (1f - beta2) * Tensor.Pow(P.gradBeta, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = m_B[i] / (1f - MathF.Pow(beta1, t));

                // Compute bias-corrected second raw momentum estimate
                vHat = v_B[i] / (1f - MathF.Pow(beta2, t));

                // Update parameters 
                P.beta = P.beta - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);
                
            });

        }
    }

}