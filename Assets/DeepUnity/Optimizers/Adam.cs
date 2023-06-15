using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper

    [Serializable]
    public class Adam : IOptimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float learningRate;
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;
        [SerializeField] private float weightDecay;

        // 1st momentum buffer
        [NonSerialized] public Tensor[] m_W;
        [NonSerialized] public Tensor[] m_B;

        // 2nd momentum buffer 
        [NonSerialized] public Tensor[] v_W;
        [NonSerialized] public Tensor[] v_B;


        public Adam(float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f)
        {
            this.t = 0;
            this.learningRate = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.weightDecay = weightDecay;
        }
        public void Initialize(IModule[] modules)
        {
            m_W = new Tensor[modules.Length];
            m_B = new Tensor[modules.Length];

            v_W = new Tensor[modules.Length];
            v_B = new Tensor[modules.Length];

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense D)
                {
                    int inputs = D.weights.Shape.height;
                    int outputs = D.weights.Shape.width;

                    m_W[i] = Tensor.Zeros(inputs, outputs);
                    m_B[i] = Tensor.Zeros(outputs);

                    v_W[i] = Tensor.Zeros(inputs, outputs);
                    v_B[i] = Tensor.Zeros(outputs);

                }
                else if (modules[i] is BatchNorm B)
                {
                    int inputs = B.beta.Shape.height;

                    m_W[i] = Tensor.Zeros(inputs); // W is for gamma
                    m_B[i] = Tensor.Zeros(inputs); // B is for beta

                    v_W[i] = Tensor.Zeros(inputs);
                    v_B[i] = Tensor.Zeros(inputs);
                }

            }
        }

        public void Step(IModule[] modules)
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, modules.Length, i =>
            {
                if (modules[i] is Dense D)
                {
                    // keep W and B separately just to use this 2 variables for both cases
                    Tensor mHat;
                    Tensor vHat;

                    // Update biased first momentum estimate
                    m_W[i] = beta1 * m_W[i] + (1f - beta1) * D.grad_Weights;

                    // Update biased second raw momentum estimate
                    v_W[i] = beta2 * v_W[i] + (1f - beta2) * Tensor.Pow(D.grad_Weights, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_W[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_W[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters
                    D.weights = D.weights * (1f - weightDecay) - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);




                    // Update biased first momentum estimate
                    m_B[i] = beta1 * m_B[i] + (1f - beta1) * D.grad_Biases;

                    // Update biased second raw momentum estimate
                    v_B[i] = beta2 * v_B[i] + (1f - beta2) * Tensor.Pow(D.grad_Biases, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_B[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_B[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters 
                    D.biases = D.biases - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);
                }
                else if (modules[i] is BatchNorm BN)
                {
                    // keep Gamma and Beta separately just to use this 2 variables for both cases
                    // W is for gamma, B is for Beta
                    Tensor mHat;
                    Tensor vHat;

                    // Update biased first momentum estimate
                    m_W[i] = beta1 * m_W[i] + (1f - beta1) * BN.grad_Gamma;

                    // Update biased second raw momentum estimate
                    v_W[i] = beta2 * v_W[i] + (1f - beta2) * Tensor.Pow(BN.grad_Gamma, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_W[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_W[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters
                    BN.gamma = BN.gamma * (1f - weightDecay) - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);




                    // Update biased first momentum estimate
                    m_B[i] = beta1 * m_B[i] + (1f - beta1) * BN.grad_Beta;

                    // Update biased second raw momentum estimate
                    v_B[i] = beta2 * v_B[i] + (1f - beta2) * Tensor.Pow(BN.grad_Beta, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_B[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_B[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters 
                    BN.beta = BN.beta - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);
                }
            });

        }
    }

}