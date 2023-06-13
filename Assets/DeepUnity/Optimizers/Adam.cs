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
        [NonSerialized] public NDArray[] m_W;
        [NonSerialized] public NDArray[] m_B;

        // 2nd momentum buffer 
        [NonSerialized] public NDArray[] v_W;
        [NonSerialized] public NDArray[] v_B;


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
            m_W = new NDArray[modules.Length];
            m_B = new NDArray[modules.Length];

            v_W = new NDArray[modules.Length];
            v_B = new NDArray[modules.Length];

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense D)
                {
                    int inputs = D.weights.Shape[1];
                    int outputs = D.weights.Shape[0];

                    m_W[i] = NDArray.Zeros(outputs, inputs);
                    m_B[i] = NDArray.Zeros(outputs);

                    v_W[i] = NDArray.Zeros(outputs, inputs);
                    v_B[i] = NDArray.Zeros(outputs);

                }
                else if (modules[i] is BatchNorm B)
                {
                    int inputs = B.beta.Shape[0];

                    m_W[i] = NDArray.Zeros(inputs); // W is for gamma
                    m_B[i] = NDArray.Zeros(inputs); // B is for beta

                    v_W[i] = NDArray.Zeros(inputs);
                    v_B[i] = NDArray.Zeros(inputs);
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
                    NDArray mHat;
                    NDArray vHat;

                    // Update biased first momentum estimate
                    m_W[i] = beta1 * m_W[i] + (1f - beta1) * D.grad_Weights;

                    // Update biased second raw momentum estimate
                    v_W[i] = beta2 * v_W[i] + (1f - beta2) * NDArray.Pow(D.grad_Weights, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_W[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_W[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters
                    D.weights = D.weights * (1f - weightDecay) - learningRate * mHat / (NDArray.Sqrt(vHat) + Utils.EPSILON);




                    // Update biased first momentum estimate
                    m_B[i] = beta1 * m_B[i] + (1f - beta1) * D.grad_Biases;

                    // Update biased second raw momentum estimate
                    v_B[i] = beta2 * v_B[i] + (1f - beta2) * NDArray.Pow(D.grad_Biases, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_B[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_B[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters 
                    D.biases = D.biases - learningRate * mHat / (NDArray.Sqrt(vHat) + Utils.EPSILON);
                }
                else if (modules[i] is BatchNorm BN)
                {
                    // keep Gamma and Beta separately just to use this 2 variables for both cases
                    // W is for gamma, B is for Beta
                    NDArray mHat;
                    NDArray vHat;

                    // Update biased first momentum estimate
                    m_W[i] = beta1 * m_W[i] + (1f - beta1) * BN.grad_Gamma;

                    // Update biased second raw momentum estimate
                    v_W[i] = beta2 * v_W[i] + (1f - beta2) * NDArray.Pow(BN.grad_Gamma, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_W[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_W[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters
                    BN.gamma = BN.gamma * (1f - weightDecay) - learningRate * mHat / (NDArray.Sqrt(vHat) + Utils.EPSILON);




                    // Update biased first momentum estimate
                    m_B[i] = beta1 * m_B[i] + (1f - beta1) * BN.grad_Beta;

                    // Update biased second raw momentum estimate
                    v_B[i] = beta2 * v_B[i] + (1f - beta2) * NDArray.Pow(BN.grad_Beta, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = m_B[i] / (1f - MathF.Pow(beta1, t));

                    // Compute bias-corrected second raw momentum estimate
                    vHat = v_B[i] / (1f - MathF.Pow(beta2, t));

                    // Update parameters 
                    BN.beta = BN.beta - learningRate * mHat / (NDArray.Sqrt(vHat) + Utils.EPSILON);
                }
            });

        }
    }

}