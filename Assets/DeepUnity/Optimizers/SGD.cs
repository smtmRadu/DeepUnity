using System;
using UnityEngine;
namespace DeepUnity
{
    // https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    [Serializable]
    public sealed class SGD : Optimizer
    {
        [SerializeField] private float momentum;
        [SerializeField] private float dampening;
        [SerializeField] private bool nesterov;
        [SerializeField] private bool maximize;

        // Momentum buffer
        [NonSerialized] public Tensor[] bGamma;
        [NonSerialized] public Tensor[] bBeta;

        public SGD(Learnable[] parameters, float lr, float momentum = 0f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false, bool maximize = false) : base(parameters, lr, weightDecay)
        {
            this.momentum = momentum;
            this.dampening = dampening;
            this.nesterov = nesterov;
            this.maximize = maximize;

            bGamma = new Tensor[parameters.Length];
            bBeta = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is Learnable P)
                {
                    bGamma[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    bBeta[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                }
            }
        }
        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                // Classic sgd (uses lr: 0.01f as default);
                // if (parameters[i] is Learnable P)
                // {
                //     m_W[i] = m_W[i] * momentum - P.gradGamma * learningRate;
                //     m_B[i] = m_B[i] * momentum - P.gradBeta * learningRate;
                // 
                //     P.gamma = P.gamma * (1f - weightDecay) + m_W[i];
                //     P.beta = P.beta + m_B[i];
                // }

                // pytorch implementation
                Learnable P = parameters[i];

                if (weightDecay != 0f)
                    P.gradGamma = P.gradGamma + weightDecay * P.gamma;

                if(momentum != 0f)
                {
                    if (t > 1)
                    {
                        bGamma[i] = bGamma[i] + (1f - dampening) * P.gradGamma;
                        bBeta[i] = bBeta[i] + (1f - dampening) * P.gradBeta;
                    }
                    else
                    {
                        bGamma[i] = P.gradGamma;
                        bBeta[i] = P.gradBeta;
                    }
                    if(nesterov)
                    {
                        P.gradGamma = P.gradGamma + momentum * bGamma[i];
                        P.gradBeta = P.gradBeta + momentum * bBeta[i];
                    }
                    else
                    {
                        P.gradGamma = Tensor.Identity(bGamma[i]);
                        P.gradBeta = Tensor.Identity(bBeta[i]);
                    }

                }
                if(maximize)
                {
                    P.gamma = P.gamma + learningRate * P.gradGamma;
                    P.beta = P.beta + learningRate * P.gradBeta;
                }
                else
                {
                    P.gamma = P.gamma - learningRate * P.gradGamma;
                    P.beta = P.beta - learningRate * P.gradBeta;
                }
            });
        }
    }

}