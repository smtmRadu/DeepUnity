using System;
using UnityEngine;
namespace DeepUnity
{
    // https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html

    public sealed class SGD : Optimizer
    {
        [SerializeField] private readonly float momentum;
        [SerializeField] private readonly float dampening;
        [SerializeField] private readonly bool nesterov;
        [SerializeField] private readonly bool maximize;

        // Momentum buffer
        [NonSerialized] private readonly Tensor[] bGamma;
        [NonSerialized] private readonly Tensor[] bBeta;

        public SGD(Learnable[] parameters, float lr, float momentum = 0.9f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false, bool maximize = false) : base(parameters, lr, weightDecay)
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
                    bGamma[i] = Tensor.Zeros(P.gamma.Shape);
                    bBeta[i] = Tensor.Zeros(P.beta.Shape);

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
                //     bGamma[i] = bGamma[i] * momentum + P.gammaGrad * learningRate;
                //     bBeta[i] = bBeta[i] * momentum + P.betaGrad * learningRate;
                // 
                //     P.gamma = P.gamma * (1f - weightDecay) - bGamma[i];
                //     P.beta = P.beta - bBeta[i];
                // }

                // pytorch implementation
                if (parameters[i] is Learnable P)
                {
                    if (weightDecay != 0f)
                        P.gammaGrad = P.gammaGrad + weightDecay * P.gamma;

                    if (momentum != 0f)
                    {
                        if (t > 1)
                        {
                            bGamma[i] = momentum * bGamma[i] + (1f - dampening) * P.gammaGrad;
                            bBeta[i] = momentum * bBeta[i] + (1f - dampening) * P.betaGrad;
                        }
                        else
                        {
                            bGamma[i] = Tensor.Identity(P.gammaGrad);
                            bBeta[i] = Tensor.Identity(P.betaGrad);
                        }
                        if (nesterov)
                        {
                            P.gammaGrad = P.gammaGrad + momentum * bGamma[i];
                            P.betaGrad = P.betaGrad + momentum * bBeta[i];
                        }
                        else
                        {
                            P.gammaGrad = Tensor.Identity(bGamma[i]);
                            P.betaGrad = Tensor.Identity(bBeta[i]);
                        }

                    }
                    if (maximize)
                    {
                        P.gamma = P.gamma + learningRate * P.gammaGrad;
                        P.beta = P.beta + learningRate * P.betaGrad;
                    }
                    else
                    {
                        P.gamma = P.gamma - learningRate * P.gammaGrad;
                        P.beta = P.beta - learningRate * P.betaGrad;
                    }
                }

                if (parameters[i] is RNNCell R)
                {
                    R.recurrentGamma = -learningRate * R.recurrentGammaGrad;
                    R.recurrentBeta = -learningRate * R.recurrentBetaGrad;
                }
            });
        }
    }

}