using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
    [Serializable]
    public class Adagrad : Optimizer
    {
        [SerializeField] private float learningRateDecay;

        [NonSerialized] public Tensor[] stateSum_W;
        [NonSerialized] public Tensor[] stateSum_B;

        public Adagrad(Learnable[] parameters, float lr = 0.01f, float lrDecay = 0f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {
            this.learningRateDecay = lrDecay;

            stateSum_W = new Tensor[parameters.Length];
            stateSum_B = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                Learnable P = parameters[i];

                stateSum_W[i] = Tensor.Zeros(P.gamma.Shape);
                stateSum_B[i] = Tensor.Zeros(P.beta.Shape);         
            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is Learnable L)
                {
                    var gammaBar = learningRate / (1 + (t - 1) * learningRateDecay);

                    if (weightDecay != 0f)
                    {
                        L.gammaGrad = L.gammaGrad + weightDecay * L.gammaGrad;
                    }

                    stateSum_W[i] = stateSum_W[i] + Tensor.Pow(L.gammaGrad, 2f);
                    stateSum_B[i] = stateSum_B[i] + Tensor.Pow(L.betaGrad, 2f);

                    L.gamma = L.gamma - gammaBar * (L.gammaGrad / (Tensor.Sqrt(stateSum_W[i]) + Utils.EPSILON));
                    L.beta = L.beta - gammaBar * (L.betaGrad / (Tensor.Sqrt(stateSum_B[i]) + Utils.EPSILON));
                }

                if (parameters[i] is ISelfOptimizable S)
                    S.SelfOptimise(learningRate);
            });

        }
    }
}