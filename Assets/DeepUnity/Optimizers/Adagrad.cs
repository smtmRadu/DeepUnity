using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
    [Serializable]
    public class Adagrad : Optimizer
    {
        [SerializeField] private float learningRateDecay;

        [NonSerialized] public Tensor[] statesum_W;
        [NonSerialized] public Tensor[] statesum_B;

        public Adagrad(Learnable[] parameters, float lr = 0.01f, float lrDecay = 0f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {
            this.learningRateDecay = lrDecay;

            statesum_W = new Tensor[parameters.Length];
            statesum_B = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is Learnable P)
                {
                    statesum_W[i] = Tensor.Zeros(P.gamma.Shape);
                    statesum_B[i] = Tensor.Zeros(P.beta.Shape);

                }
            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is RNNCell R)
                {

                    throw new NotImplementedException("RNNCell optimization not implemented yet.");
                }
                else if (parameters[i] is Learnable L)
                {
                    var gammaBar = learningRate / (1 + (t - 1) * learningRateDecay);

                    if (weightDecay != 0f)
                    {
                        L.gammaGrad = L.gammaGrad + weightDecay * L.gammaGrad;
                    }

                    statesum_W[i] = statesum_W[i] + Tensor.Pow(L.gammaGrad, 2f);
                    statesum_B[i] = statesum_B[i] + Tensor.Pow(L.betaGrad, 2f);

                    L.gamma = L.gamma - gammaBar * (L.gammaGrad / (Tensor.Sqrt(statesum_W[i]) + 1e-10f));
                    L.beta = L.beta - gammaBar * (L.betaGrad / (Tensor.Sqrt(statesum_B[i]) + 1e-10f));
                }
            });

        }
    }
}