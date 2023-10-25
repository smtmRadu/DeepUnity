using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
    [Serializable]
    public class Adadelta : Optimizer
    {
        [SerializeField] private float rho;


        // Square avg buffer
        [NonSerialized] public Tensor[] vGamma;
        [NonSerialized] public Tensor[] vBeta;


        // Accumulate var buffer
        [NonSerialized] public Tensor[] uGamma;
        [NonSerialized] public Tensor[] uBeta;



        public Adadelta(Learnable[] parameters, float lr = 1f, float rho = 0.9f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {
            this.rho = rho;

            vGamma = new Tensor[parameters.Length];
            vBeta = new Tensor[parameters.Length];

            uGamma = new Tensor[parameters.Length];
            uBeta = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                Learnable P = parameters[i];

                vGamma[i] = Tensor.Zeros(P.gamma.Shape);
                vBeta[i] = Tensor.Zeros(P.beta.Shape);

                uGamma[i] = Tensor.Zeros(P.gamma.Shape);
                uBeta[i] = Tensor.Zeros(P.beta.Shape);
                
            }
        }


        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is Learnable P)
                {
                    if (weightDecay != 0f)
                        P.gammaGrad = P.gammaGrad + weightDecay * P.gamma;

                    vGamma[i] = vGamma[i] * rho + Tensor.Pow(P.gammaGrad, 2f) * (1f - rho);
                    vBeta[i] = vBeta[i] * rho + Tensor.Pow(P.betaGrad, 2f) * (1f - rho);

                    // In Adadelta, i use v for square avg and u for accumulate variables
                    var dxGamma = Tensor.Sqrt(uGamma[i] + Utils.EPSILON) / Tensor.Sqrt(vGamma[i] + Utils.EPSILON) * P.gammaGrad;
                    var dxBeta = Tensor.Sqrt(uBeta[i] + Utils.EPSILON) / Tensor.Sqrt(vBeta[i] + Utils.EPSILON) * P.betaGrad;

                    uGamma[i] = uGamma[i] * rho + Tensor.Pow(dxGamma, 2f) * (1f - rho);
                    uBeta[i] = uBeta[i] * rho + Tensor.Pow(dxBeta, 2f) * (1f - rho);

                    P.gamma = P.gamma - learningRate * dxGamma;
                    P.beta = P.beta - learningRate * dxBeta;
                }
                if (parameters[i] is ISelfOptimizable S)
                    S.SelfOptimise(learningRate / 100f);
            });
        }
    }
}