
using UnityEngine;
namespace DeepUnity
{
    // https://arxiv.org/pdf/2208.06677.pdf
    public sealed class Adan : Optimizer
    {
        [SerializeField] private readonly float beta1;
        [SerializeField] private readonly float beta2;
        [SerializeField] private readonly float beta3;

        private readonly Tensor[] mGamma;
        private readonly Tensor[] mBeta;

        private readonly Tensor[] vGamma;
        private readonly Tensor[] vBeta;

        private readonly Tensor[] nGamma;
        private readonly Tensor[] nBeta;

        private readonly Tensor[] oldGammaGrad;
        private readonly Tensor[] oldBetaGrad;

        // Default settings at page 14
        public Adan(Learnable[] parameters, float lr = 0.001f, float beta1 = 0.02f, float beta2 = 0.08f, float beta3 = 0.01f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {

            throw new System.NotImplementedException("Adan optimizer is not implemented yey");
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.beta3 = beta3;

            mGamma = new Tensor[parameters.Length];
            mBeta = new Tensor[parameters.Length];

            vGamma = new Tensor[parameters.Length];
            vBeta = new Tensor[parameters.Length];

            nGamma = new Tensor[parameters.Length];
            nBeta = new Tensor[parameters.Length];

            oldGammaGrad = new Tensor[parameters.Length];
            oldBetaGrad = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                Learnable P = parameters[i];

                oldGammaGrad[i] = Tensor.Zeros(P.gamma.Shape);
                oldBetaGrad[i] = Tensor.Zeros(P.beta.Shape);

                mGamma[i] = Tensor.Identity(oldGammaGrad[i]);
                mBeta[i] = Tensor.Identity(oldBetaGrad[i]);

                vGamma[i] = Tensor.Zeros(P.gamma.Shape);
                vBeta[i] = Tensor.Zeros(P.beta.Shape);

                nGamma[i] = Tensor.Identity(oldGammaGrad[i]).Pow(2f);
                nBeta[i] = Tensor.Identity(oldBetaGrad[i]).Pow(2f);

            }
        }

        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is Learnable P)
                {
                    mGamma[i] = (1 - beta1) * mGamma[i] + beta1 * P.gammaGrad;
                    mBeta[i] = (1 - beta1) * mBeta[i] + beta1 * P.betaGrad;

                    vGamma[i] = (1 - beta2) * vGamma[i] + beta2 * (P.gammaGrad - oldGammaGrad[i]);
                    vBeta[i] = (1 - beta2) * vBeta[i] + beta2 * (P.betaGrad - oldBetaGrad[i]);

                    nGamma[i] = (1 - beta3) * nGamma[i] + beta3 * (P.gammaGrad + (1 - beta2) * (P.gammaGrad - oldGammaGrad[i])).Pow(2f);
                    nBeta[i] = (1 - beta3) * nBeta[i] + beta3 * (P.betaGrad + (1 - beta2) * (P.betaGrad - oldBetaGrad[i])).Pow(2f);

                    Tensor eta = Tensor.Fill(lr, nGamma[i].Shape);
                    Tensor etaGamma = eta / (nGamma[i].Sqrt() + Utils.EPSILON);
                    Tensor etaBeta = eta / (nBeta[i].Sqrt() + Utils.EPSILON);

                    P.gamma = (1 + etaGamma * lambda).Pow(-1f) * (P.gamma - etaGamma * (mGamma[i] + (1f - beta2) * vGamma[i]));
                    P.beta = (1 + etaBeta * lambda).Pow(-1f) * (P.beta - etaBeta * (mBeta[i] + (1f - beta2) * vBeta[i]));

                    oldGammaGrad[i] = P.gammaGrad;
                    oldBetaGrad[i] = P.betaGrad;
                 }

                if (parameters[i] is ISelfOptimizable S)
                    S.SelfOptimise(lr * 10f);
            });

            

        }

    }
}




