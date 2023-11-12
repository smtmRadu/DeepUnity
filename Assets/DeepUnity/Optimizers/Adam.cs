using System;
using UnityEngine;

namespace DeepUnity
{
    // Took right from the paper
    // https://arxiv.org/pdf/1412.6980.pdf
    public sealed class Adam : Optimizer
    {
        [SerializeField] private readonly float beta1;
        [SerializeField] private readonly float beta2;

        [SerializeField] private float beta1_t = 1f; // beta1^t caching
        [SerializeField] private float beta2_t = 1f;

        // 1st momentum buffer
        private readonly Tensor[] mGamma;
        private readonly Tensor[] mBeta;

        // 2nd momentum buffer 
        private readonly Tensor[] vGamma;
        private readonly Tensor[] vBeta;



        public Adam(Learnable[] parameters, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f) :base(parameters, lr, weightDecay)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;

            mGamma = new Tensor[parameters.Length];
            mBeta = new Tensor[parameters.Length];

            vGamma = new Tensor[parameters.Length];
            vBeta = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                Learnable P = parameters[i];

                mGamma[i] = Tensor.Zeros(P.gamma.Shape);
                mBeta[i] = Tensor.Zeros(P.beta.Shape);

                vGamma[i] = Tensor.Zeros(P.gamma.Shape);
                vBeta[i] = Tensor.Zeros(P.beta.Shape);

            }
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is Learnable P)
                {
                    Tensor mHat;
                    Tensor vHat;

                    // Update biased first momentum estimate
                    mGamma[i] = beta1 * mGamma[i] + (1f - beta1) * P.gammaGrad;

                    // Update biased second raw momentum estimate
                    vGamma[i] = beta2 * vGamma[i] + (1f - beta2) * Tensor.Pow(P.gammaGrad, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = mGamma[i] / (1f - beta1_t);

                    // Compute bias-corrected second raw momentum estimate
                    vHat = vGamma[i] / (1f - beta2_t);

                    // Update parameters
                    P.gamma = P.gamma * (1f - lambda) - lr * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);


                    // Update biased first momentum estimate
                    mBeta[i] = beta1 * mBeta[i] + (1f - beta1) * P.betaGrad;

                    // Update biased second raw momentum estimate
                    vBeta[i] = beta2 * vBeta[i] + (1f - beta2) * Tensor.Pow(P.betaGrad, 2f);

                    // Compute bias-corrected first momentum estimate
                    mHat = mBeta[i] / (1f - beta1_t);

                    // Compute bias-corrected second raw momentum estimate
                    vHat = vBeta[i] / (1f - beta2_t);

                    // Update parameters 
                    P.beta = P.beta - lr * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);
                }

                if (parameters[i] is ISelfOptimizable S)
                    S.SelfOptimise(lr * 10f);
            });

        }
    }

}