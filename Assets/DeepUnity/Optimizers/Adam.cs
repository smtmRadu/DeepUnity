using System;
using UnityEngine;

namespace DeepUnity
{
    // This one is took right from the paper

    public sealed class Adam : Optimizer
    {
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;

        // 1st momentum buffer
        private Tensor[] mGamma;
        private Tensor[] mBeta;

        // 2nd momentum buffer 
        private Tensor[] vGamma;
        private Tensor[] vBeta;


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


                mGamma[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                mBeta[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                vGamma[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                vBeta[i] = Tensor.Zeros(P.beta.Shape.ToArray());

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
                mGamma[i] = beta1 * mGamma[i] + (1f - beta1) * P.gradGamma;

                // Update biased second raw momentum estimate
                vGamma[i] = beta2 * vGamma[i] + (1f - beta2) * Tensor.Pow(P.gradGamma, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = mGamma[i] / (1f - MathF.Pow(beta1, t));

                // Compute bias-corrected second raw momentum estimate
                vHat = vGamma[i] / (1f - MathF.Pow(beta2, t));

                // Update parameters
                P.gamma = P.gamma * (1f - weightDecay) - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);




                // Update biased first momentum estimate
                mBeta[i] = beta1 * mBeta[i] + (1f - beta1) * P.gradBeta;

                // Update biased second raw momentum estimate
                vBeta[i] = beta2 * vBeta[i] + (1f - beta2) * Tensor.Pow(P.gradBeta, 2f);

                // Compute bias-corrected first momentum estimate
                mHat = mBeta[i] / (1f - MathF.Pow(beta1, t));

                // Compute bias-corrected second raw momentum estimate
                vHat = vBeta[i] / (1f - MathF.Pow(beta2, t));

                // Update parameters 
                P.beta = P.beta - learningRate * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON);
                
            });

        }
    }

}