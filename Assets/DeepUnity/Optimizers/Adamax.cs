using System;
using UnityEngine;

namespace DeepUnity
{ 
    // pytorch alg https://pytorch.org/docs/stable/generated/torch.optim.Adamax.html
    public class AdaMax : Optimizer
    {
        [SerializeField] private float beta1;
        [SerializeField] private float beta2;

        // 1st momentum buffer
        [NonSerialized] public Tensor[] mGamma;
        [NonSerialized] public Tensor[] mBeta;

        // exponentially weighted infinity norm
        [NonSerialized] public Tensor[] uGamma;
        [NonSerialized] public Tensor[] uBeta;


        public AdaMax(Learnable[] parameters, float lr = 0.002f, float beta1 = 0.9f, float beta2 = 0.999f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;

            mGamma = new Tensor[parameters.Length];
            mBeta = new Tensor[parameters.Length];

            uGamma = new Tensor[parameters.Length];
            uBeta = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is Learnable P)
                {
                    mGamma[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    mBeta[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                    uGamma[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    uBeta[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                }
            }
        }


        public override void Step()
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                // Paper implementation
                // if (parameters[i] is Learnable P)
                // {
                //     // Update biased first momentum estimate
                //     m_W[i] = beta1 * m_W[i] + (1f - beta1) * P.gradGamma;
                //     m_B[i] = beta1 * m_B[i] + (1f - beta1) * P.gradBeta;
                // 
                //     // Update the exponentially weighted infinity norm
                //     u_W[i] = Tensor.Max(beta2 * u_W[i], Tensor.Abs(P.gradGamma) + Utils.EPSILON);
                //     u_B[i] = Tensor.Max(beta2 * u_B[i], Tensor.Abs(P.gradBeta) + Utils.EPSILON);
                // 
                //     // Update parameters
                //     P.gamma = P.gamma * (1f - weightDecay) - (learningRate / (1f - MathF.Pow(beta1, t))) * m_W[i] / u_W[i];
                //     P.beta = P.beta - (learningRate / (1f - MathF.Pow(beta1, t))) * m_B[i] / u_B[i];
                // }

                Learnable P = parameters[i];

                // Weight decay is not applied on biases
                if (weightDecay != 0)                
                    P.gradGamma = P.gradGamma + weightDecay * P.gamma;

                mGamma[i] = beta1 * mGamma[i] + (1f - beta1) * P.gradGamma;
                mBeta[i] = beta1 * mBeta[i] + (1f - beta1) * P.gradBeta;

                uGamma[i] = Tensor.Maximum(beta2 * uGamma[i], Tensor.Abs(P.gradGamma) + Utils.EPSILON);
                uBeta[i] = Tensor.Maximum(beta2 * uBeta[i], Tensor.Abs(P.gradBeta) + Utils.EPSILON);

                P.gamma = P.gamma - learningRate * mGamma[i] / ((1f - MathF.Pow(beta1, t)) * uGamma[i]);
                P.beta = P.beta - learningRate * mBeta[i] / ((1f - MathF.Pow(beta1, t)) * uBeta[i]);

            });


        }
    }

}