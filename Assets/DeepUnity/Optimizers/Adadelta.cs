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
        [NonSerialized] public Tensor[] v_W;
        [NonSerialized] public Tensor[] v_B;


        // Accumulate var buffer
        [NonSerialized] public Tensor[] u_W;
        [NonSerialized] public Tensor[] u_B;



        public Adadelta(Learnable[] parameters, float lr = 1f, float rho = 0.9f, float weightDecay = 0f) : base(parameters, lr, weightDecay)
        {
            this.rho = rho;

            v_W = new Tensor[parameters.Length];
            v_B = new Tensor[parameters.Length];

            u_W = new Tensor[parameters.Length];
            u_B = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is Learnable P)
                {
                    v_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    v_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                    u_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    u_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());
                }
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
                        P.gradGamma = P.gradGamma + weightDecay * P.gamma;

                    v_W[i] = v_W[i] * rho + Tensor.Pow(P.gradGamma, 2f) * (1f - rho);
                    v_B[i] = v_B[i] * rho + Tensor.Pow(P.gradBeta, 2f) * (1f - rho);

                    // In Adadelta, i use v for square avg and m for accumulate variables
                    var dxWeights = Tensor.Sqrt(u_W[i] + Utils.EPSILON) / Tensor.Sqrt(v_W[i] + Utils.EPSILON) * P.gradGamma;
                    var dxBiases = Tensor.Sqrt(u_B[i] + Utils.EPSILON) / Tensor.Sqrt(v_B[i] + Utils.EPSILON) * P.gradBeta;

                    u_W[i] = u_W[i] * rho + Tensor.Pow(dxWeights, 2f) * (1f - rho);
                    u_B[i] = u_B[i] * rho + Tensor.Pow(dxBiases, 2f) * (1f - rho);

                    P.gamma = P.gamma - learningRate * dxWeights;
                    P.beta = P.beta - learningRate * dxBiases;
                }

            });
        }
    }
}