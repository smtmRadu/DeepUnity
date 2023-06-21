using System;
using UnityEngine;

namespace DeepUnity
{
    // does not work idk why
    // https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html
    [Serializable]
    public sealed class RMSProp : Optimizer
    {
        [SerializeField] private float alpha; //smoothing constant
        [SerializeField] private float momentum;
        [SerializeField] private bool centered;
        [SerializeField] private float weightDecay;

        // square average 
        [NonSerialized] public Tensor[] v_W;
        [NonSerialized] public Tensor[] v_B;

        // buffer
        [NonSerialized] public Tensor[] b_W;
        [NonSerialized] public Tensor[] b_B;

        // average gradient
        [NonSerialized] public Tensor[] gAve_W;
        [NonSerialized] public Tensor[] gAve_B;


        public RMSProp(Learnable[] parameters, float lr = 0.01f, float alpha = 0.99f, float momentum = 0.9f, float weightDecay = 0f, bool centered = false)
        {
            this.learningRate = lr;
            this.alpha = alpha;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
            this.centered = centered;




            this.parameters = parameters;

            b_W = new Tensor[parameters.Length];
            b_B = new Tensor[parameters.Length];

            v_W = new Tensor[parameters.Length];
            v_B = new Tensor[parameters.Length];

            if (centered)
            {
                gAve_W = new Tensor[parameters.Length];
                gAve_B = new Tensor[parameters.Length];
            }

            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i] is Learnable P)
                {
                    b_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    b_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                    v_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                    v_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());

                    if (centered)
                    {
                        gAve_W[i] = Tensor.Zeros(P.gamma.Shape.ToArray());
                        gAve_B[i] = Tensor.Zeros(P.beta.Shape.ToArray());
                    }
                }
            }
        }

        public override void Step()
        {
            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                if (parameters[i] is Learnable P)
                {
                    if (weightDecay != 0)
                        P.gradGamma += P.gamma * weightDecay;


                    v_W[i] = alpha * v_W[i] + (1f - alpha) * Tensor.Pow(P.gradGamma, 2);
                    v_B[i] = alpha * v_B[i] + (1f - alpha) * Tensor.Pow(P.gradBeta, 2);

                    var vBar_W = Tensor.Identity(v_W[i]);
                    var vBar_B = Tensor.Identity(v_B[i]);

                    if (centered)
                    {
                        gAve_W[i] = gAve_W[i] * alpha + (1f - alpha) * P.gradGamma;
                        gAve_B[i] = gAve_B[i] * alpha + (1f - alpha) * P.gradBeta;

                        vBar_W = vBar_W - Tensor.Pow(gAve_W[i], 2f);
                        vBar_B = vBar_B - Tensor.Pow(gAve_B[i], 2f);
                    }

                    if (momentum > 0f)
                    {
                        b_W[i] = momentum * b_W[i] + P.gradGamma / (Tensor.Sqrt(vBar_W) + Utils.EPSILON);
                        b_B[i] = momentum * b_B[i] + P.gradBeta / (Tensor.Sqrt(vBar_B) + Utils.EPSILON);

                        P.gamma = P.gamma - learningRate * b_W[i];
                        P.beta = P.beta - learningRate * b_B[i];
                    }
                    else
                    {
                        P.gamma = P.gamma - learningRate * P.gradGamma / (Tensor.Sqrt(vBar_W) + Utils.EPSILON);
                        P.beta = P.beta - learningRate * P.gradBeta / (Tensor.Sqrt(vBar_B) + Utils.EPSILON);
                    }
                }
            });
        }
    }

}