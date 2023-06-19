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


        public RMSProp(float lr = 0.01f, float alpha = 0.99f, float momentum = 0.9f, float weightDecay = 0f, bool centered = false)
        {
            this.learningRate = lr;
            this.alpha = alpha;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
            this.centered = centered;
        }

        public override void Initialize(IModule[] modules)
        {
            b_W = new Tensor[modules.Length];
            b_B = new Tensor[modules.Length];

            v_W = new Tensor[modules.Length];
            v_B = new Tensor[modules.Length];

            if (centered)
            {
                gAve_W = new Tensor[modules.Length];
                gAve_B = new Tensor[modules.Length];
            }

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense D)
                {
                    int inputs = D.weights.Shape.height;
                    int outputs = D.weights.Shape.width;

                    b_W[i] = Tensor.Zeros(inputs, outputs);
                    b_B[i] = Tensor.Zeros(outputs);

                    v_W[i] = Tensor.Zeros(inputs, outputs);
                    v_B[i] = Tensor.Zeros(outputs);

                    if (centered)
                    {
                        gAve_W[i] = Tensor.Zeros(outputs, inputs);
                        gAve_B[i] = Tensor.Zeros(outputs);
                    }
                }
            }
        }

        public override void Step(IModule[] modules)
        {
            System.Threading.Tasks.Parallel.For(0, modules.Length, i =>
            {
                if (modules[i] is Dense D)
                {
                    if (weightDecay != 0)
                        D.grad_Weights += D.weights * weightDecay;


                    v_W[i] = alpha * v_W[i] + (1f - alpha) * Tensor.Pow(D.grad_Weights, 2);
                    v_B[i] = alpha * v_B[i] + (1f - alpha) * Tensor.Pow(D.grad_Biases, 2);

                    var vBar_W = Tensor.Identity(v_W[i]);
                    var vBar_B = Tensor.Identity(v_B[i]);

                    if (centered)
                    {
                        gAve_W[i] = gAve_W[i] * alpha + (1f - alpha) * D.grad_Weights;
                        gAve_B[i] = gAve_B[i] * alpha + (1f - alpha) * D.grad_Biases;

                        vBar_W = vBar_W - Tensor.Pow(gAve_W[i], 2f);
                        vBar_B = vBar_B - Tensor.Pow(gAve_B[i], 2f);
                    }

                    if (momentum > 0f)
                    {
                        b_W[i] = momentum * b_W[i] + D.grad_Weights / (Tensor.Sqrt(vBar_W) + Utils.EPSILON);
                        b_B[i] = momentum * b_B[i] + D.grad_Biases / (Tensor.Sqrt(vBar_B) + Utils.EPSILON);

                        D.weights = D.weights - learningRate * b_W[i];
                        D.biases = D.biases - learningRate * b_B[i];
                    }
                    else
                    {
                        D.weights = D.weights - learningRate * D.grad_Weights / (Tensor.Sqrt(vBar_W) + Utils.EPSILON);
                        D.biases = D.biases - learningRate * D.grad_Biases / (Tensor.Sqrt(vBar_B) + Utils.EPSILON);
                    }
                }
            });
        }
    }

}