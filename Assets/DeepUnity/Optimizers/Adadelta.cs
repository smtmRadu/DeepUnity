using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adadelta.html
    [System.Serializable]
    public class Adadelta : IOptimizer
    {
        [SerializeField] private float learningRate;
        [SerializeField] private float rho;
        [SerializeField] private float weightDecay;


        // Square avg buffer
        [NonSerialized] public Tensor[] v_W;
        [NonSerialized] public Tensor[] v_B;


        // Accumulate var buffer
        [NonSerialized] public Tensor[] u_W;
        [NonSerialized] public Tensor[] u_B;

       

        public Adadelta(float learningRate = 1.0f, float rho = 0.9f, float weightDecay = 0f)
        {
            this.learningRate = learningRate;
            this.weightDecay = weightDecay;
            this.rho = rho;
        }

        public void Initialize(IModule[] modules)
        {
            v_W = new Tensor[modules.Length];
            v_B = new Tensor[modules.Length];

            u_W = new Tensor[modules.Length];
            u_B = new Tensor[modules.Length];

          

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense d)
                {
                    int inputs = d.param_W.Shape[1];
                    int outputs = d.param_W.Shape[0];

                    v_W[i] = Tensor.Zeros(outputs, inputs);
                    v_B[i] = Tensor.Zeros(outputs);

                    u_W[i] = Tensor.Zeros(outputs, inputs);
                    u_B[i] = Tensor.Zeros(outputs);
                }
            }
        }

        public void Step(IModule[] modules)
        {
            System.Threading.Tasks.Parallel.For(0, modules.Length, i =>
            {
                if (modules[i] is Dense D)
                {
                    if (weightDecay != 0f)
                        D.grad_W = D.grad_W + weightDecay * D.param_W;

                    v_W[i] = v_W[i] * rho + Tensor.Pow(D.grad_W, 2f) * (1f - rho);
                    v_B[i] = v_B[i] * rho + Tensor.Pow(D.grad_B, 2f) * (1f - rho);

                    // In Adadelta, i use v for square avg and m for accumulate variables
                    var dxWeights = Tensor.Sqrt(u_W[i] + Utils.EPSILON) / Tensor.Sqrt(v_W[i] + Utils.EPSILON) * D.grad_W;
                    var dxBiases = Tensor.Sqrt(u_B[i] + Utils.EPSILON) / Tensor.Sqrt(v_B[i] + Utils.EPSILON) * D.grad_B;

                    u_W[i] = u_W[i] * rho + Tensor.Pow(dxWeights, 2f) * (1f - rho);
                    u_B[i] = u_B[i] * rho + Tensor.Pow(dxBiases, 2f) * (1f - rho);

                    D.param_W = D.param_W - learningRate * dxWeights;
                    D.param_B = D.param_B - learningRate * dxBiases;
                }
              
            });
        }
    }
}

