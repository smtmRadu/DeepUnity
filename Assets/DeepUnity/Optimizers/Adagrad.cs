using System;
using UnityEngine;

namespace DeepUnity
{
    // https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html
    [Serializable]
    public class Adagrad : IOptimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float learningRate;
        [SerializeField] private float learningRateDecay;
        [SerializeField] private float weightDecay;

        [NonSerialized] public Tensor[] statesum_W;
        [NonSerialized] public Tensor[] statesum_B;

        public Adagrad(float learningRate = 0.01f, float learningRateDecay = 0f, float weightDecay = 0f)
        {
            this.t = 0;
            this.learningRate = learningRate;
            this.learningRateDecay = learningRateDecay;
            this.weightDecay = weightDecay;
        }
        public void Initialize(IModule[] modules)
        {
            statesum_W = new Tensor[modules.Length];
            statesum_B = new Tensor[modules.Length];

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense d)
                {
                    int inputs = d.weights.Shape[1];
                    int outputs = d.weights.Shape[0];

                    statesum_W[i] = Tensor.Zeros(outputs, inputs);
                    statesum_B[i] = Tensor.Zeros(outputs);

                }
            }
        }
        public void Step(IModule[] modules)
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, modules.Length, i =>
            {
                if (modules[i] is Dense D)
                {
                    var gammaBar = learningRate / (1 + (t - 1) * learningRateDecay);

                    if (weightDecay != 0f)
                    {
                        D.grad_Weights = D.grad_Weights + weightDecay * D.grad_Weights;
                    }

                    statesum_W[i] = statesum_W[i] + Tensor.Pow(D.grad_Weights, 2f);
                    statesum_B[i] = statesum_B[i] + Tensor.Pow(D.grad_Biases, 2f);

                    D.weights = D.weights - gammaBar * (D.grad_Weights / (Tensor.Sqrt(statesum_W[i]) + 1e-10f));
                    D.biases = D.biases - gammaBar * (D.grad_Biases / (Tensor.Sqrt(statesum_B[i]) + 1e-10f));
                }
            });

        }
    }
}

