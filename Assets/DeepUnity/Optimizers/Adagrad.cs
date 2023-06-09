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
                    int inputs = d.param_W.Shape[1];
                    int outputs = d.param_W.Shape[0];

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
                        D.grad_W = D.grad_W + weightDecay * D.grad_W;
                    }

                    statesum_W[i] = statesum_W[i] + Tensor.Pow(D.grad_W, 2f);
                    statesum_B[i] = statesum_B[i] + Tensor.Pow(D.grad_B, 2f);

                    D.param_W = D.param_W - gammaBar * (D.grad_W / (Tensor.Sqrt(statesum_W[i]) + 1e-10f));
                    D.param_B = D.param_B - gammaBar * (D.grad_B / (Tensor.Sqrt(statesum_B[i]) + 1e-10f));
                }
            });

        }
    }
}

