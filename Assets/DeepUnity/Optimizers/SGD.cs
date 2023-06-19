using System;
using UnityEngine;
namespace DeepUnity
{
    // https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/experimental/SGD
    // https://pytorch.org/docs/stable/generated/torch.optim.SGD.html
    [Serializable]
    public sealed class SGD : Optimizer
    {
        [SerializeField] private int t;
        [SerializeField] private float momentum;
        [SerializeField] private float weightDecay;
        [SerializeField] private float dampening;
        [SerializeField] private bool nesterov;
        [SerializeField] private bool maximize;

        // Momentum buffer
        [NonSerialized] public Tensor[] m_W;
        [NonSerialized] public Tensor[] m_B;

        public SGD(float lr = 0.01f, float momentum = 0.9f, float weightDecay = 0f, float dampening = 0f, bool nesterov = false, bool maximize = false)
        {
            this.t = 0;
            this.learningRate = lr;
            this.momentum = momentum;
            this.weightDecay = weightDecay;
            this.dampening = dampening;
            this.nesterov = nesterov;
            this.maximize = maximize;
        }

        public override void Initialize(IModule[] modules)
        {
            m_W = new Tensor[modules.Length];
            m_B = new Tensor[modules.Length];

            for (int i = 0; i < modules.Length; i++)
            {
                if (modules[i] is Dense d)
                {
                    int inputs = d.weights.Shape.height;
                    int outputs = d.weights.Shape.width;

                    m_W[i] = Tensor.Zeros(inputs, outputs);
                    m_B[i] = Tensor.Zeros(outputs);

                }
            }
        }
        public override void Step(IModule[] modules)
        {
            t++;

            System.Threading.Tasks.Parallel.For(0, modules.Length, i =>
            {
                if (modules[i] is Dense D)
                {
                    m_W[i] = m_W[i] * momentum - D.grad_Weights * learningRate;
                    m_B[i] = m_B[i] * momentum - D.grad_Biases * learningRate;

                    D.weights = D.weights * (1f - weightDecay) + m_W[i];
                    D.biases = D.biases + m_B[i];
                }

            });

            // // Tensorflow algorithm (+ L2 penalty added)
            // if(momentum == 0)
            // {
            //     L.t_W = L.t_W - learningRate * L.g_W;
            //     L.t_B = L.t_B - learningRate * L.g_B;
            // }
            // else
            // {
            //     L.v_W = momentum * L.v_W + learningRate * L.g_W;
            //     L.v_B = momentum * L.v_B + learningRate * L.g_B;
            // 
            //     if (!nesterov)
            //     {
            //        
            //         L.t_W = L.t_W * (1f - weightDecay) + L.v_W;
            //         L.t_B = L.t_B + L.v_B;
            //     }
            //     else
            //     {
            //         L.t_W = L.t_W * (1f - weightDecay) + momentum * L.v_W - learningRate * L.g_W;
            //         L.t_B = L.t_B + momentum * L.v_B - learningRate * L.g_B;
            //     }
            // }

            // Pytorch algorithm
            // if(weightDecay != 0)
            //     L.g_W = L.g_W + weightDecay * L.t_W;
            // 
            // if(momentum != 0)
            // {
            //     if(t > 1)
            //     {
            //         L.m_W = momentum * L.m_W + (1f - dampening) * L.g_W;
            //         L.m_B = momentum * L.m_B + (1f - dampening) * L.g_B;
            //     }    
            //     else
            //     {
            //         L.m_W = L.g_W.Clone() as Tensor;
            //         L.m_B = L.g_B.Clone() as Tensor;
            //     }
            // 
            //     if(nesterov)
            //     {
            //         L.g_W = L.g_W + momentum * L.m_W;
            //         L.g_B = L.g_B + momentum * L.m_B;
            //     }
            //     else
            //     {
            //         L.g_W = L.m_B.Clone() as Tensor;
            //         L.g_B = L.m_B.Clone() as Tensor;
            //     }
            // }
            // 
            // if(maximize)
            // {
            //     L.t_W = L.t_W + (1f - learningRate) * L.g_W;
            //     L.t_B = L.t_B + (1f - learningRate) * L.g_B;
            // }
            // else
            // {
            //     L.t_W = L.t_W - (1f - learningRate) * L.g_W;
            //     L.t_B = L.t_B - (1f - learningRate) * L.g_B;
            // }

        }
    }

}