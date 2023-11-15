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
        private readonly Tensor[] m;

        // 2nd momentum buffer 
        private readonly Tensor[] v;



        public Adam(Tensor[] parameters, Tensor[] gradients, float lr = 0.001f, float beta1 = 0.9f, float beta2 = 0.999f) : base(parameters, gradients, lr, 0f)
        {
            this.beta1 = beta1;
            this.beta2 = beta2;

            m = new Tensor[parameters.Length];
            v = new Tensor[parameters.Length];

            for (int i = 0; i < parameters.Length; i++)
            {
                m[i] = Tensor.Zeros(parameters[i].Shape);
                v[i] = Tensor.Zeros(parameters[i].Shape);
            }           
        }

        public override void Step()
        {
            t++;

            beta1_t *= beta1;
            beta2_t *= beta2;

            System.Threading.Tasks.Parallel.For(0, parameters.Length, i =>
            {
                // Update biased first momentum estimate
                m[i] = beta1 * m[i] + (1f - beta1) * gradients[i];

                // Update biased second raw momentum estimate
                v[i] = beta2 * v[i] + (1f - beta2) * gradients[i].Pow(2f);

                // Compute bias-corrected first momentum estimate
                Tensor mHat = m[i] / (1f - beta1_t);

                // Compute bias-corrected second raw momentum estimate
                Tensor vHat = v[i] / (1f - beta2_t);

                // Update parameters 
                parameters[i].AssignAs(parameters[i] - lr * mHat / (Tensor.Sqrt(vHat) + Utils.EPSILON));
            });

        }
    }

}