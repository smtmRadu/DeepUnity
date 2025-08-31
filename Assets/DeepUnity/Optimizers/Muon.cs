using DeepUnity.Modules;
using System;
using UnityEngine;
namespace DeepUnity.Optimizers
{
    /// <summary>
    /// Muon optimizer. Note that this optimizer should not be used in the following layers:
    /// 1. Embedding layer
    /// 2. Final output fully connected layer
    /// 3. Any {0,1}-D variables.
    /// These should all be optimized using AdamW.
    /// For any parameter that isn't 2D, <see cref="AdamW"> optimizer is used.
    /// </summary>
    public sealed class Muon : Optimizer
    {
        
        /// MUON PARAMS
        [SerializeField] private int ns_steps=5;
        [SerializeField] private float a;
        [SerializeField] private float b;
        [SerializeField] private float c;
        [SerializeField] private float mu;
        [SerializeField] private Tensor[] B;

        /// ADAMW PARAMS
        [SerializeField]private readonly float adam_beta1;
        [SerializeField]private readonly float adam_beta2;
        [SerializeField]private readonly float adam_lr_ratio;
        [SerializeField]private float beta1_t = 1f; // beta1^t caching
        [SerializeField] private float beta2_t = 1f;

        [SerializeField]private readonly Tensor[] m;
        [SerializeField] private readonly Tensor[] v;


        /// <summary>
        /// Muon optimizer.
        /// Note that this optimizer should not be used in the following layers:
        /// 1. Embedding layer
        /// 2. Final output fully connected layer
        /// 3. Any {0,1}-D variables.
        /// These should all be optimized using AdamW.
        /// 
        /// For any parameter that isn't 2D, <see cref="AdamW"> optimizer is used.
        /// </summary>
        public Muon(
            Parameter[] parameters,
            float lr = 0.001f,
            float momentum =0.95f, 
            float adam_beta_1 = 0.9f,
            float adam_beta_2 = 0.999f,
            float weight_decay = 0.1f,
            float eps = 1e-7F,
            float adam_lr_ratio = 0.1f,
            float a = 3.4445F,
            float b = -4.775F,
            float c = 2.0315F,
            int newton_schultz_steps=5 // on keras default is 6.
            )
            : base(parameters, lr, eps, weight_decay, false)
        {
            this.adam_beta1 = adam_beta_1;
            this.adam_beta2 = adam_beta_2;
            this.adam_lr_ratio =adam_lr_ratio;
            this.mu = momentum;
            this.a = a;
            this.b = b;
            this.c = c;
            this.ns_steps = newton_schultz_steps;

            this.B = new Tensor[parameters.Length];
            this.m = new Tensor[parameters.Length];
            this.v = new Tensor[parameters.Length];
            for (int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i].g.Rank == 2)
                    B[i] = Tensor.Zeros(parameters[i].param.Shape);
                else
                {
                    m[i] = Tensor.Zeros(parameters[i].param.Shape);
                    v[i] = Tensor.Zeros(parameters[i].param.Shape);
                }
            }
        }
        public static Tensor NewtonSchultz(Tensor G, int steps=5, float a = 3.4445f, float b = -4.775f, float c = 2.0315f, float eps=1e-7F)
        {
            if (G.Rank != 2)
                throw new ArgumentException("G must be a 2D tensor to perform Newton-Schultz operation.");

            Tensor X = G / (G.Norm()[0] + eps);
            if (G.Size(0) > G.Size(1))
               X = X.T();

            for(int i = 0; i < steps; i++)
            {
                Tensor A = Tensor.MatMul(X, X.T(), device:Device.GPU);
                Tensor B = b * A + c * Tensor.MatMul(A, A, device: Device.GPU);
                X = a * X + Tensor.MatMul(B, X, device: Device.GPU);
            }
            if (G.Size(0) > G.Size(1))
                X = X.T();
            return X;
        }
        public override void Step()
        {
            t++;

            beta1_t *= adam_beta1;
            beta2_t *= adam_beta2;

            // non parallel because newton schultz runs on gpu
            for(int i = 0; i < parameters.Length; i++)
            {
                if (parameters[i].g.Rank != 2)
                {
                    Tensor.FusedAdamW(
                        param: parameters[i].param,
                        g: parameters[i].g,
                        m: m[i],
                        v: v[i],
                        vMax: null,
                        gamma: gamma * this.adam_lr_ratio,
                        betas: (adam_beta1, adam_beta2),
                        betas_t: (beta1_t, beta2_t),
                        lambda: lambda,
                        eps: epsilon,
                        maximize: false,
                        amsgrad: false);
                    return;
                }
                else  // Muon optimization
                {
                    Tensor.CopyTo(this.mu * B[i] + parameters[i].g, B[i]);
                    var O_t = NewtonSchultz(G: B[i], steps: this.ns_steps, a: this.a, b: this.b, c: this.c, eps: this.epsilon);
                    Tensor.CopyTo(fromTensor:parameters[i].param - gamma * O_t, toTensor:parameters[i].param);
                }
            }
        }
    }
}