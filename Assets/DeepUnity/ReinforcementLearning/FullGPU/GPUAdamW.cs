// GPUAdamW — fully-GPU AdamW optimizer with optional global grad-norm clip.
//
//   Step():
//     1. (optional) compute ||g||² across all param buffers in one shared accumulator,
//        then derive `scale = min(1, clip/||g||)` on the GPU.
//     2. for each parameter buffer: dispatch fused AdamWStep kernel
//        (reads P, G, M, V; writes P, M, V; decoupled weight decay).
//
// All Adam state (M, V) is GPU-resident. No CPU touches between minibatches.

using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning.FullGPU
{
    public sealed class GPUAdamW : System.IDisposable
    {
        readonly ComputeShader cs;
        readonly List<GPUParam> parameters;
        readonly int kAdam, kZero, kGradNormReduce, kGradNormFinalize;

        public float lr, beta1, beta2, eps, weight_decay, clip_max;
        public float beta1_t = 1f, beta2_t = 1f;
        public int t = 0;

        // grad-norm scratch
        ComputeBuffer normSq;     // [1]
        ComputeBuffer scaleOut;   // [1]
        float[] tmp1 = new float[1];

        public GPUAdamW(IEnumerable<GPUParam> ps,
                        float lr = 3e-4f, float beta1 = 0.9f, float beta2 = 0.999f,
                        float eps = 1e-5f, float weight_decay = 0f, float max_grad_norm = -1f)
        {
            cs = Resources.Load<ComputeShader>("ComputeShaders/RLBoosterCS");
            parameters = new List<GPUParam>(ps);
            this.lr = lr;
            this.beta1 = beta1;
            this.beta2 = beta2;
            this.eps = eps;
            this.weight_decay = weight_decay;
            this.clip_max = max_grad_norm;

            kAdam = cs.FindKernel("AdamWStep");
            kZero = cs.FindKernel("ZeroBuffer");
            kGradNormReduce = cs.FindKernel("GradNormReduce");
            kGradNormFinalize = cs.FindKernel("GradNormFinalize");

            normSq   = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
            scaleOut = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
        }

        public void ZeroGrad()
        {
            // Zero G via the ZeroBuffer kernel (P binding -> the gradient buffer).
            foreach (var p in parameters)
            {
                cs.SetInt("N_", p.N);
                cs.SetBuffer(kZero, "P", p.G);
                cs.Dispatch(kZero, (p.N + 255) / 256, 1, 1);
            }
        }

        public void Step()
        {
            t++;
            beta1_t *= beta1;
            beta2_t *= beta2;

            float scale = 1f;
            if (clip_max > 0f)
            {
                // 1) zero normSq
                normSq.SetData(new float[] { 0f });
                // 2) accumulate ||g||² across all param buffers (one threadgroup per param)
                foreach (var p in parameters)
                {
                    cs.SetInt("N_", p.N);
                    cs.SetBuffer(kGradNormReduce, "G", p.G);
                    cs.SetBuffer(kGradNormReduce, "normSq", normSq);
                    cs.Dispatch(kGradNormReduce, 1, 1, 1);
                }
                // 3) finalize -> scaleOut[0]; readback (single float).
                cs.SetFloat("clip_max", clip_max);
                cs.SetBuffer(kGradNormFinalize, "normSq", normSq);
                cs.SetBuffer(kGradNormFinalize, "scaleOut", scaleOut);
                cs.Dispatch(kGradNormFinalize, 1, 1, 1);
                scaleOut.GetData(tmp1);
                scale = tmp1[0];
            }

            cs.SetFloat("lr", lr);
            cs.SetFloat("beta1", beta1);
            cs.SetFloat("beta2", beta2);
            cs.SetFloat("beta1_t", beta1_t);
            cs.SetFloat("beta2_t", beta2_t);
            cs.SetFloat("eps_adam", eps);
            cs.SetFloat("weight_decay", weight_decay);
            cs.SetFloat("scale", scale);

            foreach (var p in parameters)
            {
                cs.SetInt("N_", p.N);
                cs.SetBuffer(kAdam, "P", p.P);
                cs.SetBuffer(kAdam, "G", p.G);
                cs.SetBuffer(kAdam, "Mbuf", p.M);
                cs.SetBuffer(kAdam, "Vbuf", p.V);
                cs.Dispatch(kAdam, (p.N + 255) / 256, 1, 1);
            }
        }

        public void Dispose()
        {
            normSq?.Release();
            scaleOut?.Release();
        }
    }
}
