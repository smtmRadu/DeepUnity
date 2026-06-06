// GPUMLP — persistent-buffer MLP / LnMLP wrapper.
//
// Walks a Sequential's IModule[] and:
//   - groups modules into stages (Dense, RMSNorm, Activation, Softplus, Softmax)
//   - allocates persistent ComputeBuffers for params, grads, momentum, scratch
//   - exposes Forward(input_buf, batch) -> output_buf, Backward(dY_buf, batch) -> dX_buf
//   - supports UploadFromCpu / DownloadToCpu for save/load round-trip with the
//     existing Sequential format (so PPOGPU can read/write the same .asset/.json).
//
// Backed by RLBoosterCS.compute. Activations & gradients are FP32. Weights are
// FP32 on GPU as well (tested for clarity; FP16 weights can be retrofitted).
//
// Supported module types (anything else aborts with a clear error):
//   DeepUnity.Modules.Dense
//   DeepUnity.Modules.RMSNorm                    (with elementwise_affine = true)
//   DeepUnity.Activations.ReLU / Tanh / SiLU     (mapped to fwd_act/bwd_act kinds)
//   DeepUnity.Activations.Softplus               (sigma-network tail)
//   DeepUnity.Activations.Softmax                (discrete-network tail; replaced by an
//                                                 online-safe-softmax kernel call)

using System;
using System.Collections.Generic;
using DeepUnity.Activations;
using DeepUnity.Models;
using DeepUnity.Modules;
using UnityEngine;

namespace DeepUnity.ReinforcementLearning.FullGPU
{
    public enum GPUActKind { Linear = -1, ReLU = 0, Tanh = 1, SiLU = 2, GELU = 3 }

    /// <summary>A single GPU parameter (param + grad + Adam state) — what GPUAdamW iterates over.</summary>
    public sealed class GPUParam : IDisposable
    {
        public ComputeBuffer P, G, M, V;
        public int N;
        public string name;
        public GPUParam(int n, string name)
        {
            N = n;
            this.name = name;
            P = new ComputeBuffer(n, 4, ComputeBufferType.Structured);
            G = new ComputeBuffer(n, 4, ComputeBufferType.Structured);
            M = new ComputeBuffer(n, 4, ComputeBufferType.Structured);
            V = new ComputeBuffer(n, 4, ComputeBufferType.Structured);
            // Zero-init G/M/V (P will be filled by upload).
            float[] zeros = new float[n];
            G.SetData(zeros); M.SetData(zeros); V.SetData(zeros);
        }
        public void Dispose() { P?.Release(); G?.Release(); M?.Release(); V?.Release(); }
    }

    public sealed class GPUMLP : IDisposable
    {
        // ---------- public ----------
        public readonly int InputDim;
        public readonly int OutputDim;
        public readonly int MaxBatch;
        /// <summary>Output buffer of the network's last stage. Sized [MaxBatch, OutputDim].</summary>
        public ComputeBuffer Output { get; private set; }
        public bool RequiresGrad { get; set; } = true;

        // ---------- internals ----------
        readonly ComputeShader cs;
        readonly Sequential cpuMirror;
        readonly List<Stage> stages = new();
        readonly List<ComputeBuffer> tape = new();   // tape[i] = input to stage i; tape[N] = network output
        readonly List<ComputeBuffer> dtape = new();  // dtape[i] = grad wrt input to stage i

        // kernels
        int kDenseFwd, kDenseBwdIn, kDenseBwdW, kDenseBwdB, kApplyActGrad;
        int kRMSFwd, kRMSBwd, kRMSBwdGamma;
        int kSoftplusFwd, kSoftplusBwd, kActFwdInplace;
        int kCopyBuffer;

        // Distinct dummy for unused buffer slots. NEVER dummy-bind an already-bound buffer:
        // D3D11 nulls the earlier UAV slot when the same resource is bound to two slots.
        ComputeBuffer dummyBuf;

        public GPUMLP(Sequential cpuModel, int maxBatch)
        {
            cpuMirror = cpuModel;
            MaxBatch = maxBatch;
            cs = Resources.Load<ComputeShader>("ComputeShaders/RLBoosterCS");
            if (cs == null) throw new Exception("RLBoosterCS.compute not found in Resources/ComputeShaders.");
            CacheKernelIds();
            dummyBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);

            int curDim = -1;
            for (int i = 0; i < cpuModel.Modules.Length; i++)
            {
                var m = cpuModel.Modules[i];
                Stage st = BuildStage(m, ref curDim);
                stages.Add(st);
            }
            if (curDim < 0) throw new Exception("GPUMLP: empty model.");
            OutputDim = curDim;
            InputDim = stages[0].InDim;

            // tape & dtape: size determined by walking stages
            int dim = InputDim;
            tape.Add(new ComputeBuffer(MaxBatch * dim, 4, ComputeBufferType.Structured));
            dtape.Add(new ComputeBuffer(MaxBatch * dim, 4, ComputeBufferType.Structured));
            foreach (var st in stages)
            {
                dim = st.OutDim;
                tape.Add(new ComputeBuffer(MaxBatch * dim, 4, ComputeBufferType.Structured));
                dtape.Add(new ComputeBuffer(MaxBatch * dim, 4, ComputeBufferType.Structured));
            }
            Output = tape[tape.Count - 1];

            UploadFromCpu();
        }

        void CacheKernelIds()
        {
            kDenseFwd = cs.FindKernel("DenseFwd");
            kDenseBwdIn = cs.FindKernel("DenseBwdIn");
            kDenseBwdW = cs.FindKernel("DenseBwdW");
            kDenseBwdB = cs.FindKernel("DenseBwdB");
            kApplyActGrad = cs.FindKernel("ApplyActGrad");
            kRMSFwd = cs.FindKernel("RMSFwd");
            kRMSBwd = cs.FindKernel("RMSBwd");
            kRMSBwdGamma = cs.FindKernel("RMSBwdGamma");
            kSoftplusFwd = cs.FindKernel("SoftplusFwd");
            kSoftplusBwd = cs.FindKernel("SoftplusBwd");
            kActFwdInplace = cs.FindKernel("ActFwdInplace");
            kCopyBuffer = cs.FindKernel("CopyBuffer");
        }

        // ---------- stage construction ----------
        Stage BuildStage(IModule m, ref int curDim)
        {
            switch (m)
            {
                case Dense d:
                {
                    int inF = d.weights.Size(-1);
                    int outF = d.weights.Size(-2);
                    if (curDim < 0) curDim = inF;
                    if (curDim != inF)
                        throw new Exception($"GPUMLP: dim mismatch entering Dense: got {curDim}, expected {inF}.");
                    var st = new DenseStage(this, inF, outF, d);
                    curDim = outF;
                    return st;
                }
                case RMSNorm r:
                {
                    if (curDim < 0) throw new Exception("GPUMLP: RMSNorm cannot be the first module.");
                    int F = r.gamma.Size(0);
                    if (F != curDim) throw new Exception($"GPUMLP: RMSNorm feat={F} != curDim={curDim}");
                    return new RMSNormStage(this, F, r);
                }
                case ReLU _: return new ActivationStage(this, curDim, GPUActKind.ReLU);
                case Tanh _: return new ActivationStage(this, curDim, GPUActKind.Tanh);
                case SiLU _: return new ActivationStage(this, curDim, GPUActKind.SiLU);
                case GELU _: return new ActivationStage(this, curDim, GPUActKind.GELU);
                case Softplus _: return new SoftplusStage(this, curDim);
                case Softmax _: return new SoftmaxTailStage(this, curDim); // recognized but executed by PPOLossCS
                default:
                    throw new Exception($"GPUMLP: unsupported module type '{m.GetType().Name}'. " +
                                        "FullGPU only supports MLP/LnMLP networks (Dense/RMSNorm/ReLU/Tanh/SiLU/GELU/Softplus/Softmax).");
            }
        }

        // ---------- public API ----------
        /// <summary>Upload current CPU weights (Sequential.Modules) to GPU buffers.</summary>
        public void UploadFromCpu()
        {
            foreach (var st in stages) st.UploadFromCpu();
        }
        /// <summary>Download GPU weights back into the Sequential modules (for save).</summary>
        public void DownloadToCpu()
        {
            foreach (var st in stages) st.DownloadToCpu();
        }
        /// <summary>Run forward over [batch, inputDim] state in `inputBuf`. Returns Output buffer.</summary>
        public ComputeBuffer Forward(ComputeBuffer inputBuf, int batch)
        {
            // copy input into tape[0] (caller may pass a buffer that is already tape[0])
            if (inputBuf != tape[0])
                CopyBuf(inputBuf, tape[0], batch * InputDim);
            for (int i = 0; i < stages.Count; i++)
                stages[i].Forward(batch, tape[i], tape[i + 1]);
            return Output;
        }
        /// <summary>Run backward. dYBuf is upstream grad sized [batch, outputDim]. Computes per-param grads (replaces existing) and returns dX of size [batch, inputDim].</summary>
        public ComputeBuffer Backward(ComputeBuffer dYBuf, int batch)
        {
            // copy upstream grad into dtape[N]
            int N = stages.Count;
            if (dYBuf != dtape[N])
                CopyBuf(dYBuf, dtape[N], batch * OutputDim);
            for (int i = N - 1; i >= 0; i--)
                stages[i].Backward(batch, tape[i], tape[i + 1], dtape[i + 1], dtape[i]);
            return dtape[0];
        }

        public IEnumerable<GPUParam> Parameters()
        {
            foreach (var st in stages)
                foreach (var p in st.Parameters())
                    yield return p;
        }

        public void Dispose()
        {
            foreach (var st in stages) st.Dispose();
            foreach (var t in tape) t?.Release();
            foreach (var t in dtape) t?.Release();
            dummyBuf?.Release();
        }

        // ---------- helpers ----------
        internal void CopyBuf(ComputeBuffer src, ComputeBuffer dst, int n)
        {
            // CopyBuffer kernel reads from `G` and writes to `P` (we repurpose those slots).
            cs.SetInt("N_", n);
            cs.SetBuffer(kCopyBuffer, "G", src);
            cs.SetBuffer(kCopyBuffer, "P", dst);
            cs.Dispatch(kCopyBuffer, (n + 255) / 256, 1, 1);
        }

        internal int Div256(int n) => (n + 255) / 256;
        internal ComputeShader CS => cs;

        // ====================================================================
        // ============================ STAGES ================================
        // ====================================================================
        internal abstract class Stage : IDisposable
        {
            public int InDim, OutDim;
            public abstract void Forward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf);
            public abstract void Backward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf,
                                          ComputeBuffer dY, ComputeBuffer dX);
            public abstract IEnumerable<GPUParam> Parameters();
            public abstract void UploadFromCpu();
            public abstract void DownloadToCpu();
            public virtual void Dispose() { }
        }

        // -------------------------------------------------------------------- Dense
        internal sealed class DenseStage : Stage
        {
            readonly GPUMLP owner;
            public readonly Dense cpu;
            public GPUParam W, B; // B may be null if cpu.bias == false
            public bool hasBias;
            int OUT, IN;
            // pre-activation cache: this is the Dense forward's own output (no activation here).
            // The kernel `DenseFwd` fuses bias + activation, but we want a plain matmul because the
            // activation stage is separate. We pass activation = -1 (linear) when calling DenseFwd.

            public DenseStage(GPUMLP owner, int inF, int outF, Dense cpu)
            {
                this.owner = owner;
                this.cpu = cpu;
                IN = inF; OUT = outF;
                InDim = inF; OutDim = outF;
                hasBias = cpu.biases != null;
                W = new GPUParam(OUT * IN, "W");
                if (hasBias) B = new GPUParam(OUT, "b");
            }

            public override void UploadFromCpu()
            {
                W.P.SetData(cpu.weights.ToArray());
                if (hasBias) B.P.SetData(cpu.biases.ToArray());
            }
            public override void DownloadToCpu()
            {
                float[] w = new float[OUT * IN]; W.P.GetData(w);
                Tensor.CopyTo(Tensor.Constant(w).Reshape(OUT, IN), cpu.weights);
                if (hasBias) { float[] b = new float[OUT]; B.P.GetData(b); Tensor.CopyTo(Tensor.Constant(b), cpu.biases); }
            }

            public override void Forward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf)
            {
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("IN_", IN);
                cs.SetInt("OUT_", OUT);
                cs.SetInt("activation", -1);     // linear; activation stage runs separately
                cs.SetInt("has_bias", hasBias ? 1 : 0);
                int kFwd = owner.kDenseFwd;
                cs.SetBuffer(kFwd, "X", inBuf);
                cs.SetBuffer(kFwd, "Y", outBuf);
                cs.SetBuffer(kFwd, "W", W.P);
                if (hasBias) cs.SetBuffer(kFwd, "b", B.P);
                else { cs.SetBuffer(kFwd, "b", owner.dummyBuf); /* dummy bind — must be a distinct buffer */ }
                cs.Dispatch(kFwd, (OUT + 15) / 16, (batch + 15) / 16, 1);
            }

            public override void Backward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf,
                                          ComputeBuffer dY, ComputeBuffer dX)
            {
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("IN_", IN);
                cs.SetInt("OUT_", OUT);

                if (owner.RequiresGrad)
                {
                    // dW
                    int kBwdW = owner.kDenseBwdW;
                    cs.SetBuffer(kBwdW, "X", inBuf);
                    cs.SetBuffer(kBwdW, "dY", dY);
                    cs.SetBuffer(kBwdW, "dW", W.G);
                    cs.Dispatch(kBwdW, (IN + 15) / 16, (OUT + 15) / 16, 1);

                    if (hasBias)
                    {
                        int kBwdB = owner.kDenseBwdB;
                        cs.SetBuffer(kBwdB, "dY", dY);
                        cs.SetBuffer(kBwdB, "dB", B.G);
                        cs.Dispatch(kBwdB, owner.Div256(OUT), 1, 1);
                    }
                }

                // dX = dY @ W
                int kBwdIn = owner.kDenseBwdIn;
                cs.SetBuffer(kBwdIn, "dY", dY);
                cs.SetBuffer(kBwdIn, "W", W.P);
                cs.SetBuffer(kBwdIn, "dX", dX);
                cs.Dispatch(kBwdIn, (IN + 15) / 16, (batch + 15) / 16, 1);
            }

            public override IEnumerable<GPUParam> Parameters()
            {
                yield return W;
                if (hasBias) yield return B;
            }
            public override void Dispose() { W?.Dispose(); B?.Dispose(); }
        }

        // -------------------------------------------------------------------- RMSNorm
        internal sealed class RMSNormStage : Stage
        {
            readonly GPUMLP owner;
            public readonly RMSNorm cpu;
            public GPUParam Gamma;
            public ComputeBuffer msx;   // [maxBatch]
            public ComputeBuffer xhat;  // [maxBatch * F]
            int F;
            float eps;

            public RMSNormStage(GPUMLP owner, int features, RMSNorm cpu)
            {
                this.owner = owner;
                this.cpu = cpu;
                F = features; InDim = F; OutDim = F;
                eps = cpu.Epsilon; // mirror whatever the CPU RMSNorm was constructed with.
                Gamma = new GPUParam(F, "gamma");
                msx  = new ComputeBuffer(owner.MaxBatch, 4, ComputeBufferType.Structured);
                xhat = new ComputeBuffer(owner.MaxBatch * F, 4, ComputeBufferType.Structured);
            }
            public override void UploadFromCpu() { Gamma.P.SetData(cpu.gamma.ToArray()); }
            public override void DownloadToCpu()
            {
                float[] g = new float[F]; Gamma.P.GetData(g);
                Tensor.CopyTo(Tensor.Constant(g), cpu.gamma);
            }

            public override void Forward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf)
            {
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("F_", F);
                cs.SetFloat("eps_norm", eps);
                cs.SetBuffer(owner.kRMSFwd, "X", inBuf);
                cs.SetBuffer(owner.kRMSFwd, "Y", outBuf);
                cs.SetBuffer(owner.kRMSFwd, "gamma", Gamma.P);
                cs.SetBuffer(owner.kRMSFwd, "msx", msx);
                cs.SetBuffer(owner.kRMSFwd, "xhat", xhat);
                cs.Dispatch(owner.kRMSFwd, batch, 1, 1);
            }

            public override void Backward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf,
                                          ComputeBuffer dY, ComputeBuffer dX)
            {
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("F_", F);
                cs.SetFloat("eps_norm", eps);

                if (owner.RequiresGrad)
                {
                    cs.SetBuffer(owner.kRMSBwdGamma, "dY", dY);
                    cs.SetBuffer(owner.kRMSBwdGamma, "xhat", xhat);
                    cs.SetBuffer(owner.kRMSBwdGamma, "dGamma", Gamma.G);
                    cs.Dispatch(owner.kRMSBwdGamma, owner.Div256(F), 1, 1);
                }

                cs.SetBuffer(owner.kRMSBwd, "X", inBuf);
                cs.SetBuffer(owner.kRMSBwd, "dY", dY);
                cs.SetBuffer(owner.kRMSBwd, "dX", dX);
                cs.SetBuffer(owner.kRMSBwd, "gamma", Gamma.P);
                cs.SetBuffer(owner.kRMSBwd, "msx", msx);
                cs.Dispatch(owner.kRMSBwd, batch, 1, 1);
            }

            public override IEnumerable<GPUParam> Parameters() { yield return Gamma; }
            public override void Dispose() { Gamma?.Dispose(); msx?.Release(); xhat?.Release(); }
        }

        // -------------------------------------------------------------------- Activation
        internal sealed class ActivationStage : Stage
        {
            readonly GPUMLP owner;
            public readonly GPUActKind kind;

            public ActivationStage(GPUMLP owner, int dim, GPUActKind kind)
            {
                this.owner = owner;
                this.kind = kind;
                InDim = dim; OutDim = dim;
            }

            public override void UploadFromCpu() { }
            public override void DownloadToCpu() { }

            public override void Forward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf)
            {
                // outBuf = act(inBuf): copy then apply in-place.
                owner.CopyBuf(inBuf, outBuf, batch * OutDim);
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("OUT_", OutDim);
                cs.SetInt("activation", (int)kind);
                cs.SetBuffer(owner.kActFwdInplace, "Y", outBuf);
                cs.Dispatch(owner.kActFwdInplace, owner.Div256(batch * OutDim), 1, 1);
            }

            public override void Backward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf,
                                          ComputeBuffer dY, ComputeBuffer dX)
            {
                // dX = dY * act'(inBuf)
                // ApplyActGrad does `dY *= act'(Z)` in-place — we want it on a *copy* into dX,
                // since we shouldn't clobber dY. Strategy: copy dY -> dX, then run ApplyActGrad
                // on dX with Z = inBuf.
                owner.CopyBuf(dY, dX, batch * OutDim);
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("OUT_", OutDim);
                cs.SetInt("activation", (int)kind);
                cs.SetBuffer(owner.kApplyActGrad, "Y", inBuf); // Z (pre-activation = stage input)
                cs.SetBuffer(owner.kApplyActGrad, "dY", dX);   // operate in-place on dX
                cs.Dispatch(owner.kApplyActGrad, owner.Div256(batch * OutDim), 1, 1);
            }

            public override IEnumerable<GPUParam> Parameters() { yield break; }
        }

        // -------------------------------------------------------------------- Softplus tail
        internal sealed class SoftplusStage : Stage
        {
            readonly GPUMLP owner;
            public SoftplusStage(GPUMLP owner, int dim) { this.owner = owner; InDim = dim; OutDim = dim; }
            public override void UploadFromCpu() { }
            public override void DownloadToCpu() { }
            public override void Forward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf)
            {
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("OUT_", OutDim);
                cs.SetBuffer(owner.kSoftplusFwd, "X", inBuf);
                cs.SetBuffer(owner.kSoftplusFwd, "Y", outBuf);
                cs.Dispatch(owner.kSoftplusFwd, owner.Div256(batch * OutDim), 1, 1);
            }
            public override void Backward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf,
                                          ComputeBuffer dY, ComputeBuffer dX)
            {
                var cs = owner.CS;
                cs.SetInt("B", batch);
                cs.SetInt("OUT_", OutDim);
                cs.SetBuffer(owner.kSoftplusBwd, "X", inBuf);
                cs.SetBuffer(owner.kSoftplusBwd, "dY", dY);
                cs.SetBuffer(owner.kSoftplusBwd, "dX", dX);
                cs.Dispatch(owner.kSoftplusBwd, owner.Div256(batch * OutDim), 1, 1);
            }
            public override IEnumerable<GPUParam> Parameters() { yield break; }
        }

        // -------------------------------------------------------------------- Softmax tail (placeholder)
        // The discrete head reaches PPOLossCS which performs an online safe softmax + grad
        // directly on logits; we keep this stage as a no-op forward (logits == output) and
        // a no-op backward (dLogits is already the discrete head's dY at this point).
        internal sealed class SoftmaxTailStage : Stage
        {
            readonly GPUMLP owner;
            public SoftmaxTailStage(GPUMLP owner, int dim) { this.owner = owner; InDim = dim; OutDim = dim; }
            public override void UploadFromCpu() { }
            public override void DownloadToCpu() { }
            public override void Forward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf)
            {
                // pass-through: outputs = logits
                owner.CopyBuf(inBuf, outBuf, batch * OutDim);
            }
            public override void Backward(int batch, ComputeBuffer inBuf, ComputeBuffer outBuf,
                                          ComputeBuffer dY, ComputeBuffer dX)
            {
                owner.CopyBuf(dY, dX, batch * OutDim);
            }
            public override IEnumerable<GPUParam> Parameters() { yield break; }
        }

    }
}
