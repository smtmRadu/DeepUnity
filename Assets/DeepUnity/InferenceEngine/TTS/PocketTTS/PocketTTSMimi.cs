using System;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        using Cfg = PocketTTSConfig;

        // Mimi decoder: flow latents [T,32] -> 24 kHz waveform [T*1920]. Offline full-sequence
        // (reproduces the streaming per-latent decode; the streaming overlap-add of the causal
        // convs/convtr is exactly a full causal pass). Graph (see ../SPEC.md):
        //   denorm -> quantizer Conv1d k1 32->512 -> grouped ConvTranspose1d x16
        //   -> 2L decoder_transformer (LayerNorm + RoPE + causal MHA + layer_scale)
        //   -> SEANet decoder (ratios [6,5,4], ELU, causal convs, ConvTranspose1d) -> wav
        // [T,C] layout throughout, matching PocketTTSCS. FIRST DRAFT — ABI-uncertain spots flagged.
        public class PocketTTSMimi : IDisposable
        {
            readonly ComputeShader cs;
            readonly PocketTTSWeights w;
            int kCopy, kSlice, kSliceCols, kZero, kAdd, kScale, kChanAdd, kAct, kLinear, kLinearQ8,
                kConv, kConvTr, kConvTrG, kLN, kRope, kAttn;

            // scratch (grown per length)
            ComputeBuffer a, b, c, d, resid, attnScratch, qBuf, kBuf, vBuf, qkvBuf;
            int curCap;

            public Action<string, ComputeBuffer, int> DebugTap;   // (name, buf[T,C], count)
            public float DecodeMs { get; private set; }

            public PocketTTSMimi(PocketTTSWeights weights)
            {
                w = weights;
                cs = DeepUnityMeta.PocketTTSCS;
                kCopy = cs.FindKernel("CopyBuffer");
                kSlice = cs.FindKernel("CopySlice");
                kSliceCols = cs.FindKernel("SliceCols");
                kZero = cs.FindKernel("ZeroBuffer");
                kAdd = cs.FindKernel("AddResidual");
                kScale = cs.FindKernel("ScaleBuf");
                kChanAdd = cs.FindKernel("ChannelScaleAdd");
                kAct = cs.FindKernel("Activate");
                kLinear = cs.FindKernel("LinearBias");
                kLinearQ8 = cs.FindKernel("LinearBiasQ8");
                kConv = cs.FindKernel("Conv1D");
                kConvTr = cs.FindKernel("ConvTranspose1D");
                kConvTrG = cs.FindKernel("ConvTranspose1DGrouped");
                kLN = cs.FindKernel("LayerNormT");
                kRope = cs.FindKernel("ApplyRoPE");
                kAttn = cs.FindKernel("CausalAttention");
            }

            static int Div256(int n) => (n + 255) / 256;
            static void Grow(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(Math.Max(count, 1), 4, ComputeBufferType.Structured);
            }

            // D3D11 caps thread groups at 65535 PER DIMENSION. SEANet stages run 122880 elements
            // PER LATENT (>16.7M elements past ~136 latents = the live long-reply crash). Spill
            // extra groups into Y; the converted kernels reconstruct the linear index (LinearId),
            // and a 1-D dispatch (y==1) stays bit-identical to the old path.
            void Dispatch1D(int kernel, int elements)
            {
                int g = Div256(elements);
                if (g <= 65535) { cs.Dispatch(kernel, Math.Max(g, 1), 1, 1); return; }
                cs.Dispatch(kernel, 65535, (g + 65534) / 65535, 1);
            }

            void Ensure(int T)
            {
                int frames = T * Cfg.MIMI_STEPS_PER_LATENT;           // 16T (transformer length)
                int wav = T * Cfg.SAMPLES_PER_LATENT;                 // 1920T
                // widest activation buffer: after conv0 512ch @16T, then convtr stages grow length
                // but shrink channels; SEANet peak = 96T*256 (stage0 out) up to 1920T*64 (final).
                int peak = Math.Max(frames * 512, wav * 64);
                if (peak <= curCap) return;
                curCap = peak;
                Grow(ref a, peak); Grow(ref b, peak); Grow(ref c, peak); Grow(ref d, peak);
                Grow(ref resid, frames * 512);
                Grow(ref attnScratch, frames * 512);
                Grow(ref qBuf, frames * 512); Grow(ref kBuf, frames * 512); Grow(ref vBuf, frames * 512);
                Grow(ref qkvBuf, frames * 1536);
            }

            // ---- op helpers ----
            void Copy(ComputeBuffer dst, ComputeBuffer src, int n)
            { cs.SetInt("buffer_size", n); cs.SetBuffer(kCopy, "buf_a", dst); cs.SetBuffer(kCopy, "buf_b", src); Dispatch1D(kCopy, n); }

            void SliceCols(ComputeBuffer src, ComputeBuffer dst, int T, int inDim, int outDim, int colOff)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("copy_src_offset", colOff);
                cs.SetBuffer(kSliceCols, "X", src); cs.SetBuffer(kSliceCols, "Y", dst);
                Dispatch1D(kSliceCols, T * outDim);
            }

            void Act(ComputeBuffer buf, int n, int act, float leaky = 0.01f)
            {
                cs.SetInt("buffer_size", n); cs.SetInt("activation_type", act); cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kAct, "inout_buf", buf); Dispatch1D(kAct, n);
            }

            // fp16 OR int8: a '<name>.weight.scales' sibling => q8 (LinearBiasQ8). Mimi's
            // decoder_transformer linears are int8-able; conv/convtr kernels stay fp16 (3D).
            void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim, bool bias, int act = 0)
            {
                ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                int k = scales != null ? kLinearQ8 : kLinear;
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                cs.SetBuffer(k, "Y", y);
                cs.Dispatch(k, 1, (T + 7) / 8, (outDim + 31) / 32);
            }

            // elemOff/elemCount: #29 range sub-dispatch over output elements [off, off+count) —
            // elem_offset is set for the dispatch and reset SYNCHRONOUSLY (the shader object is
            // shared with the FlowLM AR path, so it must never stay non-zero across a yield).
            // Defaults = the whole op, bit-identical to the pre-#29 dispatch.
            void Conv(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                      int inCh, int outCh, int kernel, int stride, int dilation, int padLeft, bool bias, int act = 0,
                      bool padReplicate = false, int elemOff = 0, int elemCount = -1)
            {
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen); cs.SetInt("in_dim", inCh);
                cs.SetInt("out_dim", outCh); cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("conv_dilation", dilation); cs.SetInt("pad_left", padLeft);
                cs.SetInt("pad_replicate", padReplicate ? 1 : 0);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0); cs.SetFloat("leaky_slope", 0.01f);
                cs.SetBuffer(kConv, "X", x); cs.SetBuffer(kConv, "W", w.Get(name + ".weight"));
                cs.SetBuffer(kConv, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                cs.SetBuffer(kConv, "Y", y);
                cs.SetInt("elem_offset", elemOff);
                Dispatch1D(kConv, elemCount < 0 ? outLen * outCh : elemCount);
                if (elemOff != 0) cs.SetInt("elem_offset", 0);
            }

            // groups==1 -> ConvTranspose1D, groups>1 -> ConvTranspose1DGrouped. outLen = inLen*stride
            // (streaming-trimmed; the trailing k-s "partial" is dropped, as the ref streaming carries it).
            void ConvTr(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                        int inCh, int outCh, int kernel, int stride, int groups, bool bias,
                        int elemOff = 0, int elemCount = -1)
            {
                // ALWAYS the grouped kernel: it's proven (upsample corr 1.000000) and with n_groups=1
                // reduces bit-for-bit to the plain ConvTranspose1D math ((ic*out_dim+oc)*K+k). The
                // plain kernel path is unexercised elsewhere, so this consolidates on the tested one.
                int kk = kConvTrG;
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen); cs.SetInt("in_dim", inCh);
                cs.SetInt("out_dim", outCh); cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("pad_left", 0); cs.SetInt("has_bias", bias ? 1 : 0); cs.SetInt("n_groups", groups);
                cs.SetInt("activation_type", 0);
                cs.SetBuffer(kk, "X", x); cs.SetBuffer(kk, "W", w.Get(name + ".weight"));
                cs.SetBuffer(kk, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                cs.SetBuffer(kk, "Y", y);
                cs.SetInt("elem_offset", elemOff);
                Dispatch1D(kk, elemCount < 0 ? outLen * outCh : elemCount);
                if (elemOff != 0) cs.SetInt("elem_offset", 0);
            }

            // ================= #29: GPU-cost-bounded slicing of the fat SEANet/transformer ops ====
            // One Mimi window decode is ~13 GMAC; issued whole-op, single convs reached 25-55 ms of
            // GPU and owned the frame (the mimi_decode spikes in the talk-perf report). These
            // helpers split any op whose MAC count exceeds PocketTTS.GpuMacsPerTick (runtime
            // self-calibrated — no GPU-specific tuning) into element/row ranges (via the shader's
            // elem_offset) and yield between ranges, so a frame-paced caller never issues more
            // than ~a few ms of GPU per tick. Range splits recompute the ≤255 overlap elements at
            // 256-thread tails with identical results — parity-neutral.
            static int SlicesFor(long macs, int total)
                => (int)Math.Min(total, (macs + PocketTTS.GpuMacsPerTick - 1) / PocketTTS.GpuMacsPerTick);

            System.Collections.IEnumerator ConvSliced(string name, ComputeBuffer x, ComputeBuffer y,
                int outLen, int inLen, int inCh, int outCh, int kernel, int stride, int dilation,
                int padLeft, bool bias, int act = 0, bool padReplicate = false)
            {
                int total = outLen * outCh;
                int slices = SlicesFor((long)total * inCh * kernel, total);
                int per = (total + slices - 1) / slices;
                for (int off = 0; off < total; off += per)
                {
                    Conv(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, dilation, padLeft,
                         bias, act, padReplicate, off, Math.Min(per, total - off));
                    if (off + per < total) yield return null;
                }
            }

            System.Collections.IEnumerator ConvTrSliced(string name, ComputeBuffer x, ComputeBuffer y,
                int outLen, int inLen, int inCh, int outCh, int kernel, int stride, int groups, bool bias)
            {
                int total = outLen * outCh;
                int slices = SlicesFor((long)total * (inCh / groups) * (kernel / stride), total);
                int per = (total + slices - 1) / slices;
                for (int off = 0; off < total; off += per)
                {
                    ConvTr(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, groups, bias,
                           off, Math.Min(per, total - off));
                    if (off + per < total) yield return null;
                }
            }

            // Row-sliced LinearBias/Q8 (the 3-D linear kernels slice on rows via elem_offset).
            // Re-applies every uniform per sub-dispatch: the shared shader may be re-programmed by
            // the FlowLM AR path between our yields.
            System.Collections.IEnumerator LinearRows(string name, ComputeBuffer x, ComputeBuffer y,
                int T, int inDim, int outDim, bool bias, int act = 0)
            {
                int slices = SlicesFor((long)T * inDim * outDim, T);
                int rows = (T + slices - 1) / slices;
                for (int r0 = 0; r0 < T; r0 += rows)
                {
                    ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                    int k = scales != null ? kLinearQ8 : kLinear;
                    cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                    cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                    cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                    cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                    if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                    cs.SetBuffer(k, "Y", y);
                    cs.SetInt("elem_offset", r0);
                    cs.Dispatch(k, 1, (Math.Min(rows, T - r0) + 7) / 8, (outDim + 31) / 32);
                    cs.SetInt("elem_offset", 0);
                    if (r0 + rows < T) yield return null;
                }
            }

            void LayerNorm(string name, ComputeBuffer x, ComputeBuffer y, int T, int dim)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", dim); cs.SetFloat("norm_eps", 1e-5f);
                cs.SetBuffer(kLN, "norm_input", x); cs.SetBuffer(kLN, "norm_output", y);
                cs.SetBuffer(kLN, "ln_gamma", w.Get(name + ".weight")); cs.SetBuffer(kLN, "ln_beta", w.Get(name + ".bias"));
                cs.Dispatch(kLN, Div256(T), 1, 1);
            }

            void RoPE(ComputeBuffer buf, int T, int heads, int hd)
            {
                cs.SetInt("seq_len", T); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                cs.SetInt("pos_offset", 0); cs.SetFloat("rope_theta", Cfg.MIMI_TF_ROPE_THETA);
                cs.SetBuffer(kRope, "inout_buf", buf);
                cs.Dispatch(kRope, Div256(T * heads * (hd / 2)), 1, 1);
            }

            // residual += sublayer * layer_scale[c]
            void ChannelScaleAdd(ComputeBuffer residBuf, ComputeBuffer sub, string lsName, int T, int dim)
            {
                cs.SetInt("buffer_size", T * dim); cs.SetInt("norm_dim", dim);
                cs.SetBuffer(kChanAdd, "buf_a", residBuf); cs.SetBuffer(kChanAdd, "buf_b", sub);
                cs.SetBuffer(kChanAdd, "ch_scale", w.Get(lsName + ".scale"));
                Dispatch1D(kChanAdd, T * dim);
            }

            // SEANet causal resblock: x + conv2(elu(conv1(elu(x)))). conv1 k3 dim->hidden pad2, conv2 k1 hidden->dim pad0.
            // Uses scratch c,d; out written in place to `x` buffer's alias handled by caller (returns in `outBuf`).
            // #29: enumerator — the convs are MAC-sliced and each range is one tick for the caller.
            System.Collections.IEnumerator ResBlockTicks(string p, ComputeBuffer x, ComputeBuffer outBuf, int T, int dim)
            {
                int hidden = dim / Cfg.MIMI_COMPRESS;
                // v = elu(x); conv1 -> c
                Copy(c, x, T * dim);
                Act(c, T * dim, 7);
                var e = ConvSliced(p + "/block/1/conv", c, d, T, T, dim, hidden, 3, 1, 1, 2, bias: true);
                while (e.MoveNext()) yield return null;
                Act(d, T * hidden, 7);
                e = ConvSliced(p + "/block/3/conv", d, c, T, T, hidden, dim, 1, 1, 1, 0, bias: true);
                while (e.MoveNext()) yield return null;
                // outBuf = x + c
                if (outBuf != x) Copy(outBuf, x, T * dim);
                cs.SetInt("buffer_size", T * dim);
                cs.SetBuffer(kAdd, "buf_a", outBuf); cs.SetBuffer(kAdd, "buf_b", c);
                Dispatch1D(kAdd, T * dim);   // SEANet stage2: 122880 elems/latent — the old 1-D dispatch was the long-reply crash
            }

            // set by the issue chain: the buffer holding the final waveform + its sample count
            ComputeBuffer lastWavBuf;
            int lastWavLen;

            /// <summary>Decode Mimi latents [T,32] (row-major) -> wav float[T*1920].
            /// The Mimi decoder's input is the DENORMED latent (quantizer input); the dump's
            /// latents.npy already carries that. denorm is a flow→mimi BOUNDARY op (P4), NOT part
            /// of the Mimi decoder — pass embMean/embStd non-null only when feeding RAW flow
            /// latents (e2e); P1 passes null (latents are already denormed).
            /// SYNC form (probes/offline): drains the sliced issue chain — IDENTICAL dispatches in
            /// identical order — then a blocking readback. Parity-neutral vs the pre-split code.</summary>
            public float[] Decode(float[] latents, int T, float[] embMean = null, float[] embStd = null)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var e = DecodeIssueYielding(Denorm(latents, T, embMean, embStd), T);
                while (e.MoveNext()) { }
                float[] wav = new float[lastWavLen];
                lastWavBuf.GetData(wav, 0, 0, lastWavLen);
                sw.Stop();
                DecodeMs = (float)sw.Elapsed.TotalMilliseconds;
                return wav;
            }

            /// <summary>STREAMING form (#29): the decode chain is SLICED — one GPU-heavy group per
            /// MoveNext (callers surface each yield as a frame boundary), and the wav readback is
            /// ASYNC (delivered 1-3 frames later; the voice ring's prebuffer absorbs it). Same
            /// dispatches as Decode. wavOut must be [T*1920].</summary>
            public System.Collections.IEnumerator DecodeYielding(float[] latents, int T,
                float[] embMean, float[] embStd, float[] wavOut, bool async)
            {
                var e = DecodeIssueYielding(Denorm(latents, T, embMean, embStd), T);
                while (e.MoveNext()) yield return null;
                if (async && SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(lastWavBuf, lastWavLen * 4, 0);
                    // #29: frame-counted fallback cap (a MoveNext-counted cap tripped after ~2 frames
                    // under the budget pump and hard-stalled on the whole GPU queue — the flush outliers).
                    int startFrame = UnityEngine.Time.frameCount;
                    while (!req.done)
                    {
                        if (UnityEngine.Time.frameCount - startFrame > 600)
                        { PocketTTS.LastHeavyTick = "readback_hardwait"; req.WaitForCompletion(); break; }
                        yield return PocketTTS.GpuWait;
                    }
                    if (!req.hasError)
                    {
                        req.GetData<float>().CopyTo(wavOut);
                        yield break;
                    }
                }
                lastWavBuf.GetData(wavOut, 0, 0, lastWavLen);   // unsupported/error fallback: sync
            }

            float[] Denorm(float[] latents, int T, float[] embMean, float[] embStd)
            {
                if (embMean == null || embStd == null) return latents;
                var input = new float[T * Cfg.LDIM];
                for (int t = 0; t < T; t++)
                    for (int cc = 0; cc < Cfg.LDIM; cc++)
                        input[t * Cfg.LDIM + cc] = latents[t * Cfg.LDIM + cc] * embStd[cc] + embMean[cc];
                return input;
            }

            // The full decode dispatch chain, yielding after each GPU-heavy slice (~10 slices) so a
            // frame-paced caller never issues more than ~10-15 ms of GPU in one tick. The final wav
            // lands in lastWavBuf/lastWavLen (no readback here — the wrappers own it).
            System.Collections.IEnumerator DecodeIssueYielding(float[] input, int T)
            {
                Ensure(T);
                int F = T * Cfg.MIMI_STEPS_PER_LATENT;   // transformer frames (16T)
                a.SetData(input, 0, 0, T * Cfg.LDIM);

                // ---- quantizer Conv1d k1 32->512 -> b [T,512], then grouped ConvTr x16 -> resid
                Conv("mimi/quantizer/output_proj", a, b, T, T, Cfg.LDIM, Cfg.QUANT_OUT, 1, 1, 1, 0, bias: false);
                DebugTap?.Invoke("quant_out", b, T * Cfg.QUANT_OUT);
                ConvTr("mimi/upsample/convtr/convtr", b, resid, F, T, Cfg.QUANT_OUT, Cfg.MIMI_SEANET_DIM,
                       Cfg.UPSAMPLE_KERNEL, Cfg.UPSAMPLE_STRIDE, Cfg.UPSAMPLE_GROUPS, bias: false);
                DebugTap?.Invoke("mimi_upsampled", resid, F * Cfg.MIMI_SEANET_DIM);
                yield return null;

                // ---- decoder_transformer: 2 layers, d512/8h, LayerNorm + RoPE + causal + layer_scale
                int dim = Cfg.MIMI_TF_DIM, heads = Cfg.MIMI_TF_HEADS, hd = Cfg.MIMI_TF_HEAD_DIM;
                float attScale = 1f / Mathf.Sqrt(hd);
                for (int li = 0; li < Cfg.MIMI_TF_LAYERS; li++)
                {
                    string lp = $"mimi/decoder_transformer/transformer/layers/{li}";
                    // -- self-attn block  (#29: the whole layer used to be ONE ~2.5 GMAC tick — the
                    // fattest slice in the pipeline. Now attn and each FFN matmul are their own
                    // MAC-bounded ticks.)
                    LayerNorm(lp + "/norm1", resid, a, F, dim);
                    var lr = LinearRows(lp + "/self_attn/in_proj", a, qkvBuf, F, dim, 3 * dim, bias: false);   // [F,1536]
                    while (lr.MoveNext()) yield return null;
                    SliceCols(qkvBuf, qBuf, F, 3 * dim, dim, 0);
                    SliceCols(qkvBuf, kBuf, F, 3 * dim, dim, dim);
                    SliceCols(qkvBuf, vBuf, F, 3 * dim, dim, 2 * dim);
                    RoPE(qBuf, F, heads, hd);
                    RoPE(kBuf, F, heads, hd);
                    yield return null;
                    cs.SetInt("seq_len", F); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetInt("rope_on", 1); cs.SetFloat("scale", attScale);
                    cs.SetInt("attn_context", Cfg.MIMI_TF_CONTEXT);   // 250 — sliding causal window (Moshi/Mimi)
                    cs.SetBuffer(kAttn, "Q", qBuf); cs.SetBuffer(kAttn, "K", kBuf); cs.SetBuffer(kAttn, "V", vBuf);
                    cs.SetBuffer(kAttn, "AttendedValues", attnScratch);
                    cs.Dispatch(kAttn, F, heads, 1);
                    Linear(lp + "/self_attn/out_proj", attnScratch, a, F, dim, dim, bias: false);
                    ChannelScaleAdd(resid, a, lp + "/layer_scale_1", F, dim);
                    yield return null;
                    // -- ffn block
                    LayerNorm(lp + "/norm2", resid, a, F, dim);
                    lr = LinearRows(lp + "/linear1", a, b, F, dim, Cfg.MIMI_TF_FFN, bias: false, act: 2);  // GELU
                    while (lr.MoveNext()) yield return null;
                    yield return null;
                    lr = LinearRows(lp + "/linear2", b, a, F, Cfg.MIMI_TF_FFN, dim, bias: false);
                    while (lr.MoveNext()) yield return null;
                    ChannelScaleAdd(resid, a, lp + "/layer_scale_2", F, dim);
                    yield return null;
                }
                DebugTap?.Invoke("mimi_xf_out", resid, F * dim);

                // ---- SEANet decoder: conv0 k7 512->512, then 3 x [ELU, ConvTr, ResBlock], then ELU, conv k3 ->1
                var e0 = ConvSliced("mimi/decoder/model/0/conv", resid, a, F, F, 512, 512, 7, 1, 1, 6, bias: true);
                while (e0.MoveNext()) yield return null;
                DebugTap?.Invoke("seanet_conv0", a, F * 512);
                yield return null;
                ComputeBuffer cur = a;                 // [len, ch]
                int len = F, ch = 512;
                int[] ratios = Cfg.MIMI_RATIOS;
                int[] modelIdx = { 2, 5, 8 };          // ConvTr module indices; resblock at +1
                for (int s = 0; s < 3; s++)
                {
                    int r = ratios[s], outCh = ch / 2, outLen = len * r;
                    Act(cur, len * ch, 7);             // ELU before convtr (model[idx-1])
                    ComputeBuffer nxt = (cur == a) ? b : a;
                    var et = ConvTrSliced($"mimi/decoder/model/{modelIdx[s]}/convtr", cur, nxt, outLen, len, ch, outCh,
                                          r * 2, r, 1, bias: true);
                    while (et.MoveNext()) yield return null;
                    len = outLen; ch = outCh; cur = nxt;
                    yield return null;                 // convtr and resblock are the heavy tail — split them
                    // resblock (model[idx+1]); in place via c/d scratch, out to the other of a/b
                    ComputeBuffer rbOut = (cur == a) ? b : a;
                    var er = ResBlockTicks($"mimi/decoder/model/{modelIdx[s] + 1}", cur, rbOut, len, ch);
                    while (er.MoveNext()) yield return null;
                    cur = rbOut;
                    DebugTap?.Invoke($"seanet_stage{s}", cur, len * ch);
                    yield return null;
                }
                Act(cur, len * ch, 7);                 // final ELU (model[10])
                ComputeBuffer wavBuf = (cur == a) ? b : a;
                Conv("mimi/decoder/model/11/conv", cur, wavBuf, len, len, 64, 1, 3, 1, 1, 2, bias: true);
                lastWavBuf = wavBuf;                   // [len, 1] = [T*1920, 1]
                lastWavLen = len;
            }

            public void Dispose()
            {
                a?.Release(); b?.Release(); c?.Release(); d?.Release(); resid?.Release();
                attnScratch?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release(); qkvBuf?.Release();
            }
        }
    }
}
