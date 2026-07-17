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
                kConv, kConvT, kConvTr, kConvTrT, kConvTrG, kLN, kRope, kAttn, kAttnLegacy;
            int kLinearGemm, kLinearQ8Gemm;   // #31-P coalesced GEMM (decoder_transformer linears)

            // #30: kernel-parity switch — true routes every conv through the pre-#30 kernels and
            // attention through CausalAttentionLegacy, so the parity probe can compare tiled vs
            // legacy wav-to-wav on any machine (the .npy reference dumps live on the main dev box
            // only). Production always runs tiled (false). #31-P: also forces the LEGACY LinearBias
            // path (full-legacy baseline); the #31 axis alone is bisected via PocketTTS.FastKernels2.
            public static bool ForceLegacyKernels = false;

            // #31-P: decoder_transformer linears (K = 512 / 2048, both % 128 == 0) route to the
            // coalesced GEMM behind PocketTTS.FastKernels2. Convs are untouched (#30 owns them).
            static bool CoalEligible(int inDim)
                => PocketTTS.FastKernels2 && !ForceLegacyKernels && inDim % 128 == 0 && inDim <= 4096;

            const int CONV_TB = 8;              // output time-steps per tile — must match CT_TB
            const int CONV_TILE_SH = 7168;      // Conv1DTiled groupshared floats (CT_SHFLOATS)
            const int CONVTR_TILE_SH = 4096;    // ConvTranspose1DTiled groupshared floats

            static bool ConvTiledFits(int inCh, int kernel, int stride, int dilation)
                => !ForceLegacyKernels &&
                   ((CONV_TB - 1) * stride + (kernel - 1) * dilation + 1) * inCh <= CONV_TILE_SH;
            static bool ConvTrTiledFits(int inCh, int kernel, int stride, int groups)
                => !ForceLegacyKernels && groups == 1 &&
                   ((CONV_TB - 1 + kernel - 1) / stride + 2) * inCh <= CONVTR_TILE_SH;

            // scratch (grown per length)
            ComputeBuffer a, b, c, d, resid, attnScratch, qBuf, kBuf, vBuf, qkvBuf;
            int curCap;

            public Action<string, ComputeBuffer, int> DebugTap;   // (name, buf[T,C], count)
            public float DecodeMs { get; private set; }

            // #30 stage profiler: non-null => Mark() drains the GPU queue (1-float sync readback
            // of the stage's output buffer) after each pipeline stage and records the split.
            // Serializing the queue inflates the TOTAL vs a free-running decode — use the splits
            // for RELATIVE attribution only. Off (null) in production; RtfProbe owns it.
            public static System.Collections.Generic.List<(string name, float ms)> StageProfile;
            static readonly float[] profTmp = new float[1];
            readonly System.Diagnostics.Stopwatch profSw = new System.Diagnostics.Stopwatch();
            void Mark(string name, ComputeBuffer written)
            {
                if (StageProfile == null) return;
                written.GetData(profTmp, 0, 0, 1);          // drain: waits for all prior dispatches
                StageProfile.Add((name, (float)profSw.Elapsed.TotalMilliseconds));
                profSw.Restart();
            }

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
                kConvT = cs.FindKernel("Conv1DTiled");
                kConvTr = cs.FindKernel("ConvTranspose1D");
                kConvTrT = cs.FindKernel("ConvTranspose1DTiled");
                kConvTrG = cs.FindKernel("ConvTranspose1DGrouped");
                kLN = cs.FindKernel("LayerNormT");
                kRope = cs.FindKernel("ApplyRoPE");
                kAttn = cs.FindKernel("CausalAttention");
                kAttnLegacy = cs.FindKernel("CausalAttentionLegacy");
                kLinearGemm = cs.FindKernel("LinearBiasGemm");
                kLinearQ8Gemm = cs.FindKernel("LinearBiasQ8Gemm");
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

            // tiled conv kernels: ONE GROUP PER TILE (not per 256 elements) — same Y-spill rule.
            void DispatchTiles(int kernel, int tiles)
            {
                if (tiles <= 65535) { cs.Dispatch(kernel, Math.Max(tiles, 1), 1, 1); return; }
                cs.Dispatch(kernel, 65535, (tiles + 65534) / 65535, 1);
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
            // fromElem (#30 tail-restricted decode): first element to process — elements below it
            // keep their previous buffer contents (only ever garbage-permitted rows). 0 = whole op,
            // bit-identical to the pre-#30 dispatch.
            void Copy(ComputeBuffer dst, ComputeBuffer src, int n, int fromElem = 0)
            {
                cs.SetInt("buffer_size", n); cs.SetBuffer(kCopy, "buf_a", dst); cs.SetBuffer(kCopy, "buf_b", src);
                cs.SetInt("elem_offset", fromElem);
                Dispatch1D(kCopy, n - fromElem);
                if (fromElem != 0) cs.SetInt("elem_offset", 0);
            }

            void SliceCols(ComputeBuffer src, ComputeBuffer dst, int T, int inDim, int outDim, int colOff, int fromElem = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("copy_src_offset", colOff);
                cs.SetBuffer(kSliceCols, "X", src); cs.SetBuffer(kSliceCols, "Y", dst);
                cs.SetInt("elem_offset", fromElem);
                Dispatch1D(kSliceCols, T * outDim - fromElem);
                if (fromElem != 0) cs.SetInt("elem_offset", 0);
            }

            void Act(ComputeBuffer buf, int n, int act, int fromElem = 0, float leaky = 0.01f)
            {
                cs.SetInt("buffer_size", n); cs.SetInt("activation_type", act); cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kAct, "inout_buf", buf);
                cs.SetInt("elem_offset", fromElem);
                Dispatch1D(kAct, n - fromElem);
                if (fromElem != 0) cs.SetInt("elem_offset", 0);
            }

            // fp16 OR int8: a '<name>.weight.scales' sibling => q8 (LinearBiasQ8). Mimi's
            // decoder_transformer linears are int8-able; conv/convtr kernels stay fp16 (3D).
            // #31-P: eligible whole/tail ops route to the coalesced GEMM (elem_offset = first token
            // row = rowStart) behind PocketTTS.FastKernels2 — same tail-restriction semantics
            // (rows below rowStart are never touched), parity-gated not bit-exact.
            void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim, bool bias, int act = 0, int rowStart = 0)
            {
                ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                if (CoalEligible(inDim))
                {
                    int kc = scales != null ? kLinearQ8Gemm : kLinearGemm;
                    cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                    cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                    cs.SetBuffer(kc, "X", x); cs.SetBuffer(kc, "W", w.Get(name + ".weight"));
                    cs.SetBuffer(kc, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                    if (scales != null) cs.SetBuffer(kc, "W_scales", scales);
                    cs.SetBuffer(kc, "Y", y);
                    cs.SetInt("elem_offset", rowStart);
                    cs.Dispatch(kc, (outDim + 7) / 8, (T - rowStart + 7) / 8, 1);
                    if (rowStart != 0) cs.SetInt("elem_offset", 0);
                    return;
                }
                int k = scales != null ? kLinearQ8 : kLinear;
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                cs.SetBuffer(k, "Y", y);
                cs.SetInt("elem_offset", rowStart);
                cs.Dispatch(k, 1, (T - rowStart + 7) / 8, (outDim + 31) / 32);
                if (rowStart != 0) cs.SetInt("elem_offset", 0);
            }

            // elemOff/elemCount: #29 range sub-dispatch over output elements [off, off+count) —
            // elem_offset is set for the dispatch and reset SYNCHRONOUSLY (the shader object is
            // shared with the FlowLM AR path, so it must never stay non-zero across a yield).
            // Defaults = the whole op, bit-identical to the pre-#29 dispatch.
            // #30: whole-op calls route to the tiled kernel when the input window fits groupshared
            // (every conv in the Mimi decode does). Range sub-dispatches (#29 slicing) come from
            // ConvSliced, which slices TILES on the tiled path and never lands here with a range.
            void Conv(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                      int inCh, int outCh, int kernel, int stride, int dilation, int padLeft, bool bias, int act = 0,
                      bool padReplicate = false, int elemOff = 0, int elemCount = -1)
            {
                if (elemOff == 0 && elemCount < 0 && ConvTiledFits(inCh, kernel, stride, dilation))
                {
                    ConvTiled(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, dilation, padLeft,
                              bias, act, padReplicate, 0, (outLen + CONV_TB - 1) / CONV_TB);
                    return;
                }
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

            void ConvTiled(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                           int inCh, int outCh, int kernel, int stride, int dilation, int padLeft,
                           bool bias, int act, bool padReplicate, int tileOff, int tileCount)
            {
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen); cs.SetInt("in_dim", inCh);
                cs.SetInt("out_dim", outCh); cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("conv_dilation", dilation); cs.SetInt("pad_left", padLeft);
                cs.SetInt("pad_replicate", padReplicate ? 1 : 0);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0); cs.SetFloat("leaky_slope", 0.01f);
                cs.SetBuffer(kConvT, "X", x); cs.SetBuffer(kConvT, "W", w.Get(name + ".weight"));
                cs.SetBuffer(kConvT, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                cs.SetBuffer(kConvT, "Y", y);
                cs.SetInt("elem_offset", tileOff);
                DispatchTiles(kConvT, tileCount);
                if (tileOff != 0) cs.SetInt("elem_offset", 0);
            }

            void ConvTrTiled(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                             int inCh, int outCh, int kernel, int stride, bool bias, int tileOff, int tileCount)
            {
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen); cs.SetInt("in_dim", inCh);
                cs.SetInt("out_dim", outCh); cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("pad_left", 0); cs.SetInt("has_bias", bias ? 1 : 0);
                cs.SetBuffer(kConvTrT, "X", x); cs.SetBuffer(kConvTrT, "W", w.Get(name + ".weight"));
                cs.SetBuffer(kConvTrT, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                cs.SetBuffer(kConvTrT, "Y", y);
                cs.SetInt("elem_offset", tileOff);
                DispatchTiles(kConvTrT, tileCount);
                if (tileOff != 0) cs.SetInt("elem_offset", 0);
            }

            // groups==1 -> ConvTranspose1D, groups>1 -> ConvTranspose1DGrouped. outLen = inLen*stride
            // (streaming-trimmed; the trailing k-s "partial" is dropped, as the ref streaming carries it).
            void ConvTr(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                        int inCh, int outCh, int kernel, int stride, int groups, bool bias,
                        int elemOff = 0, int elemCount = -1)
            {
                // #30: groups==1 whole-op calls go tiled (all three SEANet upsample stages);
                // grouped (the depthwise x16 upsample, 0.1% of decode) stays on the proven
                // grouped kernel, which with n_groups=1 also serves as the legacy parity path.
                if (elemOff == 0 && elemCount < 0 && ConvTrTiledFits(inCh, kernel, stride, groups))
                {
                    ConvTrTiled(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, bias,
                                0, (outLen + CONV_TB - 1) / CONV_TB);
                    return;
                }
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

            // outRowStart (#30 tail-restricted decode): first output ROW that must be exact. On the
            // tiled path the start is floored to a tile boundary — the extra sub-tile rows may read
            // uncomputed inputs and produce garbage, which is fine: everything below outRowStart is
            // garbage-permitted by the caller's receptive-field bookkeeping and never read.
            System.Collections.IEnumerator ConvSliced(string name, ComputeBuffer x, ComputeBuffer y,
                int outLen, int inLen, int inCh, int outCh, int kernel, int stride, int dilation,
                int padLeft, bool bias, int act = 0, bool padReplicate = false, int outRowStart = 0)
            {
                if (ConvTiledFits(inCh, kernel, stride, dilation))   // #30: slice TILES, not elements
                {
                    int tiles = (outLen + CONV_TB - 1) / CONV_TB;
                    int tile0 = outRowStart / CONV_TB;
                    int tSlices = SlicesFor((long)(outLen - tile0 * CONV_TB) * outCh * inCh * kernel, tiles - tile0);
                    int perT = (tiles - tile0 + tSlices - 1) / tSlices;
                    for (int tOff = tile0; tOff < tiles; tOff += perT)
                    {
                        ConvTiled(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, dilation,
                                  padLeft, bias, act, padReplicate, tOff, Math.Min(perT, tiles - tOff));
                        if (tOff + perT < tiles) yield return null;
                    }
                    yield break;
                }
                int total = outLen * outCh;
                int start = outRowStart * outCh;
                int slices = SlicesFor((long)(total - start) * inCh * kernel, total - start);
                int per = (total - start + slices - 1) / slices;
                for (int off = start; off < total; off += per)
                {
                    Conv(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, dilation, padLeft,
                         bias, act, padReplicate, off, Math.Min(per, total - off));
                    if (off + per < total) yield return null;
                }
            }

            System.Collections.IEnumerator ConvTrSliced(string name, ComputeBuffer x, ComputeBuffer y,
                int outLen, int inLen, int inCh, int outCh, int kernel, int stride, int groups, bool bias,
                int outRowStart = 0)
            {
                if (ConvTrTiledFits(inCh, kernel, stride, groups))   // #30: slice TILES, not elements
                {
                    int tiles = (outLen + CONV_TB - 1) / CONV_TB;
                    int tile0 = outRowStart / CONV_TB;
                    int tSlices = SlicesFor((long)(outLen - tile0 * CONV_TB) * outCh * inCh * (kernel / stride), tiles - tile0);
                    int perT = (tiles - tile0 + tSlices - 1) / tSlices;
                    for (int tOff = tile0; tOff < tiles; tOff += perT)
                    {
                        ConvTrTiled(name, x, y, outLen, inLen, inCh, outCh, kernel, stride, bias,
                                    tOff, Math.Min(perT, tiles - tOff));
                        if (tOff + perT < tiles) yield return null;
                    }
                    yield break;
                }
                int total = outLen * outCh;
                int start = outRowStart * outCh;
                int slices = SlicesFor((long)(total - start) * (inCh / groups) * (kernel / stride), total - start);
                int per = (total - start + slices - 1) / slices;
                for (int off = start; off < total; off += per)
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
                int T, int inDim, int outDim, bool bias, int act = 0, int rowStart = 0)
            {
                int span = T - rowStart;
                int slices = SlicesFor((long)span * inDim * outDim, span);
                int rows = (span + slices - 1) / slices;
                for (int r0 = rowStart; r0 < T; r0 += rows)
                {
                    ComputeBuffer scales = w.Has(name + ".weight.scales") ? w.Get(name + ".weight.scales") : null;
                    int sub = Math.Min(rows, T - r0);
                    // #31-P: coalesced GEMM slice — a ragged tail tile recomputes <=7 rows also
                    // covered by the next slice with the SAME kernel -> identical values (#29 rule).
                    if (CoalEligible(inDim))
                    {
                        int kc = scales != null ? kLinearQ8Gemm : kLinearGemm;
                        cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                        cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                        cs.SetBuffer(kc, "X", x); cs.SetBuffer(kc, "W", w.Get(name + ".weight"));
                        cs.SetBuffer(kc, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                        if (scales != null) cs.SetBuffer(kc, "W_scales", scales);
                        cs.SetBuffer(kc, "Y", y);
                        cs.SetInt("elem_offset", r0);
                        cs.Dispatch(kc, (outDim + 7) / 8, (sub + 7) / 8, 1);
                        cs.SetInt("elem_offset", 0);
                        if (r0 + rows < T) yield return null;
                        continue;
                    }
                    int k = scales != null ? kLinearQ8 : kLinear;
                    cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                    cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0);
                    cs.SetBuffer(k, "X", x); cs.SetBuffer(k, "W", w.Get(name + ".weight"));
                    cs.SetBuffer(k, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                    if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                    cs.SetBuffer(k, "Y", y);
                    cs.SetInt("elem_offset", r0);
                    cs.Dispatch(k, 1, (sub + 7) / 8, (outDim + 31) / 32);
                    cs.SetInt("elem_offset", 0);
                    if (r0 + rows < T) yield return null;
                }
            }

            void LayerNorm(string name, ComputeBuffer x, ComputeBuffer y, int T, int dim, int rowStart = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", dim); cs.SetFloat("norm_eps", 1e-5f);
                cs.SetBuffer(kLN, "norm_input", x); cs.SetBuffer(kLN, "norm_output", y);
                cs.SetBuffer(kLN, "ln_gamma", w.Get(name + ".weight")); cs.SetBuffer(kLN, "ln_beta", w.Get(name + ".bias"));
                cs.SetInt("elem_offset", rowStart);
                cs.Dispatch(kLN, Div256(T - rowStart), 1, 1);
                if (rowStart != 0) cs.SetInt("elem_offset", 0);
            }

            void RoPE(ComputeBuffer buf, int T, int heads, int hd, int rowStart = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                cs.SetInt("pos_offset", 0); cs.SetFloat("rope_theta", Cfg.MIMI_TF_ROPE_THETA);
                cs.SetBuffer(kRope, "inout_buf", buf);
                cs.SetInt("elem_offset", rowStart * heads * (hd / 2));
                Dispatch1D(kRope, (T - rowStart) * heads * (hd / 2));
                if (rowStart != 0) cs.SetInt("elem_offset", 0);
            }

            // residual += sublayer * layer_scale[c]
            void ChannelScaleAdd(ComputeBuffer residBuf, ComputeBuffer sub, string lsName, int T, int dim, int rowStart = 0)
            {
                cs.SetInt("buffer_size", T * dim); cs.SetInt("norm_dim", dim);
                cs.SetBuffer(kChanAdd, "buf_a", residBuf); cs.SetBuffer(kChanAdd, "buf_b", sub);
                cs.SetBuffer(kChanAdd, "ch_scale", w.Get(lsName + ".scale"));
                cs.SetInt("elem_offset", rowStart * dim);
                Dispatch1D(kChanAdd, (T - rowStart) * dim);
                if (rowStart != 0) cs.SetInt("elem_offset", 0);
            }

            // SEANet causal resblock: x + conv2(elu(conv1(elu(x)))). conv1 k3 dim->hidden pad2, conv2 k1 hidden->dim pad0.
            // Uses scratch c,d; out written in place to `x` buffer's alias handled by caller (returns in `outBuf`).
            // #29: enumerator — the convs are MAC-sliced and each range is one tick for the caller.
            // rowStartOut (#30): first resblock OUTPUT row that must be exact. conv1 (k3, padLeft 2)
            // reads 2 rows back, and its tile floor another CONV_TB-1, so the staging Copy/ELU cover
            // from tileFloor(rowStartOut)-2; conv2 is k1 so its valid outputs read only rows >=
            // rowStartOut (all ELU'd). Rows below rowStartOut end up garbage — never read upstream.
            System.Collections.IEnumerator ResBlockTicks(string p, ComputeBuffer x, ComputeBuffer outBuf, int T, int dim, int rowStartOut = 0)
            {
                int hidden = dim / Cfg.MIMI_COMPRESS;
                int inFrom = Math.Max(0, (rowStartOut / CONV_TB) * CONV_TB - 2);
                // v = elu(x); conv1 -> c
                Copy(c, x, T * dim, inFrom * dim);
                Act(c, T * dim, 7, inFrom * dim);
                var e = ConvSliced(p + "/block/1/conv", c, d, T, T, dim, hidden, 3, 1, 1, 2, bias: true, outRowStart: rowStartOut);
                while (e.MoveNext()) yield return null;
                Act(d, T * hidden, 7, rowStartOut * hidden);
                e = ConvSliced(p + "/block/3/conv", d, c, T, T, hidden, dim, 1, 1, 1, 0, bias: true, outRowStart: rowStartOut);
                while (e.MoveNext()) yield return null;
                // outBuf = x + c
                if (outBuf != x) Copy(outBuf, x, T * dim, rowStartOut * dim);
                cs.SetInt("buffer_size", T * dim);
                cs.SetBuffer(kAdd, "buf_a", outBuf); cs.SetBuffer(kAdd, "buf_b", c);
                cs.SetInt("elem_offset", rowStartOut * dim);
                Dispatch1D(kAdd, (T - rowStartOut) * dim);   // SEANet stage2: 122880 elems/latent — the old 1-D dispatch was the long-reply crash
                if (rowStartOut != 0) cs.SetInt("elem_offset", 0);
            }

            // set by the issue chain: the buffer holding the final waveform + its sample count +
            // the first VALID sample row (#30 tail-restricted decode; 0 on full decodes)
            ComputeBuffer lastWavBuf;
            int lastWavLen;
            int lastWavStart;

            /// <summary>Decode Mimi latents [T,32] (row-major) -> wav float[T*1920].
            /// The Mimi decoder's input is the DENORMED latent (quantizer input); the dump's
            /// latents.npy already carries that. denorm is a flow→mimi BOUNDARY op (P4), NOT part
            /// of the Mimi decoder — pass embMean/embStd non-null only when feeding RAW flow
            /// latents (e2e); P1 passes null (latents are already denormed).
            /// SYNC form (probes/offline): drains the sliced issue chain — IDENTICAL dispatches in
            /// identical order — then a blocking readback. Parity-neutral vs the pre-split code.</summary>
            /// <summary>tailLatents (#30): number of NEW latents at the end of the block whose wav
            /// must be exact — callers of windowed decode keep only that tail. Values &lt;= 0 or
            /// &gt;= T mean "all" (the pre-#30 full decode, bit-identical). With a real tail the
            /// SEANet stages and per-layer transformer rows are restricted to the tail region plus
            /// exact causal receptive-field margins: kept samples stay BIT-EXACT (parity-gated),
            /// while ~60-75% of the window's dispatch work is skipped. Samples before the tail are
            /// garbage — do not read them.</summary>
            public float[] Decode(float[] latents, int T, float[] embMean = null, float[] embStd = null, int tailLatents = -1)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var e = DecodeIssueYielding(Denorm(latents, T, embMean, embStd), T, tailLatents);
                while (e.MoveNext()) { }
                float[] wav = new float[lastWavLen];
                lastWavBuf.GetData(wav, lastWavStart, lastWavStart, lastWavLen - lastWavStart);
                sw.Stop();
                DecodeMs = (float)sw.Elapsed.TotalMilliseconds;
                return wav;
            }

            /// <summary>#31-R3 (mimi/AR overlap): issue a (windowed, tail-restricted) decode with NO
            /// readback and GPU-copy the kept tail samples into `dst` at `dstSampleOffset`
            /// (CopySlice — a pure copy, so the harvested values are bit-identical to what Decode()
            /// would have read back). The caller interleaves these windows between AR blocks (the
            /// window's fat conv kernels fill the AR chain's inter-dispatch dependency bubbles) and
            /// owns the single deferred readback of the assembled buffer. Scratch reuse across
            /// windows is safe: the tail-restriction bookkeeping never reads garbage-permitted
            /// regions (probe gates 2/B4b), so a window's output is independent of buffer state.</summary>
            public void DecodeIssueTo(ComputeBuffer dst, int dstSampleOffset, float[] latents, int T,
                                      float[] embMean, float[] embStd, int tailLatents)
            {
                var e = DecodeIssueYielding(Denorm(latents, T, embMean, embStd), T, tailLatents);
                while (e.MoveNext()) { }
                int tailN = lastWavLen - lastWavStart;
                cs.SetInt("buffer_size", tailN);
                cs.SetInt("copy_src_offset", lastWavStart);
                cs.SetInt("copy_dst_offset", dstSampleOffset);
                cs.SetBuffer(kSlice, "buf_a", dst);
                cs.SetBuffer(kSlice, "buf_b", lastWavBuf);
                cs.SetInt("elem_offset", 0);
                Dispatch1D(kSlice, tailN);
            }

            /// <summary>STREAMING form (#29): the decode chain is SLICED — one GPU-heavy group per
            /// MoveNext (callers surface each yield as a frame boundary), and the wav readback is
            /// ASYNC (delivered 1-3 frames later; the voice ring's prebuffer absorbs it). Same
            /// dispatches as Decode. wavOut must be [T*1920].</summary>
            public System.Collections.IEnumerator DecodeYielding(float[] latents, int T,
                float[] embMean, float[] embStd, float[] wavOut, bool async, int tailLatents = -1)
            {
                var e = DecodeIssueYielding(Denorm(latents, T, embMean, embStd), T, tailLatents);
                while (e.MoveNext()) yield return null;
                if (async && SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(
                        lastWavBuf, (lastWavLen - lastWavStart) * 4, lastWavStart * 4);
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
                        var got = req.GetData<float>();
                        Unity.Collections.NativeArray<float>.Copy(got, 0, wavOut, lastWavStart, lastWavLen - lastWavStart);
                        yield break;
                    }
                }
                lastWavBuf.GetData(wavOut, lastWavStart, lastWavStart, lastWavLen - lastWavStart);   // unsupported/error fallback: sync
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
            System.Collections.IEnumerator DecodeIssueYielding(float[] input, int T, int tailLatents = -1)
            {
                Ensure(T);
                int F = T * Cfg.MIMI_STEPS_PER_LATENT;   // transformer frames (16T)
                a.SetData(input, 0, 0, T * Cfg.LDIM);
                if (StageProfile != null) profSw.Restart();

                // ---- #30 tail-restricted window decode: backward receptive-field bookkeeping.
                // Only wav rows >= wavStart must be exact; walking the causal graph backwards gives
                // the first output row each stage must compute (everything earlier may stay garbage,
                // it is never read). tail == T -> every start is 0 = the pre-#30 full decode.
                int tail = (tailLatents <= 0 || tailLatents > T) ? T : tailLatents;
                int wavStart = (T - tail) * Cfg.SAMPLES_PER_LATENT;
                int[] rbStart = new int[3], ctrStart = new int[3];
                int need = Math.Max(0, wavStart - 2);              // final conv (k3, pad 2) input rows
                for (int bs = 2; bs >= 0; bs--)
                {
                    rbStart[bs] = need;                            // resblock output rows needed
                    ctrStart[bs] = Math.Max(0, need - 2);          // resblock conv1 (k3, pad 2) margin
                    int rr = Cfg.MIMI_RATIOS[bs];
                    need = Math.Max(0, (ctrStart[bs] - (2 * rr - 1)) / rr);   // convtr k=2r min input row
                }
                int conv0Start = need;                             // conv0 output rows needed (16T rate)
                int xfNeed = Math.Max(0, conv0Start - 6);          // conv0 (k7, pad 6) input = transformer out
                // transformer: layer-2 queries feed conv0; each layer's K/V (and the previous
                // layer's output) reach MIMI_TF_CONTEXT rows further back.
                int q2Start = xfNeed, k2Start = Math.Max(0, q2Start - Cfg.MIMI_TF_CONTEXT);
                int q1Start = k2Start, k1Start = Math.Max(0, q1Start - Cfg.MIMI_TF_CONTEXT);
                lastWavStart = wavStart;

                // ---- quantizer Conv1d k1 32->512 -> b [T,512], then grouped ConvTr x16 -> resid
                Conv("mimi/quantizer/output_proj", a, b, T, T, Cfg.LDIM, Cfg.QUANT_OUT, 1, 1, 1, 0, bias: false);
                DebugTap?.Invoke("quant_out", b, T * Cfg.QUANT_OUT);
                ConvTr("mimi/upsample/convtr/convtr", b, resid, F, T, Cfg.QUANT_OUT, Cfg.MIMI_SEANET_DIM,
                       Cfg.UPSAMPLE_KERNEL, Cfg.UPSAMPLE_STRIDE, Cfg.UPSAMPLE_GROUPS, bias: false);
                DebugTap?.Invoke("mimi_upsampled", resid, F * Cfg.MIMI_SEANET_DIM);
                Mark("quant+upsample", resid);
                yield return null;

                // ---- decoder_transformer: 2 layers, d512/8h, LayerNorm + RoPE + causal + layer_scale
                int dim = Cfg.MIMI_TF_DIM, heads = Cfg.MIMI_TF_HEADS, hd = Cfg.MIMI_TF_HEAD_DIM;
                float attScale = 1f / Mathf.Sqrt(hd);
                for (int li = 0; li < Cfg.MIMI_TF_LAYERS; li++)
                {
                    string lp = $"mimi/decoder_transformer/transformer/layers/{li}";
                    int qRow = li == 0 ? q1Start : q2Start;   // rows whose layer OUTPUT is needed
                    int kRow = li == 0 ? k1Start : k2Start;   // rows whose K/V (LN + in_proj) is needed
                    // -- self-attn block  (#29: the whole layer used to be ONE ~2.5 GMAC tick — the
                    // fattest slice in the pipeline. Now attn and each FFN matmul are their own
                    // MAC-bounded ticks.)
                    LayerNorm(lp + "/norm1", resid, a, F, dim, kRow);
                    var lr = LinearRows(lp + "/self_attn/in_proj", a, qkvBuf, F, dim, 3 * dim, bias: false, rowStart: kRow);   // [F,1536]
                    while (lr.MoveNext()) yield return null;
                    SliceCols(qkvBuf, qBuf, F, 3 * dim, dim, 0, qRow * dim);
                    SliceCols(qkvBuf, kBuf, F, 3 * dim, dim, dim, kRow * dim);
                    SliceCols(qkvBuf, vBuf, F, 3 * dim, dim, 2 * dim, kRow * dim);
                    RoPE(qBuf, F, heads, hd, qRow);
                    RoPE(kBuf, F, heads, hd, kRow);
                    yield return null;
                    int ka = ForceLegacyKernels ? kAttnLegacy : kAttn;   // #30 parity switch
                    cs.SetInt("seq_len", F); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetInt("rope_on", 1); cs.SetFloat("scale", attScale);
                    cs.SetInt("attn_context", Cfg.MIMI_TF_CONTEXT);   // 250 — sliding causal window (Moshi/Mimi)
                    cs.SetBuffer(ka, "Q", qBuf); cs.SetBuffer(ka, "K", kBuf); cs.SetBuffer(ka, "V", vBuf);
                    cs.SetBuffer(ka, "AttendedValues", attnScratch);
                    if (ForceLegacyKernels)                   // legacy kernel has no query offset
                        cs.Dispatch(ka, F, heads, 1);
                    else
                    {
                        cs.SetInt("elem_offset", qRow);
                        cs.Dispatch(ka, F - qRow, heads, 1);
                        if (qRow != 0) cs.SetInt("elem_offset", 0);
                    }
                    Linear(lp + "/self_attn/out_proj", attnScratch, a, F, dim, dim, bias: false, rowStart: qRow);
                    ChannelScaleAdd(resid, a, lp + "/layer_scale_1", F, dim, qRow);
                    Mark($"xf{li}_attn", resid);
                    yield return null;
                    // -- ffn block
                    LayerNorm(lp + "/norm2", resid, a, F, dim, qRow);
                    lr = LinearRows(lp + "/linear1", a, b, F, dim, Cfg.MIMI_TF_FFN, bias: false, act: 2, rowStart: qRow);  // GELU
                    while (lr.MoveNext()) yield return null;
                    yield return null;
                    lr = LinearRows(lp + "/linear2", b, a, F, Cfg.MIMI_TF_FFN, dim, bias: false, rowStart: qRow);
                    while (lr.MoveNext()) yield return null;
                    ChannelScaleAdd(resid, a, lp + "/layer_scale_2", F, dim, qRow);
                    Mark($"xf{li}_ffn", resid);
                    yield return null;
                }
                DebugTap?.Invoke("mimi_xf_out", resid, F * dim);

                // ---- SEANet decoder: conv0 k7 512->512, then 3 x [ELU, ConvTr, ResBlock], then ELU, conv k3 ->1
                // Every stage computes only from its #30 start row; the ELUs cover down to the
                // earliest row any dispatched (tile-floored) conv output reads.
                var e0 = ConvSliced("mimi/decoder/model/0/conv", resid, a, F, F, 512, 512, 7, 1, 1, 6, bias: true,
                                    outRowStart: conv0Start);
                while (e0.MoveNext()) yield return null;
                DebugTap?.Invoke("seanet_conv0", a, F * 512);
                Mark("seanet_conv0", a);
                yield return null;
                ComputeBuffer cur = a;                 // [len, ch]
                int len = F, ch = 512;
                int[] ratios = Cfg.MIMI_RATIOS;
                int[] modelIdx = { 2, 5, 8 };          // ConvTr module indices; resblock at +1
                for (int s = 0; s < 3; s++)
                {
                    int r = ratios[s], outCh = ch / 2, outLen = len * r;
                    int eluFrom = Math.Max(0, ((ctrStart[s] / CONV_TB) * CONV_TB - (2 * r - 1)) / r);
                    Act(cur, len * ch, 7, eluFrom * ch);   // ELU before convtr (model[idx-1])
                    ComputeBuffer nxt = (cur == a) ? b : a;
                    var et = ConvTrSliced($"mimi/decoder/model/{modelIdx[s]}/convtr", cur, nxt, outLen, len, ch, outCh,
                                          r * 2, r, 1, bias: true, outRowStart: ctrStart[s]);
                    while (et.MoveNext()) yield return null;
                    len = outLen; ch = outCh; cur = nxt;
                    Mark($"s{s}_convtr", cur);
                    yield return null;                 // convtr and resblock are the heavy tail — split them
                    // resblock (model[idx+1]); in place via c/d scratch, out to the other of a/b
                    ComputeBuffer rbOut = (cur == a) ? b : a;
                    var er = ResBlockTicks($"mimi/decoder/model/{modelIdx[s] + 1}", cur, rbOut, len, ch, rbStart[s]);
                    while (er.MoveNext()) yield return null;
                    cur = rbOut;
                    DebugTap?.Invoke($"seanet_stage{s}", cur, len * ch);
                    Mark($"s{s}_resblock", cur);
                    yield return null;
                }
                int finalEluFrom = Math.Max(0, (wavStart / CONV_TB) * CONV_TB - 2);
                Act(cur, len * ch, 7, finalEluFrom * ch);   // final ELU (model[10])
                ComputeBuffer wavBuf = (cur == a) ? b : a;
                var ef = ConvSliced("mimi/decoder/model/11/conv", cur, wavBuf, len, len, 64, 1, 3, 1, 1, 2, bias: true,
                                    outRowStart: wavStart);
                while (ef.MoveNext()) yield return null;
                Mark("final_conv", wavBuf);
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
