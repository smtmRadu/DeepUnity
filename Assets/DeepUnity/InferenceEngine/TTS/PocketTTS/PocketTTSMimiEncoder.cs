using System;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        using Cfg = PocketTTSConfig;

        // Mimi ENCODER — reference wav [S] -> latents [T,32] (P8 runtime voice cloning). Exact
        // inverse of the decoder (PocketTTSMimi), reusing the same kernels:
        //   pad_for_conv1d(x, 1920) -> SEANet encoder (conv0 1->64 k7; 3x[resblock, ELU, strided
        //   conv]; ELU; conv 512->512 k3) at 200 Hz -> 2L encoder_transformer (LayerNorm+RoPE+
        //   causal-250+layer_scale) -> ConvDownsample1d (512->32 k32 s16, REPLICATE left-pad) at
        //   12.5 Hz -> [T,32]. Then PocketTTS applies speaker_proj [1024,32] -> audio_prompt [T,1024].
        // StreamingConv1d(model_state=None) still left-pads each conv by (kernel - stride) — constant
        // zeros (encoder) / replicate first-sample (downsample) — reproduced via Conv1D pad_left +
        // pad_replicate. Verified in WSL: this exact scheme matches encode_to_latent bit-for-bit.
        //
        // [T,C] layout throughout (transposed vs torch [C,T]) — same as the decoder port.
        public class PocketTTSMimiEncoder : IDisposable
        {
            readonly ComputeShader cs;
            readonly PocketTTSWeights w;
            int kCopy, kSliceCols, kZero, kAdd, kChanAdd, kAct, kLinear, kLinearQ8, kConv, kLN, kRope, kAttn;

            ComputeBuffer a, b, c, d, resid, attnScratch, qkvBuf, qBuf, kBuf, vBuf;
            int curCap;

            public PocketTTSMimiEncoder(PocketTTSWeights weights)
            {
                w = weights;
                cs = DeepUnityMeta.PocketTTSCS;
                kCopy = cs.FindKernel("CopyBuffer");
                kSliceCols = cs.FindKernel("SliceCols");
                kZero = cs.FindKernel("ZeroBuffer");
                kAdd = cs.FindKernel("AddResidual");
                kChanAdd = cs.FindKernel("ChannelScaleAdd");
                kAct = cs.FindKernel("Activate");
                kLinear = cs.FindKernel("LinearBias");
                kLinearQ8 = cs.FindKernel("LinearBiasQ8");
                kConv = cs.FindKernel("Conv1D");
                kLN = cs.FindKernel("LayerNormT");
                kRope = cs.FindKernel("ApplyRoPE");
                kAttn = cs.FindKernel("CausalAttention");
            }

            static int Div256(int n) => (n + 255) / 256;
            static void Grow(ref ComputeBuffer buf, int n) { if (buf != null && buf.count >= n) return; buf?.Release(); buf = new ComputeBuffer(Math.Max(n, 1), 4, ComputeBufferType.Structured); }

            // 65535-group D3D11 cap: encoder conv0 runs 64ch x S samples (a 30 s clip = 46M
            // elements) — same Y-spill guard as the decoder (kernels reconstruct via LinearId).
            void Dispatch1D(int kernel, int elements)
            {
                int g = Div256(elements);
                if (g <= 65535) { cs.Dispatch(kernel, Math.Max(g, 1), 1, 1); return; }
                cs.Dispatch(kernel, 65535, (g + 65534) / 65535, 1);
            }

            /// <summary>True once the encoder weights (mimi/encoder* + mimi/downsample) are resident.
            /// The runtime dir may omit them (baked-voice-only build) — CloneVoice checks this.</summary>
            public bool HasEncoderWeights => w.Has("mimi/encoder/model/0/conv.weight")
                                          && w.Has("mimi/downsample/conv/conv.weight");

            // ---- op helpers (mirror PocketTTSMimi) ----
            void Copy(ComputeBuffer dst, ComputeBuffer src, int n)
            { cs.SetInt("buffer_size", n); cs.SetBuffer(kCopy, "buf_a", dst); cs.SetBuffer(kCopy, "buf_b", src); Dispatch1D(kCopy, n); }

            void Act(ComputeBuffer buf, int n, int act)
            { cs.SetInt("buffer_size", n); cs.SetInt("activation_type", act); cs.SetFloat("leaky_slope", 0.01f); cs.SetBuffer(kAct, "inout_buf", buf); Dispatch1D(kAct, n); }

            void SliceCols(ComputeBuffer src, ComputeBuffer dst, int T, int inDim, int outDim, int colOff)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim); cs.SetInt("copy_src_offset", colOff);
                cs.SetBuffer(kSliceCols, "X", src); cs.SetBuffer(kSliceCols, "Y", dst);
                Dispatch1D(kSliceCols, T * outDim);
            }

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

            void Conv(string name, ComputeBuffer x, ComputeBuffer y, int outLen, int inLen,
                      int inCh, int outCh, int kernel, int stride, int padLeft, bool bias, int act = 0, bool padRep = false)
            {
                cs.SetInt("seq_len", outLen); cs.SetInt("in_len", inLen); cs.SetInt("in_dim", inCh);
                cs.SetInt("out_dim", outCh); cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", stride);
                cs.SetInt("conv_dilation", 1); cs.SetInt("pad_left", padLeft); cs.SetInt("pad_replicate", padRep ? 1 : 0);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", bias ? 1 : 0); cs.SetFloat("leaky_slope", 0.01f);
                cs.SetBuffer(kConv, "X", x); cs.SetBuffer(kConv, "W", w.Get(name + ".weight"));
                cs.SetBuffer(kConv, "W_bias", bias ? w.Get(name + ".bias") : w.Get(name + ".weight"));
                cs.SetBuffer(kConv, "Y", y);
                Dispatch1D(kConv, outLen * outCh);
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
                cs.SetBuffer(kRope, "inout_buf", buf); cs.Dispatch(kRope, Div256(T * heads * (hd / 2)), 1, 1);
            }

            void ChannelScaleAdd(ComputeBuffer residBuf, ComputeBuffer sub, string lsName, int T, int dim)
            {
                cs.SetInt("buffer_size", T * dim); cs.SetInt("norm_dim", dim);
                cs.SetBuffer(kChanAdd, "buf_a", residBuf); cs.SetBuffer(kChanAdd, "buf_b", sub);
                cs.SetBuffer(kChanAdd, "ch_scale", w.Get(lsName + ".scale"));
                Dispatch1D(kChanAdd, T * dim);
            }

            // SEANet causal encoder resblock: x + conv2(elu(conv1(elu(x)))). conv1 k3 dim->hidden
            // pad_left=2, conv2 k1 hidden->dim pad0. hidden = dim/compress. Same shape as the decoder's.
            void ResBlock(string p, ComputeBuffer x, ComputeBuffer outBuf, int T, int dim)
            {
                int hidden = dim / Cfg.MIMI_COMPRESS;
                Copy(c, x, T * dim);
                Act(c, T * dim, 7);                                             // ELU
                Conv(p + "/block/1/conv", c, d, T, T, dim, hidden, 3, 1, 2, bias: true);
                Act(d, T * hidden, 7);                                          // ELU
                Conv(p + "/block/3/conv", d, c, T, T, hidden, dim, 1, 1, 0, bias: true);
                if (outBuf != x) Copy(outBuf, x, T * dim);
                cs.SetInt("buffer_size", T * dim);
                cs.SetBuffer(kAdd, "buf_a", outBuf); cs.SetBuffer(kAdd, "buf_b", c);
                Dispatch1D(kAdd, T * dim);   // stage-0 resblock: 64ch x wav samples — needs the Y-spill
            }

            void Ensure(int wavLen, int encLen)
            {
                // peak buffers: SEANet is widest early (64ch @ wavLen after conv0) shrinking in T as
                // channels grow; encLen*512 (transformer) is small. Allocate for the max footprint.
                int peak = Math.Max(wavLen * 64, encLen * 512);
                if (peak <= curCap)
                {
                    Grow(ref resid, encLen * 512); Grow(ref attnScratch, encLen * 512);
                    Grow(ref qkvBuf, encLen * 1536); Grow(ref qBuf, encLen * 512);
                    Grow(ref kBuf, encLen * 512); Grow(ref vBuf, encLen * 512);
                    return;
                }
                curCap = peak;
                Grow(ref a, peak); Grow(ref b, peak); Grow(ref c, peak); Grow(ref d, peak);
                Grow(ref resid, encLen * 512); Grow(ref attnScratch, encLen * 512);
                Grow(ref qkvBuf, encLen * 1536); Grow(ref qBuf, encLen * 512);
                Grow(ref kBuf, encLen * 512); Grow(ref vBuf, encLen * 512);
            }

            /// <summary>Encode a 24 kHz mono wav [S] -> unquantized latents [T,32] (T = ceil(S/1920)),
            /// matching mimi.encode_to_latent. Caller applies speaker_proj to get audio_prompt.</summary>
            public float[] Encode(float[] wav, out int T)
            {
                // pad_for_conv1d at the top level: pad to a multiple of frame_size (1920) at the END.
                int fs = Cfg.SAMPLES_PER_LATENT;                 // 1920
                int padded = ((wav.Length + fs - 1) / fs) * fs;
                T = padded / fs;                                 // encoder latent frames (12.5 Hz)
                int encLen = padded / (fs / Cfg.MIMI_STEPS_PER_LATENT);   // 200 Hz frames = padded/120
                Ensure(padded, encLen);

                // upload wav [padded,1] (zero-padded tail)
                var input = new float[padded];
                Array.Copy(wav, 0, input, 0, wav.Length);
                a.SetData(input, 0, 0, padded);

                // ---- SEANet encoder (each strided conv left-pads k-s zeros; out = inLen/stride) ----
                int len = padded, ch = 1;
                // [0] conv 1->64 k7 s1 pad_left=6
                Conv("mimi/encoder/model/0/conv", a, b, len, len, 1, 64, 7, 1, 6, bias: true);
                ch = 64;
                // 3 stages: [resblock, ELU, strided conv]. ratios (encoder order) = [4,5,6].
                int[] stageConv = { 3, 6, 9 };
                int[] resIdx = { 1, 4, 7 };
                int[] convK = { 8, 10, 12 };
                int[] convS = { 4, 5, 6 };
                ComputeBuffer cur = b;                            // [len, ch]
                for (int s = 0; s < 3; s++)
                {
                    // resblock (model[resIdx]) in place -> other buffer
                    ComputeBuffer rbOut = (cur == a) ? b : a;
                    ResBlock($"mimi/encoder/model/{resIdx[s]}", cur, rbOut, len, ch);
                    cur = rbOut;
                    Act(cur, len * ch, 7);                        // ELU (model[resIdx+1])
                    // strided conv ch->2ch, out = len/stride, pad_left = k - stride
                    int outCh = ch * 2, k = convK[s], st = convS[s], outLen = len / st;
                    ComputeBuffer nxt = (cur == a) ? b : a;
                    Conv($"mimi/encoder/model/{stageConv[s]}/conv", cur, nxt, outLen, len, ch, outCh, k, st, k - st, bias: true);
                    len = outLen; ch = outCh; cur = nxt;
                }
                // [10] ELU, [11] conv 512->512 k3 s1 pad_left=2
                Act(cur, len * ch, 7);
                ComputeBuffer c11 = (cur == a) ? b : a;
                Conv("mimi/encoder/model/11/conv", cur, c11, len, len, 512, 512, 3, 1, 2, bias: true);
                cur = c11;   // [encLen, 512]

                // ---- encoder_transformer: 2L d512/8h, LayerNorm + RoPE + causal-250 + layer_scale ----
                Copy(resid, cur, encLen * 512);
                int dim = Cfg.MIMI_TF_DIM, heads = Cfg.MIMI_TF_HEADS, hd = Cfg.MIMI_TF_HEAD_DIM;
                float attScale = 1f / Mathf.Sqrt(hd);
                for (int li = 0; li < Cfg.MIMI_TF_LAYERS; li++)
                {
                    string lp = $"mimi/encoder_transformer/transformer/layers/{li}";
                    LayerNorm(lp + "/norm1", resid, a, encLen, dim);
                    Linear(lp + "/self_attn/in_proj", a, qkvBuf, encLen, dim, 3 * dim, bias: false);
                    SliceCols(qkvBuf, qBuf, encLen, 3 * dim, dim, 0);
                    SliceCols(qkvBuf, kBuf, encLen, 3 * dim, dim, dim);
                    SliceCols(qkvBuf, vBuf, encLen, 3 * dim, dim, 2 * dim);
                    RoPE(qBuf, encLen, heads, hd);
                    RoPE(kBuf, encLen, heads, hd);
                    cs.SetInt("seq_len", encLen); cs.SetInt("num_heads", heads); cs.SetInt("head_dim", hd);
                    cs.SetInt("rope_on", 1); cs.SetFloat("scale", attScale); cs.SetInt("attn_context", Cfg.MIMI_TF_CONTEXT);
                    cs.SetBuffer(kAttn, "Q", qBuf); cs.SetBuffer(kAttn, "K", kBuf); cs.SetBuffer(kAttn, "V", vBuf);
                    cs.SetBuffer(kAttn, "AttendedValues", attnScratch);
                    cs.Dispatch(kAttn, encLen, heads, 1);
                    Linear(lp + "/self_attn/out_proj", attnScratch, a, encLen, dim, dim, bias: false);
                    ChannelScaleAdd(resid, a, lp + "/layer_scale_1", encLen, dim);
                    LayerNorm(lp + "/norm2", resid, a, encLen, dim);
                    Linear(lp + "/linear1", a, b, encLen, dim, Cfg.MIMI_TF_FFN, bias: false, act: 2);   // GELU
                    Linear(lp + "/linear2", b, a, encLen, Cfg.MIMI_TF_FFN, dim, bias: false);
                    ChannelScaleAdd(resid, a, lp + "/layer_scale_2", encLen, dim);
                }
                // resid = encoder_transformer output [encLen, 512]

                // ---- downsample: Conv1d 512->32 k32 s16, REPLICATE left-pad (16) -> [T,32] ----
                Conv("mimi/downsample/conv/conv", resid, a, T, encLen, 512, Cfg.LDIM, 32, 16, 16, bias: false, padRep: true);
                var latents = new float[T * Cfg.LDIM];
                a.GetData(latents, 0, 0, T * Cfg.LDIM);
                return latents;   // [T,32] unquantized (Mimi encode_to_latent output)
            }

            public void Dispose()
            {
                a?.Release(); b?.Release(); c?.Release(); d?.Release(); resid?.Release();
                attnScratch?.Release(); qkvBuf?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release();
            }
        }
    }
}
