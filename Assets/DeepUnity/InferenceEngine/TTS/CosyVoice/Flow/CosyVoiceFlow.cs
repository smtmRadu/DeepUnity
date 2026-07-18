using System;
using System.Collections;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // CausalMaskedDiffWithDiT — CosyVoice3's speech-token(25Hz) -> mel(80 @ 50Hz) flow.
        // Port spec: TTS/CosyVoice/SPEC.md §2, verified against flow/{flow,flow_matching}.py and
        // flow/DiT/{dit,modules}.py:
        //   tok = prompt_tokens ++ gen_tokens -> Embedding(6561,80)
        //   PreLookaheadLayer: conv1 k4 RIGHT-pad3 (80->1024, leaky 0.01) -> conv2 k3 LEFT-pad2
        //                      (1024->80) -> +residual
        //   repeat_interleave x2 (25Hz -> 50Hz), spk = Linear(192->80)(L2norm(xvector))
        //   cond = [prompt_feat | zeros], x0 = FIXED seed-0 noise (exported flow/rand_noise)
        //   10 Euler steps, cosine t (t=1-cos(t·π/2)), CFG: dxdt = 1.7·cond_pass - 0.7·uncond_pass
        //   (uncond pass zeroes mu/spk/cond, keeps x; mask all-ones at batch 1)
        // DiT estimator (22L x 1024, 16h x 64d, ff 2048 GELU-tanh):
        //   input_embed cat[x,cond,mu,spk]=320 -> Linear -> += CausalConvPos (2x grouped k31 g16
        //   LEFT-pad30 + Mish); t: sinus256(scale1000) -> MLP; AdaLN-Zero per block
        //   (shift,scale,gate)x2 from Linear(SiLU(t))->6144; final norm_out order = (SCALE, shift);
        //   RoPE = x_transformers quirk: flat pre-head-split -> ONLY head 0 rotates (RopeQK kernel).
        //   Offline attention = full bidirectional; the A5 streaming path adds the 50-frame chunk mask.
        //
        // A6-max (ProbeLogs/a6max_research.md):
        //   * Batch-2 CFG — cond+uncond run as ONE stacked [2M,·] estimator pass per Euler step
        //     (matches the reference's batched solve_euler). Per-row math and accumulation order
        //     are unchanged, so this is BIT-EXACT vs the old two sequential passes.
        //   * AdaLN mods precomputed once per instance — the cosine t-schedule is fixed, so the
        //     NT x 22 block mods + NT final-norm mods are constants. Bit-exact.
        //   * Single-pass streaming (SinglePassStreaming, default ON) — each streaming chunk
        //     solves ONLY the new 50-frame blocks once. Block-causal invariance: with the chunk
        //     mask a frozen row's whole Euler trajectory is unchanged when the prefix grows (fixed
        //     noise slice, fixed t schedule, causal convs, no right context), so its per-(step,
        //     layer) K/V are cached (fp16) and never recomputed. Cross-chunk state = the K/V
        //     cache + per-step x tails (the conv-pos left apron) + frozen x rows in xBuf. The
        //     pre-lookahead boundary band never enters the frozen set: non-finalize chunks
        //     exclude the last 3 tokens from hRows entirely, and the hop schedule keeps M a
        //     multiple of 50, so every solved row of a non-finalize chunk is final.
        //     Only numeric delta vs the full re-solve: fp16 K/V storage (attention math fp32).
        public class CosyVoiceFlow : IDisposable
        {
            const int MEL = CosyVoiceConfig.MEL_DIM;
            const int DIM = CosyVoiceConfig.DIT_DIM;      // 1024
            const int FF = CosyVoiceConfig.DIT_FF;        // 2048
            const int DEPTH = CosyVoiceConfig.DIT_DEPTH;  // 22
            // conv-pos left apron: two causal k31 convs -> conv2 output row t needs conv1 rows
            // [t-30,t], which need proj rows [t-60,t]. The apron rows' proj input is rebuilt from
            // the cached per-step x tail (x is the only step-dependent input lane).
            const int APRON = 2 * (CosyVoiceConfig.DIT_CONVPOS_KERNEL - 1);   // 60
            const string EST = "flow/decoder.estimator.";

            readonly ComputeShader cs;
            readonly CosyVoiceWeights weights;

            int kTokenEmbed, kLinear, kLinearT, kLinearQ8, kLinearTQ8, kConv, kConvG, kAdaLN, kGateAdd, kRope, kBidir, kTimeEmb,
                kRepeat, kPack, kPackB, kZero, kCopy, kCopySlice, kAdd, kActivate, kEulerCfg,
                kBidirKV, kWriteKV, kBidirQT, kBidirKVQT, kLinearB2, kLinearB2Q8,
                kStats, kDitQkv, kDitQkvQ8, kDitLin, kDitLinQ8, kRopePair, kPackEst;

            /// <summary>A6-max Phase 3, Q-tiled attention (default ON): 8 queries/group share
            /// each staged 64-key K tile and the V accumulation runs on ALL 256 lanes — the
            /// Phase-1 kernel idled 75% of lanes, paid the barrier tree per query, and re-read
            /// K/V per query. Only the online-softmax tiling changes (64-key tiles vs 256) ->
            /// fp32-rounding-level deltas, corr gates unaffected. false = Phase-1 kernel.</summary>
            public static bool FastAttention = true;
            /// <summary>A6-max Phase 3, register-blocked GEMM (default ON): 4 tokens x 2 outputs
            /// per thread (~3 B of groupshared traffic per FMA vs ~6 in LinearTileBias, which
            /// was shared-bandwidth-bound). Same per-(row,out) k-order -> BIT-EXACT.
            /// false = Phase-1 kernel. Separate from FastAttention so probes can bisect.</summary>
            public static bool FastGemm = true;
            /// <summary>cosyvoice-deepopt lever 2 (DEEPOPT.md §4.2, default ON): the DiT block
            /// chain runs the #31 GemmCoal fusion set — AdaLNStats writes only (mean, rstd)
            /// per row and the modulate expression moves into the GEMM X staging (GEMM inputs
            /// bit-identical); DitQKVCoal fuses to_q/to_k/to_v (+biases) in one dispatch;
            /// DitLinearCoal fuses the GateAdd epilogues into to_out/FF2 and the modulate +
            /// GELU into FF1/proj_out; RopeQKPair ropes q and k together; PackEstIn replaces
            /// the per-step Zero+Pack chain (bit-exact data movement). 13/14 -> 8/9 dispatches
            /// per block, 299 -> 184 per offline Euler step, and the 4x-per-block [2M,1024]
            /// AdaLN/GateAdd DRAM round-trips disappear. GEMM sum order differs from
            /// LinearTileBias2 -> TOLERANCE-equal, not bit-equal (mel A/B gate: maxAbs <=
            /// 5e-3, corr >= 0.9999 — CosyVoiceFastKernelsProbe). Streaming semantics (hop
            /// schedule, K/V cache, x-tails, chunk masks, EulerCfg offsets) are untouched.
            /// false = the legacy A2/A6-validated per-op path.</summary>
            public static bool FastDit31 = true;

            ComputeBuffer tokIdsBuf, embBuf, plaBuf, hBuf, muBuf, condBuf, xBuf;
            ComputeBuffer spkInBuf, spkBuf, tFreqBuf, tMidBuf, tEmbBuf, tSiluBuf, modTmpBuf, modFBuf;
            ComputeBuffer estInBuf, eA, eB, eS, qBuf, kBuf, vBuf, attnBuf, ffBuf, dxdtA, dxdtB;
            ComputeBuffer statsBuf;                       // FastDit31: per-row (mean, rstd) pairs
            ComputeBuffer modAllSteps, modFSteps;         // precomputed AdaLN mods, all Euler steps
            int modsNT = -1;
            int curTok, curM;

            // ---- single-pass streaming state (survives across chunks of one utterance) ---------
            /// <summary>Streaming chunks solve only the NEW 50-frame blocks against a per-(step,
            /// layer) fp16 K/V cache instead of re-solving the growing prefix (~2.6x less flow +
            /// no O(T²) growth). false = legacy full re-solve per chunk (bit-exact-to-A5-baseline
            /// escape hatch). Offline synthesis is unaffected either way.</summary>
            public bool SinglePassStreaming = true;
            /// <summary>K/V cache cap in mel frames (VRAM ~1.8 MB/frame across 10x22 buffers).
            /// Utterances that outgrow it fall back to the legacy re-solve (correct, just slower).</summary>
            public int MaxStreamKvFrames = CosyVoiceConfig.FLOW_STREAM_KV_MAX_FRAMES;
            /// <summary>Euler step override (0 = CFM_TIMESTEPS). Fewer steps intentionally diverge
            /// from the 10-step reference — perceptual QA territory, keep 0 for parity.</summary>
            public int TimestepsOverride = 0;
            /// <summary>When true, a 4-byte readback after the Euler loop measures the true GPU
            /// tail (GpuWaitMs) so flow cost stops being misattributed to the vocoder (§2.2 of the
            /// research report). Adds a fence — leave off in production.</summary>
            public static bool ProfileGpuFence = false;
            public float GpuWaitMs { get; private set; }

            int streamF;              // frozen mel rows so far (multiple of 50)
            bool streamFallback;      // utterance outgrew the cache -> legacy re-solve
            ComputeBuffer[] kvCache;  // [NT*DEPTH] fp16-packed, 4 planes (half x K/V) x cap rows
            int kvCacheNT = -1, kvCacheCap;
            ComputeBuffer xTailA, xTailB;   // [NT, APRON, 80] per-step x_s of rows [F-APRON, F)

            // default voice conds (baked by dump_reference.py / make_voice.py)
            readonly int[] promptTokens;
            readonly float[] promptFeat;      // [promptMel, 80] time-major
            readonly int promptMel;
            readonly float[] spkEmbedding;    // [192], L2-normalized here
            readonly float[] randNoise;       // [80, 15000] channel-major (fixed seed-0)

            /// <summary>Per-stage tap (name, buffer, elemCount) for the parity probe.</summary>
            public Action<string, ComputeBuffer, int> DebugTap;
            public float FlowMs { get; private set; }
            /// <summary>CPU dispatch-issue time of the Euler loop (no GPU syncs) — perf triage.</summary>
            public float IssueMs { get; private set; }

            // Per-synthesis cached weight handles (re-resolved each call — cheap, and safe across
            // Defetch/reload cycles which recreate the underlying ComputeBuffers).
            ComputeBuffer[] wMod, bMod, wQ, bQ, wK, bK, wV, bV, wO, bO, wF1, bF1, wF2, bF2;
            ComputeBuffer[] sMod, sQ, sK, sV, sO, sF1, sF2;   // q8 per-row scales (null = fp16)
            ComputeBuffer wProj, bProj, wCp1, bCp1, wCp2, bCp2, ropeFreq;
            ComputeBuffer wPo, bPo, sPo;                      // proj_out (FastDit31 coal path)

            ComputeBuffer Sc(string tensor)   // scales sibling of a q8 matmul, null when fp16
                => weights.Has(tensor + ".scales") ? weights.Get(tensor + ".scales") : null;

            void ResolveWeights()
            {
                if (wMod == null)
                {
                    wMod = new ComputeBuffer[DEPTH]; bMod = new ComputeBuffer[DEPTH];
                    wQ = new ComputeBuffer[DEPTH]; bQ = new ComputeBuffer[DEPTH];
                    wK = new ComputeBuffer[DEPTH]; bK = new ComputeBuffer[DEPTH];
                    wV = new ComputeBuffer[DEPTH]; bV = new ComputeBuffer[DEPTH];
                    wO = new ComputeBuffer[DEPTH]; bO = new ComputeBuffer[DEPTH];
                    wF1 = new ComputeBuffer[DEPTH]; bF1 = new ComputeBuffer[DEPTH];
                    wF2 = new ComputeBuffer[DEPTH]; bF2 = new ComputeBuffer[DEPTH];
                    sMod = new ComputeBuffer[DEPTH]; sQ = new ComputeBuffer[DEPTH]; sK = new ComputeBuffer[DEPTH];
                    sV = new ComputeBuffer[DEPTH]; sO = new ComputeBuffer[DEPTH];
                    sF1 = new ComputeBuffer[DEPTH]; sF2 = new ComputeBuffer[DEPTH];
                }
                for (int b = 0; b < DEPTH; b++)
                {
                    string blk = EST + $"transformer_blocks.{b}.";
                    wMod[b] = weights.Get(blk + "attn_norm.linear.weight"); bMod[b] = weights.Get(blk + "attn_norm.linear.bias");
                    wQ[b] = weights.Get(blk + "attn.to_q.weight"); bQ[b] = weights.Get(blk + "attn.to_q.bias");
                    wK[b] = weights.Get(blk + "attn.to_k.weight"); bK[b] = weights.Get(blk + "attn.to_k.bias");
                    wV[b] = weights.Get(blk + "attn.to_v.weight"); bV[b] = weights.Get(blk + "attn.to_v.bias");
                    wO[b] = weights.Get(blk + "attn.to_out.0.weight"); bO[b] = weights.Get(blk + "attn.to_out.0.bias");
                    wF1[b] = weights.Get(blk + "ff.ff.0.0.weight"); bF1[b] = weights.Get(blk + "ff.ff.0.0.bias");
                    wF2[b] = weights.Get(blk + "ff.ff.2.weight"); bF2[b] = weights.Get(blk + "ff.ff.2.bias");
                    sMod[b] = Sc(blk + "attn_norm.linear.weight");
                    sQ[b] = Sc(blk + "attn.to_q.weight"); sK[b] = Sc(blk + "attn.to_k.weight");
                    sV[b] = Sc(blk + "attn.to_v.weight"); sO[b] = Sc(blk + "attn.to_out.0.weight");
                    sF1[b] = Sc(blk + "ff.ff.0.0.weight"); sF2[b] = Sc(blk + "ff.ff.2.weight");
                }
                wPo = weights.Get(EST + "proj_out.weight"); bPo = weights.Get(EST + "proj_out.bias");
                sPo = Sc(EST + "proj_out.weight");
                wProj = weights.Get(EST + "input_embed.proj.weight"); bProj = weights.Get(EST + "input_embed.proj.bias");
                wCp1 = weights.Get(EST + "input_embed.conv_pos_embed.conv1.0.weight"); bCp1 = weights.Get(EST + "input_embed.conv_pos_embed.conv1.0.bias");
                wCp2 = weights.Get(EST + "input_embed.conv_pos_embed.conv2.0.weight"); bCp2 = weights.Get(EST + "input_embed.conv_pos_embed.conv2.0.bias");
                ropeFreq = weights.Get(EST + "rotary_embed.inv_freq");
            }

            public CosyVoiceFlow(CosyVoiceWeights weights, string voice = "default")
            {
                this.weights = weights;
                cs = DeepUnityMeta.CosyVoiceFlowCS;
                kTokenEmbed = cs.FindKernel("TokenEmbed");
                kLinear = cs.FindKernel("LinearBias");
                kLinearT = cs.FindKernel("LinearTileBias");
                kLinearQ8 = cs.FindKernel("LinearBiasQ8");
                kLinearTQ8 = cs.FindKernel("LinearTileBiasQ8");
                kConv = cs.FindKernel("Conv1D");
                kConvG = cs.FindKernel("Conv1DGrouped");
                kAdaLN = cs.FindKernel("AdaLNModulate");
                kGateAdd = cs.FindKernel("GateAdd");
                kRope = cs.FindKernel("RopeQK");
                kBidir = cs.FindKernel("BidirAttention");
                kBidirKV = cs.FindKernel("BidirAttentionKV");
                kBidirQT = cs.FindKernel("BidirAttentionQT");
                kBidirKVQT = cs.FindKernel("BidirAttentionKVQT");
                kWriteKV = cs.FindKernel("WriteFlowKV");
                kLinearB2 = cs.FindKernel("LinearTileBias2");
                kLinearB2Q8 = cs.FindKernel("LinearTileBias2Q8");
                kStats = cs.FindKernel("AdaLNStats");
                kDitQkv = cs.FindKernel("DitQKVCoal");
                kDitQkvQ8 = cs.FindKernel("DitQKVCoalQ8");
                kDitLin = cs.FindKernel("DitLinearCoal");
                kDitLinQ8 = cs.FindKernel("DitLinearCoalQ8");
                kRopePair = cs.FindKernel("RopeQKPair");
                kPackEst = cs.FindKernel("PackEstIn");
                kTimeEmb = cs.FindKernel("SinusTimeEmb");
                kRepeat = cs.FindKernel("RepeatTime");
                kPack = cs.FindKernel("PackChannels");
                kPackB = cs.FindKernel("PackBroadcastCh");
                kZero = cs.FindKernel("ZeroBuffer");
                kCopy = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAdd = cs.FindKernel("AddResidual");
                kActivate = cs.FindKernel("Activate");
                kEulerCfg = cs.FindKernel("EulerCfgStep");

                promptTokens = weights.ReadInts($"voices/{voice}/prompt_speech_tokens");
                promptFeat = weights.ReadFloats($"voices/{voice}/prompt_feat");
                promptMel = weights.Shape($"voices/{voice}/prompt_feat")[0];
                spkEmbedding = weights.ReadFloats($"voices/{voice}/embedding");
                double n2 = 0; foreach (float v in spkEmbedding) n2 += (double)v * v;
                float inv = (float)(1.0 / Math.Sqrt(Math.Max(n2, 1e-12)));
                for (int i = 0; i < spkEmbedding.Length; i++) spkEmbedding[i] *= inv;
                randNoise = weights.ReadFloats("flow/rand_noise");

                spkInBuf = new ComputeBuffer(CosyVoiceConfig.SPK_EMBED_DIM, 4, ComputeBufferType.Structured);
                spkBuf = new ComputeBuffer(MEL, 4, ComputeBufferType.Structured);
                tFreqBuf = new ComputeBuffer(CosyVoiceConfig.DIT_TIME_FREQ_DIM, 4, ComputeBufferType.Structured);
                tMidBuf = new ComputeBuffer(DIM, 4, ComputeBufferType.Structured);
                tEmbBuf = new ComputeBuffer(DIM, 4, ComputeBufferType.Structured);
                tSiluBuf = new ComputeBuffer(DIM, 4, ComputeBufferType.Structured);
                modTmpBuf = new ComputeBuffer(6 * DIM, 4, ComputeBufferType.Structured);
                modFBuf = new ComputeBuffer(2 * DIM, 4, ComputeBufferType.Structured);
            }

            /// <summary>Forget all cross-chunk streaming state (call at the start of every
            /// streamed utterance; finalize also resets automatically).</summary>
            public void ResetStream()
            {
                streamF = 0;
                streamFallback = false;
            }

            static int Div256(int n) => (n + 255) / 256;

            static void Grow(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            // xBuf must survive growth mid-utterance: frozen rows [0, streamF) hold final mel.
            void GrowPreserveX(int count)
            {
                if (xBuf != null && xBuf.count >= count) return;
                var old = xBuf;
                xBuf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
                if (old != null && streamF > 0) CopyOp(xBuf, old, old.count);
                old?.Release();
            }

            void EnsureScratch(int Ttok, int M)
            {
                if (Ttok <= curTok && M <= curM) return;
                curTok = Math.Max(Ttok, curTok); curM = Math.Max(M, curM);
                Grow(ref tokIdsBuf, curTok);
                Grow(ref embBuf, curTok * MEL);
                Grow(ref plaBuf, curTok * CosyVoiceConfig.PRE_LOOKAHEAD_CH);
                Grow(ref hBuf, curTok * MEL);
                Grow(ref muBuf, curM * MEL);
                Grow(ref condBuf, curM * MEL);
                GrowPreserveX(curM * MEL);
                // estimator scratch holds both stacked CFG halves (batch-2)
                Grow(ref estInBuf, 2 * curM * CosyVoiceConfig.DIT_IN_CONCAT);
                Grow(ref eA, 2 * curM * DIM); Grow(ref eB, 2 * curM * DIM);
                Grow(ref eS, 2 * curM * DIM);
                Grow(ref qBuf, 2 * curM * DIM); Grow(ref kBuf, 2 * curM * DIM); Grow(ref vBuf, 2 * curM * DIM);
                Grow(ref attnBuf, 2 * curM * DIM);
                Grow(ref ffBuf, 2 * curM * FF);
                Grow(ref dxdtA, 2 * curM * MEL);
                Grow(ref dxdtB, curM * MEL);   // uncond tap copy for the A2 probe only
                Grow(ref statsBuf, 2 * curM * 2);   // FastDit31 per-row (mean, rstd)
            }

            void EnsureKvCache(int NT)
            {
                if (kvCache != null && kvCacheNT == NT && kvCacheCap == MaxStreamKvFrames) return;
                ReleaseKvCache();
                kvCacheNT = NT;
                kvCacheCap = MaxStreamKvFrames;
                kvCache = new ComputeBuffer[NT * DEPTH];
                for (int i = 0; i < kvCache.Length; i++)   // 4 planes x cap x 1024 fp16 = 2*cap*DIM uints
                    kvCache[i] = new ComputeBuffer(2 * kvCacheCap * DIM, 4, ComputeBufferType.Structured);
            }

            void ReleaseKvCache()
            {
                if (kvCache == null) return;
                foreach (var b in kvCache) b?.Release();
                kvCache = null;
                kvCacheNT = -1;
            }

            // ---------------- op helpers ([T, C] layout) -----------------------------------------
            void Linear(string name, ComputeBuffer x, ComputeBuffer y, int T, int inDim, int outDim, int act = 0)
                => LinearW(weights.Get(name + ".weight"), weights.Get(name + ".bias"), x, y, T, inDim, outDim, act,
                           Sc(name + ".weight"));

            void LinearW(ComputeBuffer w, ComputeBuffer bias, ComputeBuffer x, ComputeBuffer y,
                         int T, int inDim, int outDim, int act = 0, ComputeBuffer scales = null)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", 1);
                // big [T,*] matmuls (even in_dim) go through the tiled GEMM; tiny T=1 vectors
                // (t-embed chain, adaLN mods, spk) keep the simple kernel. scales != null picks
                // the q8 twin (int8 weight + per-row scale) — chosen per TENSOR, not globally.
                // FastGemm routes tiled matmuls to the register-blocked twin (bit-exact).
                bool tiled = T >= 8 && (inDim & 1) == 0;
                bool blocked = tiled && FastGemm;
                int k = scales != null ? (blocked ? kLinearB2Q8 : tiled ? kLinearTQ8 : kLinearQ8)
                                       : (blocked ? kLinearB2 : tiled ? kLinearT : kLinear);
                cs.SetBuffer(k, "X", x);
                cs.SetBuffer(k, "W", w);
                cs.SetBuffer(k, "W_bias", bias);
                cs.SetBuffer(k, "Y", y);
                if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                if (blocked) cs.Dispatch(k, (outDim + 63) / 64, (T + 31) / 32, 1);
                else if (tiled) cs.Dispatch(k, (outDim + 31) / 32, (T + 15) / 16, 1);
                else cs.Dispatch(k, 1, (T + 7) / 8, (outDim + 31) / 32);
            }

            void Conv(string name, ComputeBuffer x, ComputeBuffer y, int T, int inCh, int outCh,
                      int kernel, int padLeft, int act = 0, float leaky = 0.01f, int inLen = -1)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_len", inLen < 0 ? T : inLen);
                cs.SetInt("in_dim", inCh); cs.SetInt("out_dim", outCh);
                cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", 1);
                cs.SetInt("conv_dilation", 1); cs.SetInt("pad_left", padLeft);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", 1);
                cs.SetFloat("leaky_slope", leaky);
                cs.SetBuffer(kConv, "X", x);
                cs.SetBuffer(kConv, "W", weights.Get(name + ".weight"));
                cs.SetBuffer(kConv, "W_bias", weights.Get(name + ".bias"));
                cs.SetBuffer(kConv, "Y", y);
                cs.Dispatch(kConv, Div256(T * outCh), 1, 1);
            }

            // batchSeq != 0: X/Y hold two stacked halves of batchSeq rows; the conv window never
            // crosses the half boundary (batch-2 CFG / streaming apron builds).
            void ConvGroupedW(ComputeBuffer w, ComputeBuffer bias, ComputeBuffer x, ComputeBuffer y,
                              int T, int ch, int kernel, int groups, int padLeft, int act, int batchSeq = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_len", T);
                cs.SetInt("in_dim", ch); cs.SetInt("out_dim", ch);
                cs.SetInt("conv_kernel", kernel); cs.SetInt("conv_stride", 1);
                cs.SetInt("conv_dilation", 1); cs.SetInt("pad_left", padLeft);
                cs.SetInt("n_groups", groups);
                cs.SetInt("batch_seq", batchSeq);
                cs.SetInt("activation_type", act); cs.SetInt("has_bias", 1);
                cs.SetBuffer(kConvG, "X", x);
                cs.SetBuffer(kConvG, "W", w);
                cs.SetBuffer(kConvG, "W_bias", bias);
                cs.SetBuffer(kConvG, "Y", y);
                cs.Dispatch(kConvG, Div256(T * ch), 1, 1);
            }

            void AdaLN(ComputeBuffer x, ComputeBuffer y, int T, ComputeBuffer mod, int scaleOff, int shiftOff)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", DIM);
                cs.SetFloat("norm_eps", CosyVoiceConfig.DIT_LN_EPS);
                cs.SetInt("mod_scale_off", scaleOff); cs.SetInt("mod_shift_off", shiftOff);
                cs.SetBuffer(kAdaLN, "norm_input", x);
                cs.SetBuffer(kAdaLN, "norm_output", y);
                cs.SetBuffer(kAdaLN, "mod_vec", mod);
                cs.Dispatch(kAdaLN, T, 1, 1);   // one cooperative group per row
            }

            void GateAdd(ComputeBuffer dst, ComputeBuffer src, int T, ComputeBuffer mod, int gateOff)
            {
                cs.SetInt("seq_len", T); cs.SetInt("norm_dim", DIM);
                cs.SetInt("mod_gate_off", gateOff);
                cs.SetBuffer(kGateAdd, "buf_a", dst);
                cs.SetBuffer(kGateAdd, "buf_b", src);
                cs.SetBuffer(kGateAdd, "mod_vec", mod);
                cs.Dispatch(kGateAdd, Div256(T * DIM), 1, 1);
            }

            void PackOp(ComputeBuffer dst, int dstDim, int dstOff, ComputeBuffer src, int T, int srcDim,
                        int srcRow = 0, int dstRow = 0)
            {
                cs.SetInt("seq_len", T); cs.SetInt("in_dim", srcDim);
                cs.SetInt("pack_dst_dim", dstDim); cs.SetInt("pack_dst_off", dstOff);
                cs.SetInt("pack_src_row", srcRow); cs.SetInt("pack_dst_row", dstRow);
                cs.SetBuffer(kPack, "buf_a", dst); cs.SetBuffer(kPack, "buf_b", src);
                cs.Dispatch(kPack, Div256(T * srcDim), 1, 1);
            }

            void PackSpk(int rows, int IN)
            {
                cs.SetInt("seq_len", rows); cs.SetInt("in_dim", MEL);
                cs.SetInt("pack_dst_dim", IN); cs.SetInt("pack_dst_off", 3 * MEL);
                cs.SetBuffer(kPackB, "buf_a", estInBuf); cs.SetBuffer(kPackB, "buf_b", spkBuf);
                cs.Dispatch(kPackB, Div256(rows * MEL), 1, 1);
            }

            void ZeroOp(ComputeBuffer dst, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kZero, "buf_a", dst);
                cs.Dispatch(kZero, Div256(count), 1, 1);
            }

            void CopyOp(ComputeBuffer dst, ComputeBuffer src, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kCopy, "buf_a", dst); cs.SetBuffer(kCopy, "buf_b", src);
                cs.Dispatch(kCopy, Div256(count), 1, 1);
            }

            void CopySliceOp(ComputeBuffer dst, int dstOff, ComputeBuffer src, int srcOff, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetInt("copy_dst_offset", dstOff); cs.SetInt("copy_src_offset", srcOff);
                cs.SetBuffer(kCopySlice, "buf_a", dst); cs.SetBuffer(kCopySlice, "buf_b", src);
                cs.Dispatch(kCopySlice, Div256(count), 1, 1);
            }

            void AddOp(ComputeBuffer dstA, ComputeBuffer srcB, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kAdd, "buf_a", dstA); cs.SetBuffer(kAdd, "buf_b", srcB);
                cs.Dispatch(kAdd, Div256(count), 1, 1);
            }

            void ActivateOp(ComputeBuffer buf, int count, int act)
            {
                cs.SetInt("buffer_size", count); cs.SetInt("activation_type", act);
                cs.SetBuffer(kActivate, "inout_buf", buf);
                cs.Dispatch(kActivate, Div256(count), 1, 1);
            }

            void RopeBatched(ComputeBuffer buf, int rowsHalf, int posOff)
            {
                cs.SetInt("seq_len", 2 * rowsHalf);
                cs.SetInt("in_dim", DIM);
                cs.SetInt("batch_seq", rowsHalf);
                cs.SetInt("pos_offset", posOff);
                cs.SetBuffer(kRope, "rope_freqs", ropeFreq);
                cs.SetBuffer(kRope, "inout_buf", buf);
                cs.Dispatch(kRope, Div256(2 * rowsHalf * 32), 1, 1);
            }

            // ---------------- FastDit31 dispatch helpers (DEEPOPT §4.2) --------------------------
            // AdaLNStats: per-row (mean, rstd) into statsBuf — the exact AdaLNModulate trees.
            void StatsOp(ComputeBuffer x, int rows)
            {
                cs.SetInt("seq_len", rows); cs.SetInt("norm_dim", DIM);
                cs.SetFloat("norm_eps", CosyVoiceConfig.DIT_LN_EPS);
                cs.SetBuffer(kStats, "norm_input", x);
                cs.SetBuffer(kStats, "est_stats_w", statsBuf);
                cs.Dispatch(kStats, rows, 1, 1);
            }

            // Generic #31 GemmCoal linear: inMode 1 stages AdaLN-modulate from statsBuf +
            // mod (scaleOff/shiftOff); outMode 1 fuses the GateAdd epilogue (gateOff).
            void DitLinear(ComputeBuffer w, ComputeBuffer b, ComputeBuffer scales,
                           ComputeBuffer x, ComputeBuffer y, int rows, int inDim, int outDim,
                           ComputeBuffer mod, int inMode = 0, int scaleOff = 0, int shiftOff = 0,
                           int act = 0, int outMode = 0, int gateOff = 0)
            {
                cs.SetInt("seq_len", rows); cs.SetInt("in_dim", inDim); cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act);
                cs.SetInt("dit_in_mode", inMode); cs.SetInt("dit_out_mode", outMode);
                cs.SetInt("mod_scale_off", scaleOff); cs.SetInt("mod_shift_off", shiftOff);
                cs.SetInt("mod_gate_off", gateOff);
                int k = scales != null ? kDitLinQ8 : kDitLin;
                cs.SetBuffer(k, "X", x);
                cs.SetBuffer(k, "W", w);
                cs.SetBuffer(k, "W_bias", b);
                if (scales != null) cs.SetBuffer(k, "W_scales", scales);
                cs.SetBuffer(k, "Y", y);
                cs.SetBuffer(k, "mod_vec", mod);
                cs.SetBuffer(k, "est_stats", statsBuf);
                cs.Dispatch(k, (outDim + 7) / 8, (rows + 7) / 8, 1);
            }

            // Fused to_q/to_k/to_v (+biases) with modulate staging. Caller guarantees the
            // three tensors share one quant status (Sc() is per-tensor).
            void DitQkv(int b, ComputeBuffer x, int rows, int scaleOff, int shiftOff)
            {
                cs.SetInt("seq_len", rows); cs.SetInt("in_dim", DIM); cs.SetInt("out_dim", DIM);
                cs.SetInt("activation_type", 0);
                cs.SetInt("dit_in_mode", 1);
                cs.SetInt("mod_scale_off", scaleOff); cs.SetInt("mod_shift_off", shiftOff);
                bool q8 = sQ[b] != null;
                int k = q8 ? kDitQkvQ8 : kDitQkv;
                cs.SetBuffer(k, "X", x);
                cs.SetBuffer(k, "W", wQ[b]); cs.SetBuffer(k, "W_k", wK[b]); cs.SetBuffer(k, "W_v", wV[b]);
                cs.SetBuffer(k, "W_bias", bQ[b]); cs.SetBuffer(k, "W_k_bias", bK[b]); cs.SetBuffer(k, "W_v_bias", bV[b]);
                if (q8)
                {
                    cs.SetBuffer(k, "W_scales", sQ[b]);
                    cs.SetBuffer(k, "W_k_scales", sK[b]);
                    cs.SetBuffer(k, "W_v_scales", sV[b]);
                }
                cs.SetBuffer(k, "Y", qBuf); cs.SetBuffer(k, "Y2", kBuf); cs.SetBuffer(k, "Y3", vBuf);
                cs.SetBuffer(k, "mod_vec", modAllSteps);
                cs.SetBuffer(k, "est_stats", statsBuf);
                cs.Dispatch(k, 384, (rows + 7) / 8, 1);   // 128 Q + 128 K + 128 V row-groups
            }

            void RopePairOp(int rowsHalf, int posOff)
            {
                cs.SetInt("seq_len", 2 * rowsHalf);
                cs.SetInt("in_dim", DIM);
                cs.SetInt("batch_seq", rowsHalf);
                cs.SetInt("pos_offset", posOff);
                cs.SetBuffer(kRopePair, "rope_freqs", ropeFreq);
                cs.SetBuffer(kRopePair, "inout_buf", qBuf);
                cs.SetBuffer(kRopePair, "inout_buf2", kBuf);
                cs.Dispatch(kRopePair, Div256(2 * rowsHalf * 32), 1, 1);
            }

            // One-dispatch estimator-input build (bit-exact vs the legacy Zero+Pack chain).
            void PackEstInOp(int Le, int apron, int xrow, int condRow, int tailRow, ComputeBuffer tail)
            {
                cs.SetInt("est_rows", Le);
                cs.SetInt("est_apron", apron);
                cs.SetInt("est_xrow", xrow);
                cs.SetInt("est_cond_row", condRow);
                cs.SetInt("est_tail_row", tailRow);
                cs.SetBuffer(kPackEst, "est_x", xBuf);
                cs.SetBuffer(kPackEst, "est_tail", tail);
                cs.SetBuffer(kPackEst, "est_cond", condBuf);
                cs.SetBuffer(kPackEst, "est_mu", muBuf);
                cs.SetBuffer(kPackEst, "est_spk", spkBuf);
                cs.SetBuffer(kPackEst, "est_out", estInBuf);
                cs.Dispatch(kPackEst, Div256(2 * Le * CosyVoiceConfig.DIT_IN_CONCAT), 1, 1);
            }

            // x[F..F+rows) += dt * ((1+β)·cond − β·uncond); cond = dxdtA rows [0,rows),
            // uncond = dxdtA rows [rows, 2rows) via the kernel's src offset.
            void EulerCfg(int rows, int dstRowOff, float dt)
            {
                cs.SetInt("buffer_size", rows * MEL);
                cs.SetFloat("dt_val", dt);
                cs.SetFloat("scale_val", CosyVoiceConfig.CFM_INFERENCE_CFG_RATE);
                cs.SetInt("copy_dst_offset", dstRowOff * MEL);
                cs.SetInt("copy_src_offset", rows * MEL);
                cs.SetBuffer(kEulerCfg, "X", dxdtA);
                cs.SetBuffer(kEulerCfg, "buf_b", dxdtA);
                cs.SetBuffer(kEulerCfg, "inout_buf", xBuf);
                cs.Dispatch(kEulerCfg, Div256(rows * MEL), 1, 1);
            }

            // ---------------- AdaLN modulation precompute (bit-exact, once per instance) ---------
            // Every mod is a function of t only (DiT §3.2: AdaLN regresses from the t embedding;
            // the speaker enters via the input concat) and the cosine t-schedule is fixed, so all
            // NT x 22 block mods + NT final-norm mods are constants. Removes ~50 tiny dispatches
            // per Euler step — the big win is streaming, which used to redo them every chunk.
            void EnsureMods(int NT, float[] ts)
            {
                if (modsNT == NT) return;
                Grow(ref modAllSteps, NT * DEPTH * 6 * DIM);
                Grow(ref modFSteps, NT * 2 * DIM);
                for (int step = 1; step <= NT; step++)
                {
                    cs.SetInt("out_dim", CosyVoiceConfig.DIT_TIME_FREQ_DIM);
                    cs.SetFloat("t_scalar", ts[step - 1]);
                    cs.SetBuffer(kTimeEmb, "Y", tFreqBuf);
                    cs.Dispatch(kTimeEmb, 1, 1, 1);
                    Linear(EST + "time_embed.time_mlp.0", tFreqBuf, tMidBuf, 1, CosyVoiceConfig.DIT_TIME_FREQ_DIM, DIM, act: 1);
                    Linear(EST + "time_embed.time_mlp.2", tMidBuf, tEmbBuf, 1, DIM, DIM);
                    CopyOp(tSiluBuf, tEmbBuf, DIM);
                    ActivateOp(tSiluBuf, DIM, 1);

                    for (int b = 0; b < DEPTH; b++)
                    {
                        LinearW(wMod[b], bMod[b], tSiluBuf, modTmpBuf, 1, DIM, 6 * DIM, scales: sMod[b]);
                        CopySliceOp(modAllSteps, ((step - 1) * DEPTH + b) * 6 * DIM, modTmpBuf, 0, 6 * DIM);
                    }
                    Linear(EST + "norm_out.linear", tSiluBuf, modFBuf, 1, DIM, 2 * DIM);
                    CopySliceOp(modFSteps, (step - 1) * 2 * DIM, modFBuf, 0, 2 * DIM);
                }
                modsNT = NT;
            }

            // ---------------- DiT blocks (shared by the full and cached estimators) --------------
            // resid = residual stream [2*rowsHalf, DIM] (CFG halves stacked). cachedStep >= 0 =
            // single-pass streaming: K/V of the new rows are appended to the per-(step,layer)
            // fp16 cache and attention runs new-queries vs the whole cached prefix [0, F+rowsHalf).
            void RunBlocks(ComputeBuffer resid, int rowsHalf, int modBase, int cachedStep, int F)
            {
                int rows2 = 2 * rowsHalf;
                bool cached = cachedStep >= 0;
                for (int b = 0; b < DEPTH; b++)
                {
                    int off = modBase + b * 6 * DIM;   // shift_msa|scale_msa|gate_msa|shift_mlp|scale_mlp|gate_mlp

                    // attn branch
                    if (FastDit31)
                    {
                        StatsOp(resid, rows2);
                        // fused QKV only when q/k/v share one quant status (Sc is per-tensor)
                        bool q8m = sQ[b] != null && sK[b] != null && sV[b] != null;
                        bool fpm = sQ[b] == null && sK[b] == null && sV[b] == null;
                        if (q8m || fpm)
                            DitQkv(b, resid, rows2, off + DIM, off);
                        else
                        {
                            DitLinear(wQ[b], bQ[b], sQ[b], resid, qBuf, rows2, DIM, DIM,
                                      modAllSteps, inMode: 1, scaleOff: off + DIM, shiftOff: off);
                            DitLinear(wK[b], bK[b], sK[b], resid, kBuf, rows2, DIM, DIM,
                                      modAllSteps, inMode: 1, scaleOff: off + DIM, shiftOff: off);
                            DitLinear(wV[b], bV[b], sV[b], resid, vBuf, rows2, DIM, DIM,
                                      modAllSteps, inMode: 1, scaleOff: off + DIM, shiftOff: off);
                        }
                        RopePairOp(rowsHalf, cached ? F : 0);
                    }
                    else
                    {
                        AdaLN(resid, eB, rows2, modAllSteps, off + DIM, off);
                        LinearW(wQ[b], bQ[b], eB, qBuf, rows2, DIM, DIM, scales: sQ[b]);
                        LinearW(wK[b], bK[b], eB, kBuf, rows2, DIM, DIM, scales: sK[b]);
                        LinearW(wV[b], bV[b], eB, vBuf, rows2, DIM, DIM, scales: sV[b]);
                        RopeBatched(qBuf, rowsHalf, cached ? F : 0);
                        RopeBatched(kBuf, rowsHalf, cached ? F : 0);
                    }

                    cs.SetInt("num_heads", CosyVoiceConfig.DIT_HEADS);
                    cs.SetInt("head_dim", CosyVoiceConfig.DIT_HEAD_DIM);
                    cs.SetFloat("scale", 1f / Mathf.Sqrt(CosyVoiceConfig.DIT_HEAD_DIM));
                    if (cached)
                    {
                        var kv = kvCache[cachedStep * DEPTH + b];
                        cs.SetInt("seq_len", rowsHalf);
                        cs.SetInt("kv_row_off", F); cs.SetInt("kv_cap", kvCacheCap);
                        cs.SetBuffer(kWriteKV, "K", kBuf); cs.SetBuffer(kWriteKV, "V", vBuf);
                        cs.SetBuffer(kWriteKV, "flow_kv_w", kv);
                        cs.Dispatch(kWriteKV, Div256(rowsHalf * DIM), 1, 1);   // 2 halves x rows x 512 uints

                        cs.SetInt("batch_seq", rowsHalf);
                        cs.SetInt("pos_offset", F);
                        cs.SetInt("kv_len", F + rowsHalf);
                        int ka = FastAttention ? kBidirKVQT : kBidirKV;
                        cs.SetBuffer(ka, "Q", qBuf);
                        cs.SetBuffer(ka, "flow_kv", kv);
                        cs.SetBuffer(ka, "AttendedValues", attnBuf);
                        if (FastAttention) cs.Dispatch(ka, 2 * ((rowsHalf + 7) / 8), CosyVoiceConfig.DIT_HEADS, 1);
                        else cs.Dispatch(ka, rows2, CosyVoiceConfig.DIT_HEADS, 1);
                    }
                    else
                    {
                        cs.SetInt("seq_len", rows2);
                        cs.SetInt("batch_seq", rowsHalf);
                        int ka = FastAttention ? kBidirQT : kBidir;
                        cs.SetBuffer(ka, "Q", qBuf); cs.SetBuffer(ka, "K", kBuf);
                        cs.SetBuffer(ka, "V", vBuf);
                        cs.SetBuffer(ka, "AttendedValues", attnBuf);
                        if (FastAttention) cs.Dispatch(ka, 2 * ((rowsHalf + 7) / 8), CosyVoiceConfig.DIT_HEADS, 1);
                        else cs.Dispatch(ka, rows2, CosyVoiceConfig.DIT_HEADS, 1);
                    }

                    if (FastDit31)
                    {
                        // to_out with the gate-add epilogue fused, then the ff branch:
                        // stats -> FF1 (modulate staging + GELU-tanh) -> FF2 (gate-add)
                        DitLinear(wO[b], bO[b], sO[b], attnBuf, resid, rows2, DIM, DIM,
                                  modAllSteps, outMode: 1, gateOff: off + 2 * DIM);
                        StatsOp(resid, rows2);
                        DitLinear(wF1[b], bF1[b], sF1[b], resid, ffBuf, rows2, DIM, FF,
                                  modAllSteps, inMode: 1, scaleOff: off + 4 * DIM, shiftOff: off + 3 * DIM, act: 8);
                        DitLinear(wF2[b], bF2[b], sF2[b], ffBuf, resid, rows2, FF, DIM,
                                  modAllSteps, outMode: 1, gateOff: off + 5 * DIM);
                    }
                    else
                    {
                        LinearW(wO[b], bO[b], attnBuf, eB, rows2, DIM, DIM, scales: sO[b]);
                        GateAdd(resid, eB, rows2, modAllSteps, off + 2 * DIM);

                        // ff branch (GELU tanh)
                        AdaLN(resid, eB, rows2, modAllSteps, off + 4 * DIM, off + 3 * DIM);
                        LinearW(wF1[b], bF1[b], eB, ffBuf, rows2, DIM, FF, act: 8, scales: sF1[b]);
                        LinearW(wF2[b], bF2[b], ffBuf, eB, rows2, FF, DIM, scales: sF2[b]);
                        GateAdd(resid, eB, rows2, modAllSteps, off + 5 * DIM);
                    }
                }
            }

            // ---------------- batch-2 CFG estimator, full sequence (offline / legacy streaming) --
            // One stacked pass: rows [0,M) = cond input (x|cond|mu|spk), rows [M,2M) = uncond
            // (x|0|0|0). Bit-exact per row vs the old two sequential passes.
            void EstimatorFull(int M, int modBase, int modFBase, ComputeBuffer dxdtOut)
            {
                int IN = CosyVoiceConfig.DIT_IN_CONCAT;
                if (FastDit31)
                {
                    // one dispatch, bit-exact data movement (apron 0, est_tail never read)
                    PackEstInOp(M, apron: 0, xrow: 0, condRow: 0, tailRow: 0, tail: xBuf);
                }
                else
                {
                    ZeroOp(estInBuf, 2 * M * IN);
                    PackOp(estInBuf, IN, 0, xBuf, M, MEL);
                    PackOp(estInBuf, IN, MEL, condBuf, M, MEL);
                    PackOp(estInBuf, IN, 2 * MEL, muBuf, M, MEL);
                    PackSpk(M, IN);
                    PackOp(estInBuf, IN, 0, xBuf, M, MEL, 0, M);   // uncond half keeps x only
                }

                LinearW(wProj, bProj, estInBuf, eA, 2 * M, IN, DIM);
                // += CausalConvPos: 2x (grouped k31 LEFT-pad30 + Mish), per half
                ConvGroupedW(wCp1, bCp1, eA, eB, 2 * M, DIM,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL, CosyVoiceConfig.DIT_CONVPOS_GROUPS,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL - 1, act: 3, batchSeq: M);
                ConvGroupedW(wCp2, bCp2, eB, attnBuf, 2 * M, DIM,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL, CosyVoiceConfig.DIT_CONVPOS_GROUPS,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL - 1, act: 3, batchSeq: M);
                AddOp(eA, attnBuf, 2 * M * DIM);

                RunBlocks(eA, M, modBase, cachedStep: -1, F: 0);

                // final AdaLN — NOTE norm_out chunk order is (SCALE, shift), reversed vs blocks
                if (FastDit31)
                {
                    StatsOp(eA, 2 * M);
                    DitLinear(wPo, bPo, sPo, eA, dxdtOut, 2 * M, DIM, MEL,
                              modFSteps, inMode: 1, scaleOff: modFBase, shiftOff: modFBase + DIM);
                }
                else
                {
                    AdaLN(eA, eB, 2 * M, modFSteps, modFBase, modFBase + DIM);
                    Linear(EST + "proj_out", eB, dxdtOut, 2 * M, DIM, MEL);
                }
            }

            // ---------------- single-pass streaming estimator, one Euler step --------------------
            // Solves ONLY rows [F, M). The conv-pos stage needs a 60-row left apron whose proj
            // input depends on x_s of already-frozen rows — rebuilt exactly from the per-step
            // x-tail cache (cond/mu/spk lanes are step-independent and recomputed fresh). The
            // apron's conv1 rows [apron-30, apron) are exact; junk in earlier apron rows is never
            // consumed by the compacted [F, M) band.
            void EstimatorCachedStep(int F, int M, int step0, int modBase, int modFBase, ComputeBuffer dxdtOut)
            {
                int Mn = M - F;
                int apron = Math.Min(APRON, F);
                int Le = apron + Mn;
                int IN = CosyVoiceConfig.DIT_IN_CONCAT;

                if (FastDit31)
                {
                    // one dispatch; reproduces the Zero+Pack chain below exactly (DEEPOPT §4.2.5)
                    PackEstInOp(Le, apron, xrow: F, condRow: F - apron,
                                tailRow: step0 * APRON + (APRON - apron), tail: xTailA);
                }
                else
                {
                    ZeroOp(estInBuf, 2 * Le * IN);
                    if (apron > 0)
                    {
                        int tailRow = step0 * APRON + (APRON - apron);   // tail stores rows [F-apron, F)
                        PackOp(estInBuf, IN, 0, xTailA, apron, MEL, tailRow, 0);
                        PackOp(estInBuf, IN, 0, xTailA, apron, MEL, tailRow, Le);
                    }
                    PackOp(estInBuf, IN, 0, xBuf, Mn, MEL, F, apron);
                    PackOp(estInBuf, IN, 0, xBuf, Mn, MEL, F, Le + apron);
                    PackOp(estInBuf, IN, MEL, condBuf, Le, MEL, F - apron, 0);
                    PackOp(estInBuf, IN, 2 * MEL, muBuf, Le, MEL, F - apron, 0);
                    PackSpk(Le, IN);
                }

                LinearW(wProj, bProj, estInBuf, eA, 2 * Le, IN, DIM);
                ConvGroupedW(wCp1, bCp1, eA, eB, 2 * Le, DIM,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL, CosyVoiceConfig.DIT_CONVPOS_GROUPS,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL - 1, act: 3, batchSeq: Le);
                ConvGroupedW(wCp2, bCp2, eB, attnBuf, 2 * Le, DIM,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL, CosyVoiceConfig.DIT_CONVPOS_GROUPS,
                             CosyVoiceConfig.DIT_CONVPOS_KERNEL - 1, act: 3, batchSeq: Le);
                AddOp(eA, attnBuf, 2 * Le * DIM);

                // drop the aprons: compact to [2*Mn, DIM] halves for the block stack
                CopySliceOp(eS, 0, eA, apron * DIM, Mn * DIM);
                CopySliceOp(eS, Mn * DIM, eA, (Le + apron) * DIM, Mn * DIM);

                RunBlocks(eS, Mn, modBase, cachedStep: step0, F: F);

                if (FastDit31)
                {
                    StatsOp(eS, 2 * Mn);
                    DitLinear(wPo, bPo, sPo, eS, dxdtOut, 2 * Mn, DIM, MEL,
                              modFSteps, inMode: 1, scaleOff: modFBase, shiftOff: modFBase + DIM);
                }
                else
                {
                    AdaLN(eS, eB, 2 * Mn, modFSteps, modFBase, modFBase + DIM);
                    Linear(EST + "proj_out", eB, dxdtOut, 2 * Mn, DIM, MEL);
                }
            }

            // Save x_s of the rows that freeze this chunk ([FNew-APRON, FNew)) into xTailB[step].
            // Runs BEFORE this step's Euler update (the tail must hold the state ENTERING step s).
            // When the freeze advance is < APRON, the head of the new tail comes from the old one.
            void SaveXTail(int step0, int F, int FNew)
            {
                int n = Math.Min(APRON, FNew);
                if (n <= 0) return;
                int c1 = Math.Min(FNew - F, n);   // rows available in xBuf (x_s of [F, M))
                int c0 = n - c1;                  // rows carried over from the old tail
                int dstBase = step0 * APRON + (APRON - n);
                if (c0 > 0)
                    CopySliceOp(xTailB, dstBase * MEL, xTailA,
                                (step0 * APRON + (APRON - F + (FNew - n))) * MEL, c0 * MEL);
                if (c1 > 0)
                    CopySliceOp(xTailB, (dstBase + c0) * MEL, xBuf, (FNew - c1) * MEL, c1 * MEL);
            }

            /// <summary>genTokens: LM speech tokens (&lt; 6561). onMel(melBuf, promptMelFrames,
            /// outFrames): mel [T,80] rows [promptMelFrames, promptMelFrames+outFrames) are the
            /// synthesized output — feed rows directly to HiFTVocoder. Offline (full attention).</summary>
            public IEnumerator SynthesizeMelYielding(int[] genTokens, Action<ComputeBuffer, int, int> onMel)
                => SynthesizeMelCoreYielding(genTokens, streaming: false, finalize: true, onMel);

            /// <summary>Streaming chunk under the 50-frame chunk attention mask. With
            /// SinglePassStreaming only the NEW blocks are solved (frozen rows are bit-stable —
            /// see class header); otherwise the growing prefix is re-solved as before.
            /// finalize=false holds the last PRE_LOOKAHEAD_LEN tokens back as lookahead context
            /// (their mel is not emitted yet).</summary>
            public IEnumerator SynthesizeMelStreamingYielding(int[] tokensSoFar, bool finalize, Action<ComputeBuffer, int, int> onMel)
                => SynthesizeMelCoreYielding(tokensSoFar, streaming: true, finalize, onMel);

            IEnumerator SynthesizeMelCoreYielding(int[] genTokens, bool streaming, bool finalize, Action<ComputeBuffer, int, int> onMel)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                int Ttok = promptTokens.Length + genTokens.Length;
                int hRows = finalize ? Ttok : Ttok - CosyVoiceConfig.PRE_LOOKAHEAD_LEN;
                int M = hRows * CosyVoiceConfig.TOKEN_MEL_RATIO;
                if (M > CosyVoiceConfig.FIXED_NOISE_FRAMES)
                    throw new ArgumentException($"Utterance too long for the fixed noise buffer ({M} > {CosyVoiceConfig.FIXED_NOISE_FRAMES} mel frames).");

                bool cached = streaming && SinglePassStreaming && !streamFallback;
                if (cached && M > MaxStreamKvFrames)
                {
                    ConsoleMessage.Warning($"CosyVoiceFlow: utterance exceeds the streaming K/V cache ({M} > {MaxStreamKvFrames} mel frames) — full re-solve for the rest of this utterance.");
                    streamFallback = true;
                    cached = false;
                }
                if (cached && streamF > M) streamF = 0;   // stale state (caller skipped ResetStream)
                int F = cached ? streamF : 0;

                EnsureScratch(Ttok, M);
                ResolveWeights();
                int NT = TimestepsOverride > 0 ? TimestepsOverride : CosyVoiceConfig.CFM_TIMESTEPS;
                float[] ts = new float[NT + 1];
                for (int i = 0; i <= NT; i++)
                    ts[i] = 1f - Mathf.Cos((float)i / NT * 0.5f * Mathf.PI);
                EnsureMods(NT, ts);
                if (cached)
                {
                    EnsureKvCache(NT);
                    Grow(ref xTailA, NT * APRON * MEL);
                    Grow(ref xTailB, NT * APRON * MEL);
                }
                cs.SetInt("attn_chunk", streaming ? CosyVoiceConfig.CHUNK_MEL : 0);

                // ---- tokens -> embedding [Ttok, 80] (full prefix every chunk — cheap, and the
                // causal geometry reproduces frozen-row h/mu bit-identically)
                int[] tok = new int[Ttok];
                promptTokens.CopyTo(tok, 0);
                genTokens.CopyTo(tok, promptTokens.Length);
                tokIdsBuf.SetData(tok, 0, 0, Ttok);
                cs.SetInt("seq_len", Ttok); cs.SetInt("in_dim", MEL);
                cs.SetBuffer(kTokenEmbed, "token_ids", tokIdsBuf);
                cs.SetBuffer(kTokenEmbed, "emb_weights", weights.Get("flow/input_embedding.weight"));
                cs.SetBuffer(kTokenEmbed, "embed_output", embBuf);
                cs.Dispatch(kTokenEmbed, Div256(Ttok * MEL), 1, 1);

                // ---- PreLookaheadLayer: k4 RIGHT-pad3 leaky(0.01) -> k3 LEFT-pad2 -> +residual.
                // Non-finalize: output rows [0, Ttok-3) read only REAL token rows — exactly the
                // reference's context-split (the held-back 3 tokens act as lookahead context).
                Conv("flow/pre_lookahead_layer.conv1", embBuf, plaBuf, hRows, MEL,
                     CosyVoiceConfig.PRE_LOOKAHEAD_CH, CosyVoiceConfig.PRE_LOOKAHEAD_LEN + 1, 0,
                     act: 4, leaky: 0.01f, inLen: Ttok);
                Conv("flow/pre_lookahead_layer.conv2", plaBuf, hBuf, hRows,
                     CosyVoiceConfig.PRE_LOOKAHEAD_CH, MEL, 3, 2);
                AddOp(hBuf, embBuf, hRows * MEL);
                DebugTap?.Invoke("h_lookahead", hBuf, hRows * MEL);
                yield return null;

                // ---- mu (x2 repeat), spk, cond, x0
                cs.SetInt("seq_len", M); cs.SetInt("in_dim", MEL); cs.SetInt("factor", CosyVoiceConfig.TOKEN_MEL_RATIO);
                cs.SetBuffer(kRepeat, "X", hBuf); cs.SetBuffer(kRepeat, "Y", muBuf);
                cs.Dispatch(kRepeat, Div256(M * MEL), 1, 1);

                spkInBuf.SetData(spkEmbedding);
                Linear("flow/spk_embed_affine_layer", spkInBuf, spkBuf, 1, CosyVoiceConfig.SPK_EMBED_DIM, MEL);

                float[] condHost = new float[M * MEL];
                Array.Copy(promptFeat, condHost, promptMel * MEL);
                condBuf.SetData(condHost, 0, 0, M * MEL);

                // x0 slice of the fixed noise (rand_noise [80,15000] channel-major -> [M,80]);
                // single-pass: only the new rows — frozen rows keep their final mel in xBuf.
                int x0Lo = cached ? F : 0;
                float[] x0 = new float[(M - x0Lo) * MEL];
                for (int c = 0; c < MEL; c++)
                    for (int t = x0Lo; t < M; t++)
                        x0[(t - x0Lo) * MEL + c] = randNoise[c * CosyVoiceConfig.FIXED_NOISE_FRAMES + t];
                xBuf.SetData(x0, 0, x0Lo * MEL, x0.Length);
                yield return null;

                // rows frozen after this chunk = complete 50-frame blocks (the hop schedule makes
                // non-finalize M a multiple of 50, so normally FNew == M)
                int FNew = (M / CosyVoiceConfig.CHUNK_MEL) * CosyVoiceConfig.CHUNK_MEL;

                var swIssue = System.Diagnostics.Stopwatch.StartNew();
                for (int step = 1; step <= NT; step++)
                {
                    float dt = ts[step] - ts[step - 1];
                    int modBase = (step - 1) * DEPTH * 6 * DIM;
                    int modFBase = (step - 1) * 2 * DIM;

                    if (cached)
                    {
                        if (!finalize && FNew > F) SaveXTail(step - 1, F, FNew);
                        EstimatorCachedStep(F, M, step - 1, modBase, modFBase, dxdtA);
                        yield return null;
                        EulerCfg(M - F, F, dt);
                    }
                    else
                    {
                        EstimatorFull(M, modBase, modFBase, dxdtA);
                        if (step == 1 && DebugTap != null)
                        {
                            DebugTap("dxdt_cond_s0", dxdtA, M * MEL);
                            CopySliceOp(dxdtB, 0, dxdtA, M * MEL, M * MEL);
                            DebugTap("dxdt_uncond_s0", dxdtB, M * MEL);
                        }
                        yield return null;
                        EulerCfg(M, 0, dt);
                    }
                    yield return null;
                }
                swIssue.Stop();
                IssueMs = (float)swIssue.Elapsed.TotalMilliseconds;

                if (ProfileGpuFence)   // 9c: expose the true GPU tail (4-byte fence readback)
                {
                    var swF = System.Diagnostics.Stopwatch.StartNew();
                    float[] one = new float[1];
                    xBuf.GetData(one, 0, (M - 1) * MEL, 1);
                    swF.Stop();
                    GpuWaitMs = (float)swF.Elapsed.TotalMilliseconds;
                }

                if (streaming)
                {
                    if (finalize) { streamF = 0; streamFallback = false; }
                    else if (cached && FNew > F)
                    {
                        streamF = FNew;
                        (xTailA, xTailB) = (xTailB, xTailA);
                    }
                }

                sw.Stop();
                FlowMs = (float)sw.Elapsed.TotalMilliseconds;
                DebugTap?.Invoke("mel_full", xBuf, M * MEL);
                onMel?.Invoke(xBuf, promptMel, M - promptMel);
            }

            public void Dispose()
            {
                tokIdsBuf?.Release(); embBuf?.Release(); plaBuf?.Release(); hBuf?.Release();
                muBuf?.Release(); condBuf?.Release(); xBuf?.Release();
                spkInBuf?.Release(); spkBuf?.Release();
                tFreqBuf?.Release(); tMidBuf?.Release(); tEmbBuf?.Release(); tSiluBuf?.Release();
                modTmpBuf?.Release(); modFBuf?.Release();
                modAllSteps?.Release(); modFSteps?.Release();
                estInBuf?.Release(); eA?.Release(); eB?.Release(); eS?.Release();
                statsBuf?.Release();
                qBuf?.Release(); kBuf?.Release(); vBuf?.Release(); attnBuf?.Release();
                ffBuf?.Release(); dxdtA?.Release(); dxdtB?.Release();
                xTailA?.Release(); xTailB?.Release();
                ReleaseKvCache();
            }
        }
    }
}
