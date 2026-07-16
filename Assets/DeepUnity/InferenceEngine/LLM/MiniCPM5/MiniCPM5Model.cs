using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace MiniCPM5Modeling
    {
        // MiniCPM5-1B — vanilla llama decoder, full-GPU inference.
        //
        // Runs entirely on the Gemma3CS.compute kernel set: those kernels are dim-agnostic
        // GQA/RMSNorm/GLU building blocks (every size arrives as a uniform), and MiniCPM5's llama
        // architecture is a strict subset of Gemma3's (no sliding window -> sliding_window_size=0,
        // no embedding scale -> embed_scale=1, SiLU -> activation_type=0, no qk-norm -> the
        // RmsNormHead kernel is simply never dispatched). Precedent: Gemma3ForEmbeddings already
        // reuses the same asset. NOTE: keyword state (INT8_WEIGHTS / KV_FP16 / ...) lives on the
        // shared shader asset — one quant mode per session across Gemma3 AND MiniCPM5 instances.
        public class MiniCPM5Model : IDisposable
        {
            // FlashAttention-1 fused attention (see Gemma3Model). head_dim=128 <= 256, so the
            // fused kernel is always eligible; the legacy 4-dispatch path stays as an A/B fallback.
            public static bool UseFlashAttention = true;

            // #31 (2026-07-16): coalesced GEMV/GEMM kernels (shared Gemma3CS port — see
            // Gemma3Model). Static so probes can A/B against the legacy thread-per-row kernels.
            public static bool ForceLegacyGemv = false;
            const int GVC_ROWS = 8;   // rows per group — must match Gemma3CS GVC_ROWS
            const int GMM_TOK = 8;    // tokens per prefill tile — must match Gemma3CS GMM_TOK

            public readonly LLMQuant Quant;
            public readonly KVQuant KV;

            ComputeShader cs;

            // kernel ids
            int kEmbedLookup, kRmsNormHidden, kSplitQKV;
            int kApplyRope, kWriteCacheFull, kApplyMask, kSoftmaxRows, kFlashAttn;
            int kArgMax, kSampleToken, kCopyBuffer, kCopySlice, kAddResidual;
            int kQKVProj, kAttnScores, kAttendValues, kOProj;
            int kGateUp, kDown, kGateUp1Vec, kDown1Vec;
            int kLmHead, kLmHead1Vec;
            // #31 coalesced variants (decode 1VecCoal + prefill GemmCoal)
            int kQKVProj1C, kOProj1C, kGateUp1C, kDown1C, kLmHead1C;
            int kQKVProjG, kOProjG, kGateUpG, kDownG;

            public MiniCPM5Weights weights;
            public MiniCPM5Cache cache;

            // RoPE (packed FP16 in uint buffers) — llama has ONE theta, so one cos/sin pair.
            ComputeBuffer ropeCos, ropeSin;

            // FP32 scratch buffers
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf;
            ComputeBuffer qkvBuf, qBuf, kBuf, vBuf;
            ComputeBuffer attnScoresBuf, attendedBuf, attnOutBuf;
            ComputeBuffer mlpInterBuf;
            ComputeBuffer logitsBuf, probsBuf;
            ComputeBuffer argmaxBuf;
            ComputeBuffer tokenIdsBuf;
            ComputeBuffer lastHiddenBuf;
            ComputeBuffer normSingleBuf;

            int curSeqAlloc, curKvAlloc;

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;
            readonly int ropeMaxSeq;
            readonly float rmsEps, attnScaling;

            // RoPE tables are computed on a background thread and uploaded by a main-thread
            // coroutine when ready (same pattern as Gemma3Model).
            volatile bool ropeReady;
            volatile bool _ropeComputed;
            uint[] _ropeC, _ropeS;

            public bool IsReady => weights.IsReady && ropeReady;

            public MiniCPM5Model(string paramsPath, int cacheCapacity, LLMQuant quant = LLMQuant.FP16,
                                 KVQuant kvQuant = KVQuant.FP16)
            {
                Quant = quant;
                KV = kvQuant;
                numLayers = MiniCPM5Config.NUM_LAYERS;
                hiddenSize = MiniCPM5Config.HIDDEN_SIZE;
                headDim = MiniCPM5Config.HEAD_DIM;
                headsQ = MiniCPM5Config.HEADS_Q;
                headsKV = MiniCPM5Config.HEADS_KV;
                intermediateSize = MiniCPM5Config.MLP_INTERMEDIATE_SIZE;
                vocabSize = MiniCPM5Config.VOCAB_SIZE;
                rmsEps = MiniCPM5Config.RMS_EPS;
                attnScaling = MathF.Pow(headDim, -0.5f);   // llama: 1/sqrt(head_dim)

                innerEmbDim = headsQ * headDim;                       // 2048 (≠ hidden 1536)
                qkvProjDim = innerEmbDim + 2 * (headsKV * headDim);

                // The KV cache is pre-allocated to cacheCapacity, so no position beyond it can ever
                // be roped — size the tables to the session length, not the 131k model maximum.
                ropeMaxSeq = cacheCapacity;

                cs = DeepUnityMeta.Gemma3CS;   // shared dim-agnostic kernel set (see class comment)
                // Keyword state lives on the shared shader asset — one quant mode per session.
                if (quant == LLMQuant.INT8) cs.EnableKeyword("INT8_WEIGHTS"); else cs.DisableKeyword("INT8_WEIGHTS");
                if (quant == LLMQuant.INT4) cs.EnableKeyword("INT4_WEIGHTS"); else cs.DisableKeyword("INT4_WEIGHTS");
                KVQuantUtil.SetKeyword(cs, kvQuant);
                CacheKernelIds();

                weights = new MiniCPM5Weights(paramsPath, quant);
                cache = new MiniCPM5Cache(numLayers, cacheCapacity, headsKV, headDim, kvQuant);

                PrecomputeRoPEAsync();

                // Fixed-size FP32 buffers
                probsBuf = new ComputeBuffer(vocabSize, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(hiddenSize, 4, ComputeBufferType.Structured);
            }

            void CacheKernelIds()
            {
                kEmbedLookup = cs.FindKernel("EmbeddingLookup");
                kRmsNormHidden = cs.FindKernel("RmsNormHidden");
                kSplitQKV = cs.FindKernel("SplitQKV");
                kApplyRope = cs.FindKernel("ApplyRopeSplitHalf");
                kWriteCacheFull = cs.FindKernel("WriteCacheFull");
                kApplyMask = cs.FindKernel("ApplyMask");
                kSoftmaxRows = cs.FindKernel("SoftmaxRows");
                kFlashAttn = cs.FindKernel("FlashAttention");
                kArgMax = cs.FindKernel("ArgMax");
                kSampleToken = cs.FindKernel("SampleToken");
                kCopyBuffer = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAddResidual = cs.FindKernel("AddResidual");
                kQKVProj = cs.FindKernel("QKVProj");
                kAttnScores = cs.FindKernel("ComputeAttentionScores");
                kAttendValues = cs.FindKernel("AttendValues");
                kOProj = cs.FindKernel("OProj");
                kGateUp = cs.FindKernel("GateUp");
                kDown = cs.FindKernel("Down");
                kGateUp1Vec = cs.FindKernel("GateUp1Vec");
                kDown1Vec = cs.FindKernel("Down1Vec");
                kLmHead = cs.FindKernel("LmHeadPredict");
                kLmHead1Vec = cs.FindKernel("LmHeadPredict1Vec");
                // #31 coalesced variants
                kQKVProj1C = cs.FindKernel("QKVProj1VecCoal");
                kOProj1C   = cs.FindKernel("OProj1VecCoal");
                kGateUp1C  = cs.FindKernel("GateUp1VecCoal");
                kDown1C    = cs.FindKernel("Down1VecCoal");
                kLmHead1C  = cs.FindKernel("LmHeadPredict1VecCoal");
                kQKVProjG  = cs.FindKernel("QKVProjGemmCoal");
                kOProjG    = cs.FindKernel("OProjGemmCoal");
                kGateUpG   = cs.FindKernel("GateUpGemmCoal");
                kDownG     = cs.FindKernel("DownGemmCoal");
            }

            // Pack FP16 RoPE caches into uint buffers
            static ComputeBuffer PackedHalfBuf(int halfCount)
            {
                return new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            }

            void PrecomputeRoPEAsync()
            {
                int maxSeq = ropeMaxSeq;
                int hd2 = headDim / 2;
                int hDim = headDim;
                float theta = MiniCPM5Config.ROPE_THETA;
                int packedLen = (maxSeq * hd2) / 2;

                ropeCos = PackedHalfBuf(maxSeq * hd2);
                ropeSin = PackedHalfBuf(maxSeq * hd2);

                _ = Task.Run(() =>
                {
                    uint[] c = new uint[packedLen], s = new uint[packedLen];
                    Parallel.For(0, maxSeq, pos =>
                    {
                        int baseU = pos * (hd2 / 2);
                        for (int j = 0; j < hd2 / 2; j++)
                        {
                            int i0 = 2 * j, i1 = 2 * j + 1;
                            float f0 = 1f / MathF.Pow(theta, 2f * i0 / hDim);
                            float f1 = 1f / MathF.Pow(theta, 2f * i1 / hDim);
                            c[baseU + j] = (uint)F32ToF16(MathF.Cos(pos * f0)) | ((uint)F32ToF16(MathF.Cos(pos * f1)) << 16);
                            s[baseU + j] = (uint)F32ToF16(MathF.Sin(pos * f0)) | ((uint)F32ToF16(MathF.Sin(pos * f1)) << 16);
                        }
                    });
                    _ropeC = c; _ropeS = s;
                    _ropeComputed = true;
                });

                DeepUnityDispatcher.Run(UploadRopeWhenReady());
            }

            IEnumerator UploadRopeWhenReady()
            {
                while (!_ropeComputed) yield return null;
                ropeCos.SetData(_ropeC); ropeSin.SetData(_ropeS);
                _ropeC = _ropeS = null;
                ropeReady = true;
            }

            // Managed IEEE-754 float32 -> float16 (round to nearest), same as Gemma3Model.
            static ushort F32ToF16(float value)
            {
                int i = BitConverter.SingleToInt32Bits(value);
                int s = (i >> 16) & 0x00008000;
                int e = ((i >> 23) & 0x000000ff) - (127 - 15);
                int m = i & 0x007fffff;
                if (e <= 0)
                {
                    if (e < -10) return (ushort)s;
                    m |= 0x00800000;
                    int t = 14 - e;
                    int a = (1 << (t - 1)) - 1;
                    int b = (m >> t) & 1;
                    m = (m + a + b) >> t;
                    return (ushort)(s | m);
                }
                else if (e == 0xff - (127 - 15))
                {
                    if (m == 0) return (ushort)(s | 0x7c00);
                    m >>= 13;
                    return (ushort)(s | 0x7c00 | m | (m == 0 ? 1 : 0));
                }
                else
                {
                    m = m + 0x00000fff + ((m >> 13) & 1);
                    if ((m & 0x00800000) != 0) { m = 0; e += 1; }
                    if (e > 30) return (ushort)(s | 0x7c00);
                    return (ushort)(s | (e << 10) | (m >> 13));
                }
            }

            void Realloc(ref ComputeBuffer buf, int count)
            {
                if (buf != null && buf.count >= count) return;
                buf?.Release();
                buf = new ComputeBuffer(count, 4, ComputeBufferType.Structured);
            }

            void EnsureScratch(int seqLen, int totalKvLen)
            {
                if (seqLen <= curSeqAlloc && totalKvLen <= curKvAlloc) return;
                int sL = Math.Max(seqLen, curSeqAlloc);
                int kL = Math.Max(totalKvLen, curKvAlloc);

                Realloc(ref hiddenBuf, sL * hiddenSize);
                Realloc(ref skipBuf, sL * hiddenSize);
                Realloc(ref normOutBuf, sL * hiddenSize);
                Realloc(ref qkvBuf, sL * qkvProjDim);
                Realloc(ref qBuf, sL * headsQ * headDim);
                Realloc(ref kBuf, sL * headsKV * headDim);
                Realloc(ref vBuf, sL * headsKV * headDim);
                Realloc(ref attnScoresBuf, headsQ * sL * kL);
                Realloc(ref attendedBuf, sL * headsQ * headDim);
                Realloc(ref attnOutBuf, sL * hiddenSize);
                Realloc(ref mlpInterBuf, sL * intermediateSize);
                Realloc(ref tokenIdsBuf, sL);

                curSeqAlloc = sL;
                curKvAlloc = kL;
            }

            void UploadTokens(Tensor ids, int seqLen)
            {
                uint[] arr = new uint[seqLen];
                for (int i = 0; i < seqLen; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);
            }

            static int Div256(int n) => (n + 255) / 256;

            // Quantized modes: bind the scale buffer next to its weight buffer. No-op in FP16.
            void BindScales(int kernel, string name, ComputeBuffer scales)
            {
                if (Quant != LLMQuant.FP16 && scales != null) cs.SetBuffer(kernel, name, scales);
            }

            // INT8 KV only: the attention read kernels dequantize K/V via the per-(token,head)
            // scale/zp buffers (see Gemma3Model.BindKvScaleZp). No-op for FP32/FP16.
            void BindKvScaleZp(int kernel, int li)
            {
                if (KV != KVQuant.INT8) return;
                cs.SetBuffer(kernel, "k_scale_zp", cache.kScaleZp[li]);
                cs.SetBuffer(kernel, "v_scale_zp", cache.vScaleZp[li]);
            }

            // ---- layer dispatch: plain llama block ----
            //   x  = x + O(attn(rope(split(QKV(rms(x, ln1))))))
            //   x  = x + Down(SiLU-GLU(GateUp(rms(x, ln2))))
            // Fewer dispatches than Gemma3 (no qk-norm, no sandwich norms).
            void DispatchLayer(int li, int seqLen, int totalKvLen, bool useCache)
            {
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int kvLen = useCache ? totalKvLen : seqLen;
                int hd2 = headDim / 2;
                int hidTotal = seqLen * hiddenSize;

                // 1. copy hidden → skip
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", hiddenBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                // 2. input layernorm
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.inputLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 3. QKV proj — #31 coalesced decode GEMV / prefill GEMM
                bool coalQ = seqLen == 1 && !ForceLegacyGemv;
                bool gemmQ = seqLen > 1 && !ForceLegacyGemv;
                int kQP = coalQ ? kQKVProj1C : gemmQ ? kQKVProjG : kQKVProj;
                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetInt("qkv_proj_dim", qkvProjDim);
                cs.SetBuffer(kQP, "X", normOutBuf);
                cs.SetBuffer(kQP, "W_QKV", weights.W_QKV[li]);
                BindScales(kQP, "W_QKV_scales", weights.W_QKVScales[li]);
                cs.SetBuffer(kQP, "QKV", qkvBuf);
                if (coalQ)      cs.Dispatch(kQP, (qkvProjDim + GVC_ROWS - 1) / GVC_ROWS, 1, 1);
                else if (gemmQ) cs.Dispatch(kQP, (qkvProjDim + GVC_ROWS - 1) / GVC_ROWS, (seqLen + GMM_TOK - 1) / GMM_TOK, 1);
                else            cs.Dispatch(kQP, 1, (seqLen + 7) / 8, (qkvProjDim + 31) / 32);

                // 4. split QKV
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("qkv_proj_dim", qkvProjDim);
                cs.SetInt("num_heads_q", headsQ);
                cs.SetInt("num_heads_kv", headsKV);
                cs.SetInt("head_dim", headDim);
                cs.SetBuffer(kSplitQKV, "qkv_packed", qkvBuf);
                cs.SetBuffer(kSplitQKV, "split_q", qBuf);
                cs.SetBuffer(kSplitQKV, "split_k", kBuf);
                cs.SetBuffer(kSplitQKV, "split_v", vBuf);
                cs.Dispatch(kSplitQKV, Div256(seqLen * qkvProjDim), 1, 1);

                // 5. RoPE Q (in place — llama has no qk-norm)
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("head_dim", headDim);
                cs.SetInt("rope_num_heads", headsQ);
                cs.SetInt("position_offset", cacheLen);
                cs.SetBuffer(kApplyRope, "rope_buf", qBuf);
                cs.SetBuffer(kApplyRope, "rope_cos", ropeCos);
                cs.SetBuffer(kApplyRope, "rope_sin", ropeSin);
                cs.Dispatch(kApplyRope, (seqLen * headsQ * hd2 + 127) / 128, 1, 1);

                // 6. RoPE K (in place)
                cs.SetInt("rope_num_heads", headsKV);
                cs.SetBuffer(kApplyRope, "rope_buf", kBuf);
                cs.Dispatch(kApplyRope, (seqLen * headsKV * hd2 + 127) / 128, 1, 1);

                ComputeBuffer kForAttn, vForAttn;
                if (useCache)
                {
                    cs.SetInt("seq_len", seqLen);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetInt("cache_len", cacheLen);
                    bool kvInt8 = KV == KVQuant.INT8;
                    cs.SetBuffer(kWriteCacheFull, "kv_new", kBuf);
                    cs.SetBuffer(kWriteCacheFull, "kv_cache", cache.kCaches[li]);
                    if (kvInt8)
                    {
                        cs.SetBuffer(kWriteCacheFull, "kv_scale_zp_w", cache.kScaleZp[li]);
                        cs.Dispatch(kWriteCacheFull, seqLen * headsKV, 1, 1);
                    }
                    else cs.Dispatch(kWriteCacheFull, Div256(KVQuantUtil.WriteUnits(seqLen * headsKV * headDim, KV)), 1, 1);
                    cs.SetBuffer(kWriteCacheFull, "kv_new", vBuf);
                    cs.SetBuffer(kWriteCacheFull, "kv_cache", cache.vCaches[li]);
                    if (kvInt8)
                    {
                        cs.SetBuffer(kWriteCacheFull, "kv_scale_zp_w", cache.vScaleZp[li]);
                        cs.Dispatch(kWriteCacheFull, seqLen * headsKV, 1, 1);
                    }
                    else cs.Dispatch(kWriteCacheFull, Div256(KVQuantUtil.WriteUnits(seqLen * headsKV * headDim, KV)), 1, 1);
                    kForAttn = cache.kCaches[li];
                    vForAttn = cache.vCaches[li];
                }
                else
                {
                    kForAttn = kBuf;
                    vForAttn = vBuf;
                }

                if (UseFlashAttention && headDim <= 256)
                {
                    // fused attention (one threadgroup per query x head); sliding_window_size=0 =
                    // full causal attention on every layer
                    cs.SetInt("seq_len_q", seqLen);
                    cs.SetInt("seq_len_k", kvLen);
                    cs.SetInt("num_heads_q", headsQ);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetInt("sliding_window_size", 0);
                    cs.SetInt("bidirectional", 0); // causal LM
                    cs.SetFloat("scale", attnScaling);
                    cs.SetBuffer(kFlashAttn, "Q", qBuf);
                    cs.SetBuffer(kFlashAttn, "K", kForAttn);
                    cs.SetBuffer(kFlashAttn, "V", vForAttn);
                    BindKvScaleZp(kFlashAttn, li);
                    cs.SetBuffer(kFlashAttn, "AttendedValues", attendedBuf);
                    cs.Dispatch(kFlashAttn, seqLen, headsQ, 1);
                }
                else
                {
                    // legacy 4-dispatch path (scores / mask / softmax / attend)
                    cs.SetInt("batch_size", 1);
                    cs.SetInt("sequence_length_q", seqLen);
                    cs.SetInt("sequence_length_k", kvLen);
                    cs.SetInt("num_heads_q", headsQ);
                    cs.SetInt("num_heads_kv", headsKV);
                    cs.SetInt("head_dim", headDim);
                    cs.SetFloat("scale", attnScaling);
                    cs.SetBuffer(kAttnScores, "Q", qBuf);
                    cs.SetBuffer(kAttnScores, "K", kForAttn);
                    BindKvScaleZp(kAttnScores, li);
                    cs.SetBuffer(kAttnScores, "AttentionWeights", attnScoresBuf);
                    cs.Dispatch(kAttnScores, (seqLen + 3) / 4, (kvLen + 31) / 32, (headsQ + 3) / 4);

                    cs.SetInt("seq_len_q", seqLen);
                    cs.SetInt("seq_len_k", kvLen);
                    cs.SetInt("num_heads_q", headsQ);
                    cs.SetInt("sliding_window_size", 0);
                    cs.SetInt("bidirectional", 0); // causal LM
                    cs.SetBuffer(kApplyMask, "AttentionWeights", attnScoresBuf);
                    cs.Dispatch(kApplyMask, (kvLen + 15) / 16, (headsQ * seqLen + 15) / 16, 1);

                    cs.SetInt("seq_len_q", seqLen);
                    cs.SetInt("seq_len_k", kvLen);
                    cs.SetBuffer(kSoftmaxRows, "AttentionWeights", attnScoresBuf);
                    cs.Dispatch(kSoftmaxRows, Div256(headsQ * seqLen), 1, 1);

                    cs.SetInt("sequence_length_v", kvLen);
                    cs.SetBuffer(kAttendValues, "AttentionWeights", attnScoresBuf);
                    cs.SetBuffer(kAttendValues, "V", vForAttn);
                    BindKvScaleZp(kAttendValues, li);
                    cs.SetBuffer(kAttendValues, "AttendedValues", attendedBuf);
                    cs.Dispatch(kAttendValues, (headDim + 63) / 64, (seqLen + 3) / 4, (headsQ + 3) / 4);
                }

                // 7. O proj — #31 coalesced decode GEMV / prefill GEMM
                bool coalO = seqLen == 1 && !ForceLegacyGemv;
                bool gemmO = seqLen > 1 && !ForceLegacyGemv;
                int kOP = coalO ? kOProj1C : gemmO ? kOProjG : kOProj;
                cs.SetInt("inner_embedding_dim", innerEmbDim);
                cs.SetInt("embedding_dim", hiddenSize);
                cs.SetBuffer(kOP, "AttendedValues", attendedBuf);
                cs.SetBuffer(kOP, "W_O", weights.W_O[li]);
                BindScales(kOP, "W_O_scales", weights.W_OScales[li]);
                cs.SetBuffer(kOP, "O", attnOutBuf);
                if (coalO)      cs.Dispatch(kOP, (hiddenSize + GVC_ROWS - 1) / GVC_ROWS, 1, 1);
                else if (gemmO) cs.Dispatch(kOP, (hiddenSize + GVC_ROWS - 1) / GVC_ROWS, (seqLen + GMM_TOK - 1) / GMM_TOK, 1);
                else            cs.Dispatch(kOP, 1, (seqLen + 3) / 4, (hiddenSize + 31) / 32);

                // 8. residual: attnOut += skip   (llama adds straight after o_proj — no post norm here)
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kAddResidual, "buf_a", attnOutBuf);
                cs.SetBuffer(kAddResidual, "buf_b", skipBuf);
                cs.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);

                // 9. copy attnOut → skip (second residual stream)
                cs.SetBuffer(kCopyBuffer, "buf_a", skipBuf);
                cs.SetBuffer(kCopyBuffer, "buf_b", attnOutBuf);
                cs.Dispatch(kCopyBuffer, Div256(hidTotal), 1, 1);

                // 10. post-attention layernorm (llama's pre-MLP norm) → hiddenBuf
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", attnOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.postAttnLnGamma[li]);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                // 11-12. MLP (SiLU GLU: activation_type=0) — #31 coalesced GEMVs/GEMMs
                bool vec1 = seqLen == 1;
                bool coalMlp = vec1 && !ForceLegacyGemv;
                bool gemmMlp = !vec1 && !ForceLegacyGemv;
                int kGU = coalMlp ? kGateUp1C : gemmMlp ? kGateUpG : vec1 ? kGateUp1Vec : kGateUp;
                int kDN = coalMlp ? kDown1C   : gemmMlp ? kDownG   : vec1 ? kDown1Vec   : kDown;
                int mlpTokTiles = (seqLen + GMM_TOK - 1) / GMM_TOK;

                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("intermediate_size", intermediateSize);
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("activation_type", 0);   // 0 = SiLU (Gemma3 passes 1 = GELU)
                cs.SetBuffer(kGU, "input", hiddenBuf);
                cs.SetBuffer(kGU, "mlp_weights", weights.mlpWeights[li]);
                BindScales(kGU, "mlp_scales", weights.mlpScales[li]);
                cs.SetBuffer(kGU, "intermediate", mlpInterBuf);
                if (coalMlp)      cs.Dispatch(kGU, (intermediateSize + GVC_ROWS - 1) / GVC_ROWS, 1, 1);
                else if (gemmMlp) cs.Dispatch(kGU, (intermediateSize + GVC_ROWS - 1) / GVC_ROWS, mlpTokTiles, 1);
                else if (vec1)    cs.Dispatch(kGU, (intermediateSize + 255) / 256, 1, 1);
                else              cs.Dispatch(kGU, (intermediateSize + 63) / 64, (seqLen + 3) / 4, 1);

                cs.SetBuffer(kDN, "input", hiddenBuf);
                cs.SetBuffer(kDN, "mlp_weights", weights.mlpWeights[li]);
                BindScales(kDN, "mlp_scales", weights.mlpScales[li]);
                cs.SetBuffer(kDN, "intermediate", mlpInterBuf);
                if (coalMlp)      cs.Dispatch(kDN, (hiddenSize + GVC_ROWS - 1) / GVC_ROWS, 1, 1);
                else if (gemmMlp) cs.Dispatch(kDN, (hiddenSize + GVC_ROWS - 1) / GVC_ROWS, mlpTokTiles, 1);
                else if (vec1)    cs.Dispatch(kDN, (intermediateSize + 319) / 320, 1, 1);
                else              cs.Dispatch(kDN, (hiddenSize + 31) / 32, (seqLen + 3) / 4, 1);

                // 13. residual: hidden += skip → next layer input already in hiddenBuf
                cs.SetInt("buffer_size", hidTotal);
                cs.SetBuffer(kAddResidual, "buf_a", hiddenBuf);
                cs.SetBuffer(kAddResidual, "buf_b", skipBuf);
                cs.Dispatch(kAddResidual, Div256(hidTotal), 1, 1);
            }

#if UNITY_EDITOR
            /// <summary>#31 probes: see Gemma3Model.LoadBlockingForProbe (same pattern).</summary>
            public void LoadBlockingForProbe()
            {
                var pump = weights.EditorUploadPump();
                while (pump.MoveNext()) System.Threading.Thread.Sleep(0);   // yields = IO reads in flight
                var rope = UploadRopeWhenReady();
                while (rope.MoveNext()) System.Threading.Thread.Sleep(1);   // waits the background rope compute
            }
#endif

            public void Forward(Tensor input_ids, bool useCache, bool lastPosOnly)
            {
                int seqLen = input_ids.Size(-1);
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int totalKvLen = cacheLen + seqLen;

                EnsureScratch(seqLen, totalKvLen);
                UploadTokens(input_ids, seqLen);

                DispatchEmbed(seqLen);

                for (int i = 0; i < numLayers; i++)
                    DispatchLayer(i, seqLen, totalKvLen, useCache);

                if (useCache) cache.CachedTokenCount += seqLen;

                if (lastPosOnly) DispatchFinalLast(seqLen);
                else DispatchFinalAll(seqLen);
            }

            public IEnumerator ForwardYielding(Tensor input_ids, bool useCache, bool lastPosOnly)
            {
                int seqLen = input_ids.Size(-1);
                int cacheLen = useCache ? cache.CachedTokenCount : 0;
                int totalKvLen = cacheLen + seqLen;

                EnsureScratch(seqLen, totalKvLen);
                UploadTokens(input_ids, seqLen);

                DispatchEmbed(seqLen);
                yield return null;

                for (int i = 0; i < numLayers; i++)
                {
                    DispatchLayer(i, seqLen, totalKvLen, useCache);
                    yield return null;
                }

                if (useCache) cache.CachedTokenCount += seqLen;

                if (lastPosOnly) DispatchFinalLast(seqLen);
                else DispatchFinalAll(seqLen);
                yield return null;
            }

            void DispatchEmbed(int seqLen)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("embed_scale", 1f);   // llama: no sqrt(hidden) embedding scale
                cs.SetBuffer(kEmbedLookup, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbedLookup, "embed_weights", weights.embed);
                cs.SetBuffer(kEmbedLookup, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbedLookup, Div256(seqLen * hiddenSize), 1, 1);
            }

            void DispatchFinalLast(int seqLen)
            {
                cs.SetInt("buffer_size", hiddenSize);
                cs.SetInt("copy_src_offset", (seqLen - 1) * hiddenSize);
                cs.SetBuffer(kCopySlice, "buf_a", lastHiddenBuf);
                cs.SetBuffer(kCopySlice, "buf_b", hiddenBuf);
                cs.Dispatch(kCopySlice, Div256(hiddenSize), 1, 1);

                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", lastHiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normSingleBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsNormHidden, 1, 1, 1);

                Realloc(ref logitsBuf, vocabSize);
                int kLH = ForceLegacyGemv ? kLmHead1Vec : kLmHead1C;   // #31 coalesced decode GEMV
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("vocab_size", vocabSize);
                cs.SetBuffer(kLH, "lm_weights", weights.lmHead);   // UNTIED head
                cs.SetBuffer(kLH, "lm_input", normSingleBuf);
                cs.SetBuffer(kLH, "lm_output", logitsBuf);
                if (ForceLegacyGemv) cs.Dispatch(kLH, (vocabSize + 511) / 512, 1, 1);
                else                 cs.Dispatch(kLH, (vocabSize + GVC_ROWS - 1) / GVC_ROWS, 1, 1);
            }

            void DispatchFinalAll(int seqLen)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetFloat("norm_eps", rmsEps);
                cs.SetBuffer(kRmsNormHidden, "norm_input", hiddenBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_output", normOutBuf);
                cs.SetBuffer(kRmsNormHidden, "norm_gamma", weights.finalNormGamma);
                cs.Dispatch(kRmsNormHidden, Div256(seqLen), 1, 1);

                Realloc(ref logitsBuf, seqLen * vocabSize);
                bool v1 = seqLen == 1;
                int k = v1 ? (ForceLegacyGemv ? kLmHead1Vec : kLmHead1C) : kLmHead;   // #31 coalesced decode GEMV
                cs.SetInt("batch_size", 1);
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", hiddenSize);
                cs.SetInt("vocab_size", vocabSize);
                cs.SetBuffer(k, "lm_weights", weights.lmHead);   // UNTIED head
                cs.SetBuffer(k, "lm_input", normOutBuf);
                cs.SetBuffer(k, "lm_output", logitsBuf);
                if (!v1) cs.Dispatch(k, (vocabSize + 31) / 32, (seqLen + 7) / 8, 1);
                else if (ForceLegacyGemv) cs.Dispatch(k, (vocabSize + 511) / 512, 1, 1);
                else cs.Dispatch(k, (vocabSize + GVC_ROWS - 1) / GVC_ROWS, 1, 1);
            }

            // Queues the sampler kernel; the chosen token id lands in argmaxBuf on the GPU.
            void DispatchSampleKernels(float temperature, int topK, float topP, float minP)
            {
                cs.SetInt("vocab_size", vocabSize);
                if (temperature == 0f)
                {
                    cs.SetBuffer(kArgMax, "logits_buf", logitsBuf);
                    cs.SetBuffer(kArgMax, "argmax_result", argmaxBuf);
                    cs.Dispatch(kArgMax, 1, 1, 1);
                }
                else
                {
                    cs.SetFloat("temperature", temperature);
                    cs.SetInt("top_k_val", topK);
                    cs.SetFloat("top_p_val", topP);
                    cs.SetFloat("min_p_val", minP);
                    cs.SetInt("rng_seed", UnityEngine.Random.Range(int.MinValue, int.MaxValue));
                    cs.SetBuffer(kSampleToken, "logits_buf", logitsBuf);
                    cs.SetBuffer(kSampleToken, "probs_buf", probsBuf);
                    cs.SetBuffer(kSampleToken, "argmax_result", argmaxBuf);
                    cs.Dispatch(kSampleToken, 1, 1, 1);
                }
            }

            public int SampleGreedy() => Sample(0f, 0, 1f, 0f);

            public int SampleStochastic(float temperature, int topK, float topP, float minP)
                => Sample(temperature, topK, topP, minP);

            // Synchronous sample: blocks until every queued GPU dispatch has finished.
            public int Sample(float temperature, int topK, float topP, float minP)
            {
                DispatchSampleKernels(temperature, topK, topP, minP);
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                return (int)r[0];
            }

            // Async sample via AsyncGPUReadback; writes result[0].
            public IEnumerator SampleYielding(float temperature, int topK, float topP, float minP, int[] result)
            {
                DispatchSampleKernels(temperature, topK, topP, minP);
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(argmaxBuf);
                    while (!req.done) yield return null;
                    if (!req.hasError)
                    {
                        result[0] = (int)req.GetData<uint>()[0];
                        yield break;
                    }
                }
                uint[] r = new uint[1]; argmaxBuf.GetData(r); // fallback: sync readback
                result[0] = (int)r[0];
            }

            public Tensor ReadLogits(int seqLen)
            {
                return seqLen == 1
                    ? Tensor.Constant(logitsBuf, vocabSize)
                    : Tensor.Constant(logitsBuf, seqLen, vocabSize);
            }

            public void ResetCache() => cache.Reset();

            // Same shader asset as Gemma3 — one prewarm covers both families.
            public static IEnumerator PrewarmKernels() => Gemma3Modeling.Gemma3Model.PrewarmKernels();

            bool _warmedUp;

            // Kernel compiles + throwaway forwards behind the loading screen (see Gemma3Model.Warmup).
            public IEnumerator Warmup()
            {
                if (_warmedUp) yield break;

                var pk = PrewarmKernels();              // no weights needed — overlaps the upload
                while (pk.MoveNext()) yield return pk.Current;

                while (!IsReady) yield return null;

                int[] tok = new int[1];
                foreach (int n in new[] { 4, 1 })
                {
                    var e = ForwardYielding(Tensor.Constant(new float[n]), useCache: true, lastPosOnly: true);
                    while (e.MoveNext()) yield return e.Current;
                    var s = SampleYielding(1f, 64, 0.95f, 0f, tok);
                    while (s.MoveNext()) yield return s.Current;
                }
                var g = SampleYielding(0f, 0, 1f, 0f, tok); // greedy (temperature==0) path
                while (g.MoveNext()) yield return g.Current;
                yield return null;

                ResetCache(); // undo the warmup's cache writes
                _warmedUp = true;
            }

            public void Dispose()
            {
                weights?.Dispose(); cache?.Dispose();
                ropeCos?.Release(); ropeSin?.Release();
                hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release();
                qkvBuf?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release();
                attnScoresBuf?.Release(); attendedBuf?.Release(); attnOutBuf?.Release();
                mlpInterBuf?.Release(); logitsBuf?.Release(); probsBuf?.Release();
                argmaxBuf?.Release(); tokenIdsBuf?.Release();
                lastHiddenBuf?.Release(); normSingleBuf?.Release();
            }
        }
    }
}
