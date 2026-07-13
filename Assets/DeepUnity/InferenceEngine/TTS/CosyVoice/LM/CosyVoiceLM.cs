using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // CosyVoice3LM — Qwen2.5-0.5B backbone + untied speech head, full-GPU
        // (CosyVoiceLMCS.compute). Port spec: TTS/CosyVoice/SPEC.md §1.
        //
        // Prefill embeds = [speech_emb(sos 6561) | embed_tokens(prompt_text ++ text) |
        // speech_emb(task 6563) | speech_emb(prompt_speech_tokens)] — then AR decode:
        // hidden -> RMSNorm -> llm_decoder(6761, no bias) -> logits READBACK (27KB) ->
        // CPU RAS sampling (utils/common.py::ras_sampling port); sampled ids re-embed via
        // speech_embedding. Stop = any id >= 6561, suppressed below 2x textLen, cap 20x.
        //
        // Block: RMSNorm(gamma) -> q/k/v proj WITH bias -> full RoPE theta 1e6 (CPU tables)
        // -> fp16 KV cache -> GQA FlashAttention(14Q/2KV, hd64) -> o_proj (no bias) ->
        // +res -> RMSNorm -> SiLU gate/up/down 4864 -> +res. No QK-norm, no attn gate.
        public class CosyVoiceLM : IDisposable
        {
            const int H = CosyVoiceConfig.LM_HIDDEN;          // 896
            const int LAYERS = CosyVoiceConfig.LM_LAYERS;     // 24
            const int HQ = CosyVoiceConfig.LM_HEADS_Q;        // 14
            const int HKV = CosyVoiceConfig.LM_HEADS_KV;      // 2
            const int HD = CosyVoiceConfig.LM_HEAD_DIM;       // 64
            const int MLP_DIM = CosyVoiceConfig.LM_MLP;       // 4864
            const int VOCAB = CosyVoiceConfig.SPEECH_EMB_ROWS;// 6761
            const int EMB_SHARD = 9496;                       // embed_tokens rows per part (151936/16)
            readonly int CHUNK;                               // prefill positions per burst (default 8;
                                                              // probes pass >=seq for full-seq taps)

            readonly ComputeShader cs;
            readonly CosyVoiceWeights weights;
            readonly int cacheCapacity;

            /// <summary>Per-stage tap (name, buffer, elemCount) for the parity probe.</summary>
            public Action<string, ComputeBuffer, int> DebugTap;

            int kEmbed, kRmsNorm, kQProj, kKProj, kVProj, kRope, kWriteCache, kFlashAttn,
                kOProj, kGateUp, kDown, kLmHead, kZero, kCopy, kCopySlice, kAdd, kSampleRas,
                kDecQKV, kDecRope, kDecOProj, kDecGateUp, kDecDown, kDecHead;

            /// <summary>A6-max Phase 6b fused decode path (default ON): 6 dispatches/layer + 1
            /// head (145/token, was 411), split-k GEMVs (32-64 lanes per output row, coalesced
            /// W streams, fixed groupshared reduction trees) with fused RMSNorms and residual
            /// epilogues. DETERMINISTIC (same input == same output every run) but NOT
            /// bit-identical to the legacy kernels — accumulation order differs (~ulp noise).
            /// Phase-6b contract: A3 logp corr > 0.999 + argmax MATCH + per-layer corr >=
            /// 0.99999; sampled token sequences may diverge from pre-6b runs. false = the
            /// legacy per-op kernels for bisection. Prefill always uses the original kernels.</summary>
            public static bool FastLM = true;

            /// <summary>GPU-side RAS sampling (SampleTokenRAS kernel + 4-byte async readback) —
            /// removes the per-token 27 KB logits readback entirely. NOT bit-equal to the CPU
            /// sampler (different RNG stream), so it defaults OFF: the parity-exact default is
            /// the CPU RAS fed by a non-blocking async logits readback.</summary>
            public bool GpuSampler = false;
            // Async readback spin cap: probes pump enumerators without frames, so req.done may
            // never flip there — after the cap we hard-wait (same fence the old GetData paid).
            const int ASYNC_SPIN_CAP = 240;

            readonly ComputeBuffer[] kCaches = new ComputeBuffer[LAYERS];
            readonly ComputeBuffer[] vCaches = new ComputeBuffer[LAYERS];
            public int CachedTokenCount { get; private set; }

            // Reusable per-voice KV prefix = [sos | prompt_text...] — NOTE for CosyVoice3 the
            // utterance text sits BETWEEN prompt_text and task_id, so only [sos|prompt_text]
            // is text-independent (SPEC §1). Wire-up lands at A4; the hooks mirror T3Model.
            public int PrefixTokenCount { get; private set; }
            public void SavePrefix() => PrefixTokenCount = CachedTokenCount;
            public void RestoreToPrefix() => CachedTokenCount = PrefixTokenCount;

            ComputeBuffer embedsBuf, hiddenBuf, skipBuf, normOutBuf, attnOutBuf;
            ComputeBuffer qBuf, kBuf, vBuf, attendedBuf, mlpInterBuf;
            ComputeBuffer lastHiddenBuf, normSingleBuf, logitsBuf;
            ComputeBuffer tokenIdsBuf, ropeCosBuf, ropeSinBuf;
            ComputeBuffer probsBuf, argmaxBuf, rasHistBuf;   // GPU RAS sampler scratch

            // Per-utterance cached weight handles (resolved in BuildPrefillEmbeds — CosyVoiceFlow
            // pattern; safe across Defetch/reload, which recreates buffers between utterances).
            // The old per-token path paid ~17 string-concats + dictionary lookups x 24 layers
            // PER DECODED TOKEN — a real slice of the ~19 ms/token CPU issue cost. Bit-exact.
            ComputeBuffer[] lwQ, lwK, lwV, lwO, lwG, lwU, lwD, lwLn1, lwLn2, lbQ, lbK, lbV;
            ComputeBuffer[] lsQ, lsK, lsV, lsO, lsG, lsU, lsD;   // int8 scales (null slots when fp16)
            ComputeBuffer wNormG, wDecoder, wSpeechEmb;
            readonly int[] decodeTok = new int[1];               // per-token embed without allocs

            public bool IsReady => weights.IsReady;
            public float PrefillMs { get; private set; }
            readonly bool isInt8;   // manifest carries .scales siblings -> INT8_WEIGHTS variant

            // prefillChunk 8 -> 64 (A6-max Phase 4): the 155-token prefill at burst 8 was a
            // 1.52 s TTFA floor — 20 bursts of badly underfilled GEMMs. Burst size cannot change
            // results (per-query causal attention + per-row matmuls; the A3 bisect mode already
            // runs chunk 256 against the same dumps), it only widens the per-burst GEMMs.
            public CosyVoiceLM(CosyVoiceWeights weights, int cacheCapacity = 4096, int prefillChunk = 64)
            {
                this.weights = weights;
                this.cacheCapacity = cacheCapacity;
                CHUNK = prefillChunk;
                cs = DeepUnityMeta.CosyVoiceLMCS;
                cs.EnableKeyword("KV_FP16");
                cs.DisableKeyword("KV_INT8");
                isInt8 = weights.Has("llm/layers.0.self_attn.q_proj.weight.scales");
                if (isInt8) cs.EnableKeyword("INT8_WEIGHTS"); else cs.DisableKeyword("INT8_WEIGHTS");

                kEmbed = cs.FindKernel("EmbeddingLookup");
                kRmsNorm = cs.FindKernel("RmsNormHidden");
                kQProj = cs.FindKernel("QProjBias");
                kKProj = cs.FindKernel("KProjBias");
                kVProj = cs.FindKernel("VProjBias");
                kRope = cs.FindKernel("ApplyRopePartial");
                kWriteCache = cs.FindKernel("WriteCacheFull");
                kFlashAttn = cs.FindKernel("FlashAttention");
                kOProj = cs.FindKernel("OProj");
                kGateUp = cs.FindKernel("GateUp");
                kDown = cs.FindKernel("Down");
                kLmHead = cs.FindKernel("LmHeadPredict1Vec");
                kZero = cs.FindKernel("ZeroBuffer");
                kCopy = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAdd = cs.FindKernel("AddResidual");
                kSampleRas = cs.FindKernel("SampleTokenRAS");
                kDecQKV = cs.FindKernel("DecQKV");
                kDecRope = cs.FindKernel("DecRopeCache");
                kDecOProj = cs.FindKernel("DecOProjRes");
                kDecGateUp = cs.FindKernel("DecGateUp");
                kDecDown = cs.FindKernel("DecDownRes");
                kDecHead = cs.FindKernel("DecHead");

                for (int i = 0; i < LAYERS; i++)
                {
                    kCaches[i] = new ComputeBuffer(cacheCapacity * HKV * HD / 2, 4, ComputeBufferType.Structured);
                    vCaches[i] = new ComputeBuffer(cacheCapacity * HKV * HD / 2, 4, ComputeBufferType.Structured);
                }

                embedsBuf = new ComputeBuffer(cacheCapacity * H, 4, ComputeBufferType.Structured);
                hiddenBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                skipBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                normOutBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                attnOutBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                qBuf = new ComputeBuffer(CHUNK * HQ * HD, 4, ComputeBufferType.Structured);
                kBuf = new ComputeBuffer(CHUNK * HKV * HD, 4, ComputeBufferType.Structured);
                vBuf = new ComputeBuffer(CHUNK * HKV * HD, 4, ComputeBufferType.Structured);
                attendedBuf = new ComputeBuffer(CHUNK * HQ * HD, 4, ComputeBufferType.Structured);
                mlpInterBuf = new ComputeBuffer(CHUNK * MLP_DIM, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(H, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(H, 4, ComputeBufferType.Structured);
                logitsBuf = new ComputeBuffer(VOCAB, 4, ComputeBufferType.Structured);
                tokenIdsBuf = new ComputeBuffer(1024, 4, ComputeBufferType.Structured);
                probsBuf = new ComputeBuffer(VOCAB, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                rasHistBuf = new ComputeBuffer(CosyVoiceConfig.RAS_WIN_SIZE, 4, ComputeBufferType.Structured);

                BuildRopeTables();
            }

            static int Div256(int n) => (n + 255) / 256;

            public void ResetCache() { CachedTokenCount = 0; PrefixTokenCount = 0; }

            ComputeBuffer Sc(string tensor)   // int8 scale sibling, null when fp16
                => isInt8 && weights.Has(tensor + ".scales") ? weights.Get(tensor + ".scales") : null;

            void ResolveWeights()
            {
                if (lwQ == null)
                {
                    lwQ = new ComputeBuffer[LAYERS]; lwK = new ComputeBuffer[LAYERS]; lwV = new ComputeBuffer[LAYERS];
                    lwO = new ComputeBuffer[LAYERS]; lwG = new ComputeBuffer[LAYERS]; lwU = new ComputeBuffer[LAYERS];
                    lwD = new ComputeBuffer[LAYERS]; lwLn1 = new ComputeBuffer[LAYERS]; lwLn2 = new ComputeBuffer[LAYERS];
                    lbQ = new ComputeBuffer[LAYERS]; lbK = new ComputeBuffer[LAYERS]; lbV = new ComputeBuffer[LAYERS];
                    lsQ = new ComputeBuffer[LAYERS]; lsK = new ComputeBuffer[LAYERS]; lsV = new ComputeBuffer[LAYERS];
                    lsO = new ComputeBuffer[LAYERS]; lsG = new ComputeBuffer[LAYERS]; lsU = new ComputeBuffer[LAYERS];
                    lsD = new ComputeBuffer[LAYERS];
                }
                for (int li = 0; li < LAYERS; li++)
                {
                    string lp = $"llm/layers.{li}.";
                    lwQ[li] = weights.Get(lp + "self_attn.q_proj.weight"); lbQ[li] = weights.Get(lp + "self_attn.q_proj.bias");
                    lwK[li] = weights.Get(lp + "self_attn.k_proj.weight"); lbK[li] = weights.Get(lp + "self_attn.k_proj.bias");
                    lwV[li] = weights.Get(lp + "self_attn.v_proj.weight"); lbV[li] = weights.Get(lp + "self_attn.v_proj.bias");
                    lwO[li] = weights.Get(lp + "self_attn.o_proj.weight");
                    lwG[li] = weights.Get(lp + "mlp.gate_proj.weight");
                    lwU[li] = weights.Get(lp + "mlp.up_proj.weight");
                    lwD[li] = weights.Get(lp + "mlp.down_proj.weight");
                    lwLn1[li] = weights.Get(lp + "input_layernorm.weight");
                    lwLn2[li] = weights.Get(lp + "post_attention_layernorm.weight");
                    lsQ[li] = Sc(lp + "self_attn.q_proj.weight"); lsK[li] = Sc(lp + "self_attn.k_proj.weight");
                    lsV[li] = Sc(lp + "self_attn.v_proj.weight"); lsO[li] = Sc(lp + "self_attn.o_proj.weight");
                    lsG[li] = Sc(lp + "mlp.gate_proj.weight"); lsU[li] = Sc(lp + "mlp.up_proj.weight");
                    lsD[li] = Sc(lp + "mlp.down_proj.weight");
                }
                wNormG = weights.Get("llm/norm.weight");
                wDecoder = weights.Get("llm/llm_decoder");
                wSpeechEmb = weights.Get("llm/speech_embedding");
            }

            // Full RoPE theta 1e6, split-half convention (HF Qwen2): cos/sin [maxSeq, 32] fp16.
            void BuildRopeTables()
            {
                int hd2 = HD / 2;
                uint[] cosW = new uint[(cacheCapacity * hd2 + 1) / 2];
                uint[] sinW = new uint[(cacheCapacity * hd2 + 1) / 2];
                for (int pos = 0; pos < cacheCapacity; pos++)
                    for (int i = 0; i < hd2; i++)
                    {
                        double f = pos * Math.Pow(CosyVoiceConfig.LM_ROPE_THETA, -2.0 * i / HD);
                        int idx = pos * hd2 + i;
                        uint c = Mathf.FloatToHalf((float)Math.Cos(f));
                        uint s = Mathf.FloatToHalf((float)Math.Sin(f));
                        if ((idx & 1) == 0) { cosW[idx >> 1] |= c; sinW[idx >> 1] |= s; }
                        else { cosW[idx >> 1] |= c << 16; sinW[idx >> 1] |= s << 16; }
                    }
                ropeCosBuf = new ComputeBuffer(cosW.Length, 4, ComputeBufferType.Structured);
                ropeSinBuf = new ComputeBuffer(sinW.Length, 4, ComputeBufferType.Structured);
                ropeCosBuf.SetData(cosW);
                ropeSinBuf.SetData(sinW);
            }

            void UploadTokens(int[] ids)
            {
                uint[] arr = new uint[ids.Length];
                for (int i = 0; i < ids.Length; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);
            }

            void EmbedSegment(string embName, int[] tokens, int dstElemOffset, ComputeBuffer dst)
                => EmbedSegmentW(weights.Get(embName), tokens, dstElemOffset, dst);

            void EmbedSegmentW(ComputeBuffer emb, int[] tokens, int dstElemOffset, ComputeBuffer dst)
            {
                UploadTokens(tokens);
                cs.SetInt("seq_len", tokens.Length);
                cs.SetInt("hidden_size", H);
                cs.SetInt("embed_dst_offset", dstElemOffset);
                cs.SetBuffer(kEmbed, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbed, "embed_weights", emb);
                cs.SetBuffer(kEmbed, "embed_output", dst);
                cs.Dispatch(kEmbed, Div256(tokens.Length * H), 1, 1);
            }

            /// <summary>[sos | embed_tokens(text — caller pre-concats prompt_text) | task_id |
            /// prompt speech tokens] into embedsBuf; returns total length.</summary>
            public int BuildPrefillEmbeds(int[] textTokens, int[] promptSpeechTokens)
            {
                int L = 1 + textTokens.Length + 1 + promptSpeechTokens.Length;
                if (L > cacheCapacity)
                    throw new Exception($"CosyVoice LM prefill too long ({L} > {cacheCapacity}).");

                ResolveWeights();   // per-utterance handle cache for the decode hot path
                EmbedSegment("llm/speech_embedding", new[] { CosyVoiceConfig.SOS_TOKEN }, 0, embedsBuf);
                // text: 16-sharded embed_tokens — group consecutive same-shard ids
                int i = 0;
                while (i < textTokens.Length)
                {
                    int shard = textTokens[i] / EMB_SHARD;
                    int j = i;
                    while (j < textTokens.Length && textTokens[j] / EMB_SHARD == shard) j++;
                    int[] local = new int[j - i];
                    for (int k = 0; k < local.Length; k++) local[k] = textTokens[i + k] % EMB_SHARD;
                    EmbedSegment($"llm/embed_tokens/part_{shard}", local, (1 + i) * H, embedsBuf);
                    i = j;
                }
                EmbedSegment("llm/speech_embedding", new[] { CosyVoiceConfig.TASK_ID_TOKEN },
                             (1 + textTokens.Length) * H, embedsBuf);
                EmbedSegment("llm/speech_embedding", promptSpeechTokens,
                             (2 + textTokens.Length) * H, embedsBuf);
                return L;
            }

            void RmsNormW(ComputeBuffer gamma, ComputeBuffer input, ComputeBuffer output, int seqLen)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", H);
                cs.SetFloat("norm_eps", CosyVoiceConfig.LM_RMS_EPS);
                cs.SetBuffer(kRmsNorm, "norm_input", input);
                cs.SetBuffer(kRmsNorm, "norm_output", output);
                cs.SetBuffer(kRmsNorm, "norm_gamma", gamma);
                cs.Dispatch(kRmsNorm, Div256(seqLen), 1, 1);
            }

            void CopyOp(ComputeBuffer dst, ComputeBuffer src, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kCopy, "buf_a", dst); cs.SetBuffer(kCopy, "buf_b", src);
                cs.Dispatch(kCopy, Div256(count), 1, 1);
            }

            void AddOp(ComputeBuffer dstA, ComputeBuffer srcB, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kAdd, "buf_a", dstA); cs.SetBuffer(kAdd, "buf_b", srcB);
                cs.Dispatch(kAdd, Div256(count), 1, 1);
            }

            void RopeOp(ComputeBuffer buf, int seqLen, int numHeads, int positionOffset)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("rope_num_heads", numHeads);
                cs.SetInt("rope_rot_dim", HD);           // full RoPE
                cs.SetInt("head_dim", HD);
                cs.SetInt("position_offset", positionOffset);
                cs.SetBuffer(kRope, "rope_buf", buf);
                cs.SetBuffer(kRope, "rope_cos", ropeCosBuf);
                cs.SetBuffer(kRope, "rope_sin", ropeSinBuf);
                cs.Dispatch(kRope, (seqLen * numHeads * (HD / 2) + 127) / 128, 1, 1);
            }

            void DispatchLayer(int li, int seqLen)
            {
                int hidTotal = seqLen * H;

                CopyOp(skipBuf, hiddenBuf, hidTotal);
                RmsNormW(lwLn1[li], hiddenBuf, normOutBuf, seqLen);

                cs.SetInt("batch_size", 1);
                cs.SetInt("sequence_length_q", seqLen);
                cs.SetInt("embedding_dim", H);
                cs.SetInt("num_heads_q", HQ);
                cs.SetInt("num_heads_kv", HKV);
                cs.SetInt("head_dim", HD);

                cs.SetBuffer(kQProj, "X", normOutBuf);
                cs.SetBuffer(kQProj, "W_Q", lwQ[li]);
                cs.SetBuffer(kQProj, "proj_bias", lbQ[li]);
                cs.SetBuffer(kQProj, "Q_out", qBuf);
                if (isInt8) cs.SetBuffer(kQProj, "W_Q_scales", lsQ[li]);
                cs.Dispatch(kQProj, 1, (seqLen + 7) / 8, (HQ * HD + 31) / 32);

                cs.SetBuffer(kKProj, "X", normOutBuf);
                cs.SetBuffer(kKProj, "W_K", lwK[li]);
                cs.SetBuffer(kKProj, "proj_bias", lbK[li]);
                cs.SetBuffer(kKProj, "K_out", kBuf);
                if (isInt8) cs.SetBuffer(kKProj, "W_K_scales", lsK[li]);
                cs.Dispatch(kKProj, 1, (seqLen + 7) / 8, (HKV * HD + 31) / 32);

                cs.SetBuffer(kVProj, "X", normOutBuf);
                cs.SetBuffer(kVProj, "W_V", lwV[li]);
                cs.SetBuffer(kVProj, "proj_bias", lbV[li]);
                cs.SetBuffer(kVProj, "V_out", vBuf);
                if (isInt8) cs.SetBuffer(kVProj, "W_V_scales", lsV[li]);
                cs.Dispatch(kVProj, 1, (seqLen + 7) / 8, (HKV * HD + 31) / 32);

                RopeOp(qBuf, seqLen, HQ, CachedTokenCount);
                RopeOp(kBuf, seqLen, HKV, CachedTokenCount);

                cs.SetInt("seq_len", seqLen);
                cs.SetInt("cache_len", CachedTokenCount);
                cs.SetBuffer(kWriteCache, "kv_new", kBuf);
                cs.SetBuffer(kWriteCache, "kv_cache", kCaches[li]);
                cs.Dispatch(kWriteCache, Div256(seqLen * HKV * HD / 2), 1, 1);
                cs.SetBuffer(kWriteCache, "kv_new", vBuf);
                cs.SetBuffer(kWriteCache, "kv_cache", vCaches[li]);
                cs.Dispatch(kWriteCache, Div256(seqLen * HKV * HD / 2), 1, 1);

                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", CachedTokenCount + seqLen);
                cs.SetFloat("scale", 1f / Mathf.Sqrt(HD));
                cs.SetBuffer(kFlashAttn, "Q", qBuf);
                cs.SetBuffer(kFlashAttn, "K", kCaches[li]);
                cs.SetBuffer(kFlashAttn, "V", vCaches[li]);
                cs.SetBuffer(kFlashAttn, "AttendedValues", attendedBuf);
                cs.Dispatch(kFlashAttn, seqLen, HQ, 1);

                cs.SetInt("inner_embedding_dim", HQ * HD);
                cs.SetBuffer(kOProj, "AttendedValues", attendedBuf);
                cs.SetBuffer(kOProj, "W_O", lwO[li]);
                cs.SetBuffer(kOProj, "O", attnOutBuf);
                if (isInt8) cs.SetBuffer(kOProj, "W_O_scales", lsO[li]);
                cs.Dispatch(kOProj, 1, (seqLen + 3) / 4, (H + 31) / 32);
                AddOp(attnOutBuf, skipBuf, hidTotal);

                CopyOp(skipBuf, attnOutBuf, hidTotal);
                RmsNormW(lwLn2[li], attnOutBuf, normOutBuf, seqLen);

                cs.SetInt("seq_len", seqLen);
                cs.SetInt("intermediate_size", MLP_DIM);
                cs.SetInt("activation_type", 0);   // silu
                cs.SetBuffer(kGateUp, "input", normOutBuf);
                cs.SetBuffer(kGateUp, "mlp_gate_w", lwG[li]);
                cs.SetBuffer(kGateUp, "mlp_up_w", lwU[li]);
                cs.SetBuffer(kGateUp, "intermediate", mlpInterBuf);
                if (isInt8)
                {
                    cs.SetBuffer(kGateUp, "mlp_gate_scales", lsG[li]);
                    cs.SetBuffer(kGateUp, "mlp_up_scales", lsU[li]);
                }
                cs.Dispatch(kGateUp, (MLP_DIM + 63) / 64, (seqLen + 7) / 8, 1);

                cs.SetBuffer(kDown, "intermediate", mlpInterBuf);
                cs.SetBuffer(kDown, "mlp_down_w", lwD[li]);
                cs.SetBuffer(kDown, "input", hiddenBuf);   // Down writes `input` -> new hidden
                if (isInt8) cs.SetBuffer(kDown, "mlp_down_scales", lsD[li]);
                cs.Dispatch(kDown, (H + 63) / 64, (seqLen + 7) / 8, 1);
                AddOp(hiddenBuf, skipBuf, hidTotal);
            }

            void DispatchFinalLast(int seqLen)
            {
                cs.SetInt("buffer_size", H);
                cs.SetInt("copy_src_offset", (seqLen - 1) * H);
                cs.SetInt("copy_dst_offset", 0);
                cs.SetBuffer(kCopySlice, "buf_a", lastHiddenBuf);
                cs.SetBuffer(kCopySlice, "buf_b", hiddenBuf);
                cs.Dispatch(kCopySlice, Div256(H), 1, 1);

                RmsNormW(wNormG, lastHiddenBuf, normSingleBuf, 1);
                DebugTap?.Invoke("final_norm", normSingleBuf, H);

                cs.SetInt("vocab_size", VOCAB);
                cs.SetInt("hidden_size", H);
                cs.SetBuffer(kLmHead, "lm_input", normSingleBuf);
                cs.SetBuffer(kLmHead, "lm_weights", wDecoder);
                cs.SetBuffer(kLmHead, "lm_output", logitsBuf);
                cs.Dispatch(kLmHead, (VOCAB + 511) / 512, 1, 1);
            }

            public static int LayersPerYield = LAYERS * 8;

            /// <summary>Prefill from embedsBuf; last position's speech-head logits land in logitsBuf.</summary>
            public IEnumerator PrefillYielding(int totalLen)
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                DebugTap?.Invoke("embeds", embedsBuf, totalLen * H);
                int sinceYield = 0;
                for (int start = 0; start < totalLen; start += CHUNK)
                {
                    int len = Math.Min(CHUNK, totalLen - start);
                    cs.SetInt("buffer_size", len * H);
                    cs.SetInt("copy_src_offset", start * H);
                    cs.SetInt("copy_dst_offset", 0);
                    cs.SetBuffer(kCopySlice, "buf_a", hiddenBuf);
                    cs.SetBuffer(kCopySlice, "buf_b", embedsBuf);
                    cs.Dispatch(kCopySlice, Div256(len * H), 1, 1);

                    for (int li = 0; li < LAYERS; li++)
                    {
                        DispatchLayer(li, len);
                        if (start == 0) DebugTap?.Invoke($"layer{li}", hiddenBuf, len * H);
                        if (++sinceYield >= LayersPerYield) { sinceYield = 0; yield return null; }
                    }
                    CachedTokenCount += len;
                }
                DispatchFinalLast(Math.Min(CHUNK, ((totalLen - 1) % CHUNK) + 1));
                sw.Stop();
                PrefillMs = (float)sw.Elapsed.TotalMilliseconds;
                yield return null;
            }

            // ---------------- A6-max Phase 6: fused decode dispatch (T=1) ------------------------
            // 6 dispatches/layer: QKV (fused input-norm), rope+cache, attention, O+residual,
            // gate*up (fused post-norm), down+residual. Split-k (Phase 6b): deterministic,
            // corr-exact vs DispatchLayer(li, 1) — see the FastLM contract above.
            void DispatchLayerFast(int li)
            {
                // 1. fused input RMSNorm + Q/K/V: 8 rows x 32 lanes; 112 Q + 16 K + 16 V groups
                cs.SetBuffer(kDecQKV, "X", hiddenBuf);
                cs.SetBuffer(kDecQKV, "norm_gamma", lwLn1[li]);
                cs.SetBuffer(kDecQKV, "W_Q", lwQ[li]);
                cs.SetBuffer(kDecQKV, "W_K", lwK[li]);
                cs.SetBuffer(kDecQKV, "W_V", lwV[li]);
                cs.SetBuffer(kDecQKV, "proj_bias", lbQ[li]);
                cs.SetBuffer(kDecQKV, "proj_bias_k", lbK[li]);
                cs.SetBuffer(kDecQKV, "proj_bias_v", lbV[li]);
                cs.SetBuffer(kDecQKV, "Q_out", qBuf);
                cs.SetBuffer(kDecQKV, "K_out", kBuf);
                cs.SetBuffer(kDecQKV, "V_out", vBuf);
                if (isInt8)
                {
                    cs.SetBuffer(kDecQKV, "W_Q_scales", lsQ[li]);
                    cs.SetBuffer(kDecQKV, "W_K_scales", lsK[li]);
                    cs.SetBuffer(kDecQKV, "W_V_scales", lsV[li]);
                }
                cs.Dispatch(kDecQKV, 144, 1, 1);   // 112 Q (896/8) + 16 K + 16 V (128/8)

                // 2. rope(q,k) + append K and V to their caches — one dispatch
                cs.SetBuffer(kDecRope, "Q_out", qBuf);
                cs.SetBuffer(kDecRope, "K_out", kBuf);
                cs.SetBuffer(kDecRope, "V_out", vBuf);
                cs.SetBuffer(kDecRope, "rope_cos", ropeCosBuf);
                cs.SetBuffer(kDecRope, "rope_sin", ropeSinBuf);
                cs.SetBuffer(kDecRope, "kv_cache", kCaches[li]);
                cs.SetBuffer(kDecRope, "kv_cache2", vCaches[li]);
                cs.Dispatch(kDecRope, 1, 1, 1);

                // 3. attention (existing fused kernel — untouched numerics)
                cs.SetBuffer(kFlashAttn, "Q", qBuf);
                cs.SetBuffer(kFlashAttn, "K", kCaches[li]);
                cs.SetBuffer(kFlashAttn, "V", vCaches[li]);
                cs.SetBuffer(kFlashAttn, "AttendedValues", attendedBuf);
                cs.Dispatch(kFlashAttn, 1, HQ, 1);

                // 4. O projection + residual (both skip-copies eliminated: hiddenBuf read directly)
                cs.SetBuffer(kDecOProj, "AttendedValues", attendedBuf);
                cs.SetBuffer(kDecOProj, "W_O", lwO[li]);
                cs.SetBuffer(kDecOProj, "norm_input", hiddenBuf);
                cs.SetBuffer(kDecOProj, "O", attnOutBuf);
                if (isInt8) cs.SetBuffer(kDecOProj, "W_O_scales", lsO[li]);
                cs.Dispatch(kDecOProj, (H + 7) / 8, 1, 1);        // 112 groups, 8 rows x 32 lanes

                // 5. fused post-attention RMSNorm + gate&up + silu*mul
                cs.SetBuffer(kDecGateUp, "input", attnOutBuf);
                cs.SetBuffer(kDecGateUp, "norm_gamma", lwLn2[li]);
                cs.SetBuffer(kDecGateUp, "mlp_gate_w", lwG[li]);
                cs.SetBuffer(kDecGateUp, "mlp_up_w", lwU[li]);
                cs.SetBuffer(kDecGateUp, "intermediate", mlpInterBuf);
                if (isInt8)
                {
                    cs.SetBuffer(kDecGateUp, "mlp_gate_scales", lsG[li]);
                    cs.SetBuffer(kDecGateUp, "mlp_up_scales", lsU[li]);
                }
                cs.Dispatch(kDecGateUp, (MLP_DIM + 7) / 8, 1, 1); // 608 groups, 8 rows x 32 lanes

                // 6. down projection + residual -> next layer's hidden
                cs.SetBuffer(kDecDown, "intermediate", mlpInterBuf);
                cs.SetBuffer(kDecDown, "mlp_down_w", lwD[li]);
                cs.SetBuffer(kDecDown, "norm_input", attnOutBuf);
                cs.SetBuffer(kDecDown, "input", hiddenBuf);
                if (isInt8) cs.SetBuffer(kDecDown, "mlp_down_scales", lsD[li]);
                cs.Dispatch(kDecDown, (H + 3) / 4, 1, 1);         // 224 groups, 4 rows x 64 lanes
            }

            void DispatchHeadFast()   // fused final RMSNorm + speech head
            {
                cs.SetBuffer(kDecHead, "X", hiddenBuf);
                cs.SetBuffer(kDecHead, "norm_gamma", wNormG);
                cs.SetBuffer(kDecHead, "lm_weights", wDecoder);
                cs.SetBuffer(kDecHead, "lm_output", logitsBuf);
                cs.Dispatch(kDecHead, (VOCAB + 7) / 8, 1, 1);     // 846 groups, 8 rows x 32 lanes
            }

            /// <summary>One AR step: re-embed prev speech token via speech_embedding, run all
            /// layers against the KV cache, project speech-head logits.</summary>
            public IEnumerator DecodeStepYielding(int prevToken)
            {
                if (wSpeechEmb == null) ResolveWeights();   // defensive: callers prefill first
                decodeTok[0] = prevToken;
                EmbedSegmentW(wSpeechEmb, decodeTok, 0, hiddenBuf);
                if (FastLM)
                {
                    // per-token uniforms shared by every fused dispatch this step
                    cs.SetInt("hidden_size", H);
                    cs.SetInt("embedding_dim", H);
                    cs.SetInt("intermediate_size", MLP_DIM);
                    cs.SetInt("num_heads_q", HQ);
                    cs.SetInt("num_heads_kv", HKV);
                    cs.SetInt("head_dim", HD);
                    cs.SetFloat("norm_eps", CosyVoiceConfig.LM_RMS_EPS);
                    cs.SetInt("position_offset", CachedTokenCount);
                    cs.SetInt("cache_len", CachedTokenCount);
                    cs.SetInt("seq_len_q", 1);
                    cs.SetInt("seq_len_k", CachedTokenCount + 1);
                    cs.SetFloat("scale", 1f / Mathf.Sqrt(HD));
                    cs.SetInt("vocab_size", VOCAB);
                }
                int sinceYield = 0;
                for (int li = 0; li < LAYERS; li++)
                {
                    if (FastLM) DispatchLayerFast(li); else DispatchLayer(li, 1);
                    if (++sinceYield >= LayersPerYield) { sinceYield = 0; yield return null; }
                }
                CachedTokenCount += 1;
                if (FastLM) DispatchHeadFast(); else DispatchFinalLast(1);
            }

            /// <summary>Blocking readback of the 6761 speech logits (27 KB) for the CPU sampler.
            /// Probes only — interactive paths use ReadLogitsYielding (no per-token fence).</summary>
            public float[] ReadLogits()
            {
                float[] r = new float[VOCAB];
                logitsBuf.GetData(r);
                return r;
            }

            /// <summary>Async readback of the 6761 speech logits into dst — same values as
            /// ReadLogits, but the CPU never stalls on the in-flight forward (the sync GetData
            /// was a full GPU fence per token). Falls back to blocking after ASYNC_SPIN_CAP
            /// yields (edit-mode probes never pump frames) or when async readback is unsupported.</summary>
            public IEnumerator ReadLogitsYielding(float[] dst)
            {
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(logitsBuf);
                    int spins = 0;
                    while (!req.done)
                    {
                        if (++spins > ASYNC_SPIN_CAP) { req.WaitForCompletion(); break; }
                        yield return null;
                    }
                    if (!req.hasError)
                    {
                        req.GetData<float>().CopyTo(dst);
                        yield break;
                    }
                }
                logitsBuf.GetData(dst);
            }

            /// <summary>GPU RAS sample (lever 7A): SampleTokenRAS runs the full RAS rule on the
            /// GPU (nucleus top_p 0.8 / top_k 25, last-10 repeat -> multinomial, ignoreEos) and
            /// only the 4-byte token id is read back asynchronously. result[0] = sampled id.
            /// The kernel appends the pick to its own GPU history ring — the caller only passes
            /// how many speech tokens were decoded so far.</summary>
            public IEnumerator SampleRasYielding(int decodedCount, bool ignoreEos, int stepSeed, int[] result)
            {
                cs.SetInt("vocab_size", VOCAB);
                cs.SetFloat("temperature", 1f);
                cs.SetInt("top_k_val", CosyVoiceConfig.RAS_TOP_K);
                cs.SetFloat("top_p_val", CosyVoiceConfig.RAS_TOP_P);
                cs.SetInt("rng_seed", stepSeed);
                cs.SetInt("ras_count", decodedCount);
                cs.SetInt("ras_ignore_eos", ignoreEos ? 1 : 0);
                cs.SetInt("ras_win_size", CosyVoiceConfig.RAS_WIN_SIZE);
                cs.SetInt("ras_rep_thresh", (int)(CosyVoiceConfig.RAS_WIN_SIZE * CosyVoiceConfig.RAS_TAU_R));
                cs.SetInt("ras_eos_floor", CosyVoiceConfig.SPEECH_VOCAB);
                cs.SetBuffer(kSampleRas, "logits_buf", logitsBuf);
                cs.SetBuffer(kSampleRas, "probs_buf", probsBuf);
                cs.SetBuffer(kSampleRas, "argmax_result", argmaxBuf);
                cs.SetBuffer(kSampleRas, "ras_history", rasHistBuf);
                cs.Dispatch(kSampleRas, 1, 1, 1);

                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(argmaxBuf);
                    int spins = 0;
                    while (!req.done)
                    {
                        if (++spins > ASYNC_SPIN_CAP) { req.WaitForCompletion(); break; }
                        yield return null;
                    }
                    if (!req.hasError)
                    {
                        result[0] = (int)req.GetData<uint>()[0];
                        yield break;
                    }
                }
                uint[] r = new uint[1];
                argmaxBuf.GetData(r);
                result[0] = (int)r[0];
            }

            // ---------------- CPU RAS sampling (utils/common.py::ras_sampling exact port) --------
            // nucleus: sort probs desc (stable), keep while cum < top_p AND count < top_k, sample.
            // If the candidate appeared >= win_size*tau_r (=1) times in the last win_size decoded
            // tokens -> fall back to plain multinomial over the full softmax. ignoreEos (below
            // min_len): resample until the candidate is a non-stop id (< 6561).
            public static int RasSample(float[] logits, IReadOnlyList<int> decoded, bool ignoreEos, System.Random rng)
            {
                int n = logits.Length;
                double max = double.MinValue;
                for (int i = 0; i < n; i++) if (logits[i] > max) max = logits[i];
                double[] probs = new double[n];
                double sum = 0;
                for (int i = 0; i < n; i++) { probs[i] = Math.Exp(logits[i] - max); sum += probs[i]; }
                for (int i = 0; i < n; i++) probs[i] /= sum;

                // top-K selection, descending (NucleusPick only ever reads order[0..RAS_TOP_K)):
                // replaces a full 6761-element delegate sort (~ms/token of pure CPU) with one
                // O(n·K) pass. Same candidates and order (exact prob ties are ~impossible in fp).
                int K = CosyVoiceConfig.RAS_TOP_K;
                int[] order = new int[K];
                double[] topP = new double[K];
                int filled = 0;
                for (int i = 0; i < n; i++)
                {
                    double p = probs[i];
                    if (filled == K && p <= topP[K - 1]) continue;
                    int pos = filled < K ? filled++ : K - 1;
                    while (pos > 0 && topP[pos - 1] < p)
                    {
                        topP[pos] = topP[pos - 1];
                        order[pos] = order[pos - 1];
                        pos--;
                    }
                    topP[pos] = p;
                    order[pos] = i;
                }

                for (int attempt = 0; attempt < 100; attempt++)
                {
                    int cand = NucleusPick(probs, order, rng);
                    int reps = 0;
                    int lo = Math.Max(0, decoded.Count - CosyVoiceConfig.RAS_WIN_SIZE);
                    for (int i = lo; i < decoded.Count; i++) if (decoded[i] == cand) reps++;
                    if (reps >= CosyVoiceConfig.RAS_WIN_SIZE * CosyVoiceConfig.RAS_TAU_R)
                        cand = MultinomialPick(probs, rng);
                    if (!ignoreEos || cand < CosyVoiceConfig.SPEECH_VOCAB)
                        return cand;
                }
                return MultinomialPick(probs, rng);   // pathological fallback

                static int NucleusPick(double[] probs, int[] order, System.Random rng)
                {
                    double cum = 0;
                    int count = 0;
                    Span<double> kept = stackalloc double[CosyVoiceConfig.RAS_TOP_K];
                    while (count < CosyVoiceConfig.RAS_TOP_K && cum < CosyVoiceConfig.RAS_TOP_P)
                    {
                        cum += probs[order[count]];
                        kept[count] = probs[order[count]];
                        count++;
                    }
                    double r = rng.NextDouble() * cum;
                    double acc = 0;
                    for (int i = 0; i < count; i++)
                    {
                        acc += kept[i];
                        if (r <= acc) return order[i];
                    }
                    return order[count - 1];
                }

                static int MultinomialPick(double[] probs, System.Random rng)
                {
                    double r = rng.NextDouble();
                    double acc = 0;
                    for (int i = 0; i < probs.Length; i++)
                    {
                        acc += probs[i];
                        if (r <= acc) return i;
                    }
                    return probs.Length - 1;
                }
            }

            /// <summary>Full AR generation (offline): prefill + RAS decode until a stop id.
            /// textTokens must already be prompt_text ++ utterance text (with &lt;|endofprompt|&gt;);
            /// uttTextLen = the UTTERANCE token count only (min/max lengths scale on it — llm.py:497).
            /// Yields every decode step; onToken fires per emitted speech token (streaming tap).</summary>
            public IEnumerator GenerateYielding(int[] textTokens, int[] promptSpeechTokens,
                                                List<int> outTokens, int uttTextLen,
                                                Action<int> onToken = null, int seed = 0)
            {
                ResetCache();
                int L = BuildPrefillEmbeds(textTokens, promptSpeechTokens);
                var pf = PrefillYielding(L);
                while (pf.MoveNext()) yield return pf.Current;

                int minLen = (int)(uttTextLen * CosyVoiceConfig.MIN_TOKEN_TEXT_RATIO);
                int maxLen = (int)(uttTextLen * CosyVoiceConfig.MAX_TOKEN_TEXT_RATIO);
                var rng = new System.Random(seed);

                float[] logits = GpuSampler ? null : new float[VOCAB];
                int[] picked = GpuSampler ? new int[1] : null;
                for (int step = 0; step < maxLen; step++)
                {
                    int tok;
                    if (GpuSampler)
                    {
                        var s = SampleRasYielding(outTokens.Count, outTokens.Count < minLen, rng.Next(), picked);
                        while (s.MoveNext()) yield return s.Current;
                        tok = picked[0];
                    }
                    else
                    {
                        var rd = ReadLogitsYielding(logits);   // async — same values, no fence
                        while (rd.MoveNext()) yield return rd.Current;
                        tok = RasSample(logits, outTokens, ignoreEos: outTokens.Count < minLen, rng);
                    }
                    if (tok >= CosyVoiceConfig.SPEECH_VOCAB) break;   // any tail id is a stop id
                    outTokens.Add(tok);
                    onToken?.Invoke(tok);
                    var d = DecodeStepYielding(tok);
                    while (d.MoveNext()) yield return d.Current;
                }
            }

            public void Dispose()
            {
                for (int i = 0; i < LAYERS; i++) { kCaches[i]?.Release(); vCaches[i]?.Release(); }
                embedsBuf?.Release(); hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release();
                attnOutBuf?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release();
                attendedBuf?.Release(); mlpInterBuf?.Release(); lastHiddenBuf?.Release();
                normSingleBuf?.Release(); logitsBuf?.Release(); tokenIdsBuf?.Release();
                ropeCosBuf?.Release(); ropeSinBuf?.Release();
                probsBuf?.Release(); argmaxBuf?.Release(); rasHistBuf?.Release();
            }
        }
    }
}
