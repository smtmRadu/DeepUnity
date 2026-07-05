using System;
using System.Collections;
using UnityEngine;

namespace DeepUnity
{
    namespace ChatterboxModeling
    {
        // Chatterbox-Turbo T3: GPT2-medium speech-token decoder, full-GPU (T3CS.compute).
        // Graph per SPEC.md §2: embeds = [spkr_enc(speaker_emb) | speech_emb(prompt 375) |
        // text_emb(text) | speech_emb(BOS 6561)] + wpe -> 24 pre-LN GPT2 blocks (fused biased QKV,
        // causal MHA 16x64, gelu_new MLP) -> ln_f -> biased speech head (6563) -> sample
        // (temp/top-k/top-p + HF repetition penalty over generated tokens).
        public class T3Model : IDisposable
        {
            const int H = ChatterboxConfig.T3_HIDDEN;
            const int LAYERS = ChatterboxConfig.T3_LAYERS;
            const int HEADS = ChatterboxConfig.T3_HEADS;
            const int HEAD_DIM = ChatterboxConfig.T3_HEAD_DIM;
            const int QKV_DIM = ChatterboxConfig.T3_QKV_DIM;
            const int MLP = ChatterboxConfig.T3_MLP;
            const int SPEECH_VOCAB = ChatterboxConfig.T3_SPEECH_VOCAB;
            const int CHUNK = 8;              // prefill positions per dispatch burst (frame budget)

            readonly ComputeShader cs;
            readonly ChatterboxWeights weights;
            readonly int cacheCapacity;

            // kernels
            int kEmbedToken, kAddWpe, kSpkrProj, kLayerNorm, kProjBias, kProjBias1Vec, kSplitQKV;
            int kWriteCache, kFlashAttn, kLmHead, kRepPenalty, kArgMax, kSampleToken;
            int kZero, kCopy, kCopySlice, kAddResidual;

            // KV cache (FP16 packed, one K + one V buffer per layer) — layout matches KVCache.hlsl
            readonly ComputeBuffer[] kCaches = new ComputeBuffer[LAYERS];
            readonly ComputeBuffer[] vCaches = new ComputeBuffer[LAYERS];
            public int CachedTokenCount { get; private set; }

            // The conditioning prefix (spkr + 375 speech-prompt tokens) is TEXT-INDEPENDENT, so its
            // KV entries can be computed once and reused for every utterance: SavePrefix() after
            // prefilling it, RestoreToPrefix() before each new text. Positions stay absolute
            // (cond | text | speech contiguous), so the cache contents are bit-identical to a full
            // re-prefill. Cuts per-utterance prefill from ~390 tokens to just the text (~10-30).
            public int PrefixTokenCount { get; private set; }
            public void SavePrefix() => PrefixTokenCount = CachedTokenCount;
            public void RestoreToPrefix() => CachedTokenCount = PrefixTokenCount;

            // scratch
            ComputeBuffer embedsBuf;                   // [maxPrefill, H] prefill embeddings
            ComputeBuffer hiddenBuf, skipBuf, normOutBuf, attnOutBuf;
            ComputeBuffer qkvBuf, qBuf, kBuf, vBuf, attendedBuf, mlpInterBuf;
            ComputeBuffer lastHiddenBuf, normSingleBuf;
            ComputeBuffer logitsBuf, probsBuf, argmaxBuf;
            ComputeBuffer tokenIdsBuf, genIdsBuf, spkBuf;

            public bool IsReady => weights.IsReady;

            readonly bool int8Weights;   // manifest is the source of truth: int8 exports carry per-row scale entries

            public T3Model(ChatterboxWeights weights, int cacheCapacity = 2048)
            {
                this.weights = weights;
                this.cacheCapacity = cacheCapacity;
                cs = DeepUnityMeta.T3CS;
                cs.EnableKeyword("KV_FP16");   // packed fp16 KV (KVCache.hlsl)
                cs.DisableKeyword("KV_INT8");
                int8Weights = weights.Has("t3/layer_0/qkv.w.scales");
                if (int8Weights) cs.EnableKeyword("INT8_WEIGHTS"); else cs.DisableKeyword("INT8_WEIGHTS");

                kEmbedToken = cs.FindKernel("EmbedToken");
                kAddWpe = cs.FindKernel("AddWpe");
                kSpkrProj = cs.FindKernel("SpkrProj");
                kLayerNorm = cs.FindKernel("LayerNorm");
                kProjBias = cs.FindKernel("ProjBias");
                kProjBias1Vec = cs.FindKernel("ProjBias1Vec");
                kSplitQKV = cs.FindKernel("SplitQKV");
                kWriteCache = cs.FindKernel("WriteCacheFull");
                kFlashAttn = cs.FindKernel("FlashAttention");
                kLmHead = cs.FindKernel("LmHeadBias1Vec");
                kRepPenalty = cs.FindKernel("RepetitionPenalty");
                kArgMax = cs.FindKernel("ArgMax");
                kSampleToken = cs.FindKernel("SampleToken");
                kZero = cs.FindKernel("ZeroBuffer");
                kCopy = cs.FindKernel("CopyBuffer");
                kCopySlice = cs.FindKernel("CopySlice");
                kAddResidual = cs.FindKernel("AddResidual");

                for (int i = 0; i < LAYERS; i++)
                {
                    kCaches[i] = new ComputeBuffer(cacheCapacity * HEADS * HEAD_DIM / 2, 4, ComputeBufferType.Structured);
                    vCaches[i] = new ComputeBuffer(cacheCapacity * HEADS * HEAD_DIM / 2, 4, ComputeBufferType.Structured);
                }

                embedsBuf = new ComputeBuffer(cacheCapacity * H, 4, ComputeBufferType.Structured);
                hiddenBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                skipBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                normOutBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                attnOutBuf = new ComputeBuffer(CHUNK * H, 4, ComputeBufferType.Structured);
                qkvBuf = new ComputeBuffer(CHUNK * QKV_DIM, 4, ComputeBufferType.Structured);
                qBuf = new ComputeBuffer(CHUNK * HEADS * HEAD_DIM, 4, ComputeBufferType.Structured);
                kBuf = new ComputeBuffer(CHUNK * HEADS * HEAD_DIM, 4, ComputeBufferType.Structured);
                vBuf = new ComputeBuffer(CHUNK * HEADS * HEAD_DIM, 4, ComputeBufferType.Structured);
                attendedBuf = new ComputeBuffer(CHUNK * HEADS * HEAD_DIM, 4, ComputeBufferType.Structured);
                mlpInterBuf = new ComputeBuffer(CHUNK * MLP, 4, ComputeBufferType.Structured);
                lastHiddenBuf = new ComputeBuffer(H, 4, ComputeBufferType.Structured);
                normSingleBuf = new ComputeBuffer(H, 4, ComputeBufferType.Structured);
                logitsBuf = new ComputeBuffer(SPEECH_VOCAB, 4, ComputeBufferType.Structured);
                probsBuf = new ComputeBuffer(SPEECH_VOCAB, 4, ComputeBufferType.Structured);
                argmaxBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                tokenIdsBuf = new ComputeBuffer(512, 4, ComputeBufferType.Structured);
                genIdsBuf = new ComputeBuffer(ChatterboxConfig.MAX_SPEECH_TOKENS + 1, 4, ComputeBufferType.Structured);
                spkBuf = new ComputeBuffer(ChatterboxConfig.SPEAKER_EMB_DIM, 4, ComputeBufferType.Structured);
            }

            static int Div256(int n) => (n + 255) / 256;

            public void ResetCache() => CachedTokenCount = 0;

            /// <summary>Uploads the baked speaker embedding (conds/t3_speaker_emb, 256 floats).</summary>
            public void SetSpeakerEmbedding(float[] emb) => spkBuf.SetData(emb);

            void UploadTokens(int[] ids, int count)
            {
                uint[] arr = new uint[count];
                for (int i = 0; i < count; i++) arr[i] = (uint)ids[i];
                tokenIdsBuf.SetData(arr);
            }

            // ---- prefill embedding assembly (SPEC §2: cond | text | speech BOS, then +wpe) --------
            void EmbedSegment(string embName, int[] tokens, int posOffsetElems)
            {
                UploadTokens(tokens, tokens.Length);
                cs.SetInt("seq_len", tokens.Length);
                cs.SetInt("hidden_size", H);
                cs.SetInt("embed_dst_offset", posOffsetElems);
                cs.SetBuffer(kEmbedToken, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbedToken, "emb_weights", weights.Get(embName));
                cs.SetBuffer(kEmbedToken, "embed_output", embedsBuf);
                cs.Dispatch(kEmbedToken, Div256(tokens.Length * H), 1, 1);
            }

            /// <summary>
            /// Builds the full prefill embedding sequence into embedsBuf and returns its length:
            /// [spkr_enc(1) | speech prompt (375) | text (N) | BOS 6561 (1)] + wpe[0..L).
            /// promptTokens = conds/t3_prompt_tokens.
            /// </summary>
            public int BuildPrefillEmbeds(int[] promptTokens, int[] textTokens)
            {
                int condLen = BuildCondEmbeds(promptTokens);
                return condLen + BuildTextEmbeds(textTokens, condLen, dstElemOffset: condLen * H);
            }

            /// <summary>Conditioning segment only: [spkr(1) | speech prompt] + wpe. Prefill this once,
            /// SavePrefix(), then per utterance RestoreToPrefix() + BuildTextEmbeds + prefill text.</summary>
            public int BuildCondEmbeds(int[] promptTokens)
            {
                int L = 1 + promptTokens.Length;
                cs.SetInt("hidden_size", H);
                cs.SetInt("spk_in_dim", ChatterboxConfig.SPEAKER_EMB_DIM);
                cs.SetInt("embed_dst_offset", 0);
                cs.SetBuffer(kSpkrProj, "spk_input", spkBuf);
                cs.SetBuffer(kSpkrProj, "spk_w", weights.Get("t3/spkr_enc.w"));
                cs.SetBuffer(kSpkrProj, "spk_b", weights.Get("t3/spkr_enc.b"));
                cs.SetBuffer(kSpkrProj, "embed_output", embedsBuf);
                cs.Dispatch(kSpkrProj, Div256(H), 1, 1);

                EmbedSegment("t3/speech_emb", promptTokens, 1 * H);

                cs.SetInt("seq_len", L);
                cs.SetInt("position_offset", 0);
                cs.SetInt("embed_dst_offset", 0);   // EmbedSegment left it at the prompt offset
                cs.SetBuffer(kAddWpe, "wpe", weights.Get("t3/wpe"));
                cs.SetBuffer(kAddWpe, "embed_output", embedsBuf);
                cs.Dispatch(kAddWpe, Div256(L * H), 1, 1);
                return L;
            }

            /// <summary>Text + BOS segment at absolute position <paramref name="posOffset"/>, written
            /// into embedsBuf at <paramref name="dstElemOffset"/> (0 for the streaming per-utterance
            /// path — prefill then reads the segment from offset 0). Returns segment length.</summary>
            public int BuildTextEmbeds(int[] textTokens, int posOffset, int dstElemOffset = 0)
            {
                int L = textTokens.Length + 1;
                if (posOffset + L > cacheCapacity)
                    throw new Exception($"T3 prefill too long ({posOffset + L} > {cacheCapacity}); shorten the text.");

                cs.SetInt("hidden_size", H);
                EmbedSegment("t3/text_emb", textTokens, dstElemOffset);
                EmbedSegment("t3/speech_emb", new[] { ChatterboxConfig.START_SPEECH_TOKEN },
                             dstElemOffset + textTokens.Length * H);

                // wpe over just this segment (positions posOffset..posOffset+L)
                cs.SetInt("seq_len", L);
                cs.SetInt("position_offset", posOffset);
                cs.SetInt("embed_dst_offset", dstElemOffset);
                cs.SetBuffer(kAddWpe, "wpe", weights.Get("t3/wpe"));
                cs.SetBuffer(kAddWpe, "embed_output", embedsBuf);
                cs.Dispatch(kAddWpe, Div256(L * H), 1, 1);
                return L;
            }

            // ---- one GPT2 block over the current hiddenBuf chunk ---------------------------------
            void Proj(int kernel, string wName, ComputeBuffer x, ComputeBuffer y, int seqLen,
                      int inDim, int outDim, int act)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("in_dim", inDim);
                cs.SetInt("out_dim", outDim);
                cs.SetInt("activation_type", act);
                cs.SetBuffer(kernel, "X", x);
                cs.SetBuffer(kernel, "W", weights.Get(wName + ".w"));
                if (int8Weights) cs.SetBuffer(kernel, "W_scales", weights.Get(wName + ".w.scales"));
                cs.SetBuffer(kernel, "W_bias", weights.Get(wName + ".b"));
                cs.SetBuffer(kernel, "Y", y);
                if (kernel == kProjBias1Vec) cs.Dispatch(kernel, Div256(outDim), 1, 1);
                else cs.Dispatch(kernel, 1, (seqLen + 7) / 8, (outDim + 31) / 32);
            }

            void LayerNormOp(string lnName, ComputeBuffer input, ComputeBuffer output, int seqLen)
            {
                cs.SetInt("seq_len", seqLen);
                cs.SetInt("hidden_size", H);
                cs.SetFloat("norm_eps", ChatterboxConfig.T3_LN_EPS);
                cs.SetBuffer(kLayerNorm, "norm_input", input);
                cs.SetBuffer(kLayerNorm, "norm_output", output);
                cs.SetBuffer(kLayerNorm, "ln_gamma", weights.Get(lnName + ".w"));
                cs.SetBuffer(kLayerNorm, "ln_beta", weights.Get(lnName + ".b"));
                cs.Dispatch(kLayerNorm, Div256(seqLen), 1, 1);
            }

            void CopyOp(ComputeBuffer dst, ComputeBuffer src, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kCopy, "buf_a", dst);
                cs.SetBuffer(kCopy, "buf_b", src);
                cs.Dispatch(kCopy, Div256(count), 1, 1);
            }

            void AddOp(ComputeBuffer dstA, ComputeBuffer srcB, int count)
            {
                cs.SetInt("buffer_size", count);
                cs.SetBuffer(kAddResidual, "buf_a", dstA);
                cs.SetBuffer(kAddResidual, "buf_b", srcB);
                cs.Dispatch(kAddResidual, Div256(count), 1, 1);
            }

            void DispatchLayer(int li, int seqLen)
            {
                string lp = $"t3/layer_{li}/";
                int hidTotal = seqLen * H;
                int kProj = seqLen == 1 ? kProjBias1Vec : kProjBias;

                // x -> skip; attn = c_proj(FA(split(c_attn(ln_1(x))))); x = skip + attn
                CopyOp(skipBuf, hiddenBuf, hidTotal);
                LayerNormOp(lp + "ln_1", hiddenBuf, normOutBuf, seqLen);
                Proj(kProj, lp + "qkv", normOutBuf, qkvBuf, seqLen, H, QKV_DIM, 0);

                cs.SetInt("seq_len", seqLen);
                cs.SetInt("qkv_proj_dim", QKV_DIM);
                cs.SetInt("num_heads_q", HEADS);
                cs.SetInt("num_heads_kv", HEADS);
                cs.SetInt("head_dim", HEAD_DIM);
                cs.SetBuffer(kSplitQKV, "qkv_packed", qkvBuf);
                cs.SetBuffer(kSplitQKV, "split_q", qBuf);
                cs.SetBuffer(kSplitQKV, "split_k", kBuf);
                cs.SetBuffer(kSplitQKV, "split_v", vBuf);
                cs.Dispatch(kSplitQKV, Div256(seqLen * QKV_DIM), 1, 1);

                // write K/V to the fp16 cache at cache_len = tokens already cached
                cs.SetInt("cache_len", CachedTokenCount);
                cs.SetBuffer(kWriteCache, "kv_new", kBuf);
                cs.SetBuffer(kWriteCache, "kv_cache", kCaches[li]);
                cs.Dispatch(kWriteCache, Div256(seqLen * HEADS * HEAD_DIM / 2), 1, 1);
                cs.SetBuffer(kWriteCache, "kv_new", vBuf);
                cs.SetBuffer(kWriteCache, "kv_cache", vCaches[li]);
                cs.Dispatch(kWriteCache, Div256(seqLen * HEADS * HEAD_DIM / 2), 1, 1);

                int kvLen = CachedTokenCount + seqLen;
                cs.SetInt("seq_len_q", seqLen);
                cs.SetInt("seq_len_k", kvLen);
                cs.SetInt("sliding_window_size", 0);
                cs.SetInt("bidirectional", 0);
                cs.SetFloat("scale", 1f / Mathf.Sqrt(HEAD_DIM));
                cs.SetBuffer(kFlashAttn, "Q", qBuf);
                cs.SetBuffer(kFlashAttn, "K", kCaches[li]);
                cs.SetBuffer(kFlashAttn, "V", vCaches[li]);
                cs.SetBuffer(kFlashAttn, "AttendedValues", attendedBuf);
                cs.Dispatch(kFlashAttn, seqLen, HEADS, 1);

                Proj(kProj, lp + "attn_out", attendedBuf, attnOutBuf, seqLen, H, H, 0);
                AddOp(attnOutBuf, skipBuf, hidTotal);

                // x -> skip; mlp = c_proj(gelu_new(c_fc(ln_2(x)))); x = skip + mlp
                CopyOp(skipBuf, attnOutBuf, hidTotal);
                LayerNormOp(lp + "ln_2", attnOutBuf, normOutBuf, seqLen);
                Proj(kProj, lp + "fc", normOutBuf, mlpInterBuf, seqLen, H, MLP, 1);   // gelu_new fused
                Proj(kProj, lp + "mlp_out", mlpInterBuf, hiddenBuf, seqLen, MLP, H, 0);
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

                LayerNormOp("t3/ln_f", lastHiddenBuf, normSingleBuf, 1);

                cs.SetInt("vocab_size", SPEECH_VOCAB);
                cs.SetInt("hidden_size", H);
                cs.SetBuffer(kLmHead, "lm_input", normSingleBuf);
                cs.SetBuffer(kLmHead, "lm_weights", weights.Get("t3/speech_head.w"));
                cs.SetBuffer(kLmHead, "lm_bias", weights.Get("t3/speech_head.b"));
                cs.SetBuffer(kLmHead, "lm_output", logitsBuf);
                cs.Dispatch(kLmHead, (SPEECH_VOCAB + 511) / 512, 1, 1);
            }

            /// <summary>Pacing knob: layers dispatched between coroutine yields. The default (all
            /// 24 x CHUNK) runs a whole prefill chunk / decode step per frame — a T3 step is tiny
            /// (~Gemma3-270M sized), so this is what real-time speech needs at vsync'd framerates.
            /// Lower it (e.g. 4) if TTS work ever shows up in a frame-spike probe.</summary>
            public static int LayersPerYield = LAYERS * CHUNK;

            /// <summary>Prefill from embedsBuf (built by BuildPrefillEmbeds / BuildTextEmbeds),
            /// chunked; logits for the last position land in logitsBuf.</summary>
            public IEnumerator PrefillYielding(int totalLen)
            {
                int sinceYield = 0;
                for (int start = 0; start < totalLen; start += CHUNK)
                {
                    int len = Math.Min(CHUNK, totalLen - start);
                    // slice this chunk's embeddings into hiddenBuf
                    cs.SetInt("buffer_size", len * H);
                    cs.SetInt("copy_src_offset", start * H);
                    cs.SetInt("copy_dst_offset", 0);
                    cs.SetBuffer(kCopySlice, "buf_a", hiddenBuf);
                    cs.SetBuffer(kCopySlice, "buf_b", embedsBuf);
                    cs.Dispatch(kCopySlice, Div256(len * H), 1, 1);

                    for (int li = 0; li < LAYERS; li++)
                    {
                        DispatchLayer(li, len);
                        if (++sinceYield >= LayersPerYield) { sinceYield = 0; yield return null; }
                    }
                    CachedTokenCount += len;
                }
                DispatchFinalLast(Math.Min(CHUNK, ((totalLen - 1) % CHUNK) + 1));
                yield return null;
            }

            /// <summary>One decode step: embed the previous speech token at the next absolute
            /// position, run all layers with the KV cache, project logits. No internal yields —
            /// the sampler's async readback is the natural frame boundary.</summary>
            public IEnumerator DecodeStepYielding(int prevToken)
            {
                UploadTokens(new[] { prevToken }, 1);
                cs.SetInt("seq_len", 1);
                cs.SetInt("hidden_size", H);
                cs.SetInt("embed_dst_offset", 0);
                cs.SetBuffer(kEmbedToken, "token_ids", tokenIdsBuf);
                cs.SetBuffer(kEmbedToken, "emb_weights", weights.Get("t3/speech_emb"));
                cs.SetBuffer(kEmbedToken, "embed_output", hiddenBuf);
                cs.Dispatch(kEmbedToken, Div256(H), 1, 1);

                cs.SetInt("position_offset", CachedTokenCount);
                cs.SetBuffer(kAddWpe, "wpe", weights.Get("t3/wpe"));
                cs.SetBuffer(kAddWpe, "embed_output", hiddenBuf);
                cs.Dispatch(kAddWpe, Div256(H), 1, 1);

                int sinceYield = 0;
                for (int li = 0; li < LAYERS; li++)
                {
                    DispatchLayer(li, 1);
                    if (++sinceYield >= LayersPerYield) { sinceYield = 0; yield return null; }
                }
                CachedTokenCount += 1;
                DispatchFinalLast(1);
            }

            /// <summary>Synchronous sample: dispatches the sampler and blocks on the 4-byte readback.
            /// For the BUDGET-PUMPED real-time path — the sync forces the queued (tiny) T3 step to
            /// drain now instead of spending a frame per token on an async readback. Interactive
            /// per-frame paths should prefer SampleYielding.</summary>
            public int SampleNow(int[] genIds, int genCount, float temperature, int topK,
                                 float topP, float repPenalty)
            {
                DispatchSampleOps(genIds, genCount, temperature, topK, topP, repPenalty);
                uint[] r = new uint[1];
                argmaxBuf.GetData(r);
                return (int)r[0];
            }

            void DispatchSampleOps(int[] genIds, int genCount, float temperature, int topK,
                                   float topP, float repPenalty)
            {
                if (repPenalty != 1f && genCount > 0)
                {
                    uint[] arr = new uint[genCount];
                    for (int i = 0; i < genCount; i++) arr[i] = (uint)genIds[i];
                    genIdsBuf.SetData(arr);
                    cs.SetInt("gen_count", genCount);
                    cs.SetInt("vocab_size", SPEECH_VOCAB);
                    cs.SetFloat("rep_penalty", repPenalty);
                    cs.SetBuffer(kRepPenalty, "gen_ids", genIdsBuf);
                    cs.SetBuffer(kRepPenalty, "logits_buf", logitsBuf);
                    cs.Dispatch(kRepPenalty, Div256(genCount), 1, 1);
                }

                cs.SetInt("vocab_size", SPEECH_VOCAB);
                if (temperature <= 0f)
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
                    cs.SetInt("rng_seed", UnityEngine.Random.Range(int.MinValue, int.MaxValue));
                    cs.SetBuffer(kSampleToken, "logits_buf", logitsBuf);
                    cs.SetBuffer(kSampleToken, "probs_buf", probsBuf);
                    cs.SetBuffer(kSampleToken, "argmax_result", argmaxBuf);
                    cs.Dispatch(kSampleToken, 1, 1, 1);
                }
            }

            /// <summary>GPU sample from logitsBuf. genIds = UNIQUE previously-generated speech tokens
            /// (HF repetition-penalty domain; first step: just the BOS 6561, per inference_turbo).
            /// Writes the sampled id into result[0] via async readback.</summary>
            public IEnumerator SampleYielding(int[] genIds, int genCount, float temperature, int topK,
                                              float topP, float repPenalty, int[] result)
            {
                DispatchSampleOps(genIds, genCount, temperature, topK, topP, repPenalty);
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
                uint[] r = new uint[1]; argmaxBuf.GetData(r);
                result[0] = (int)r[0];
            }

            /// <summary>Blocking readback of the current speech-head logits (parity probes only).</summary>
            public float[] ReadLogits()
            {
                float[] r = new float[SPEECH_VOCAB];
                logitsBuf.GetData(r);
                return r;
            }

            public void Dispose()
            {
                for (int i = 0; i < LAYERS; i++) { kCaches[i]?.Release(); vCaches[i]?.Release(); }
                embedsBuf?.Release(); hiddenBuf?.Release(); skipBuf?.Release(); normOutBuf?.Release();
                attnOutBuf?.Release(); qkvBuf?.Release(); qBuf?.Release(); kBuf?.Release(); vBuf?.Release();
                attendedBuf?.Release(); mlpInterBuf?.Release(); lastHiddenBuf?.Release(); normSingleBuf?.Release();
                logitsBuf?.Release(); probsBuf?.Release(); argmaxBuf?.Release();
                tokenIdsBuf?.Release(); genIdsBuf?.Release(); spkBuf?.Release();
            }
        }
    }
}
