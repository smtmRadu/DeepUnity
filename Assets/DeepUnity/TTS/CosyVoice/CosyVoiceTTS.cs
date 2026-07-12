using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    namespace CosyVoiceModeling
    {
        // Fun-CosyVoice3-0.5B — streaming-native zero-shot TTS, full-GPU (SPEC.md §0):
        //   text ──Qwen BPE──▶ CosyVoice3LM (AR, 25Hz FSQ tokens, RAS sampling)
        //        ──▶ CausalMaskedDiffWithDiT flow (tokens → mel 80@50Hz, 10-step CFM+CFG)
        //        ──▶ CausalHiFT vocoder (F0+NSF+iSTFT) ──▶ 24 kHz mono PCM
        // Voices are baked folders under the weights dir (voices/<name>/...: prompt transcript
        // tokens incl. <|endofprompt|>, prompt FSQ tokens, prompt mel, campplus x-vector).
        //
        // A4 = this offline path (Synthesize/Speak). A5 adds token-level streaming: the LM's
        // onToken tap already fires per token; the chunked flow/vocoder land there.
        public class CosyVoiceTTS : TTS
        {
            public override int SampleRate => CosyVoiceConfig.SAMPLE_RATE;
            readonly string paramsPath;
            public override string ResidencyLabel => ResidencyLog.Label(paramsPath);

            readonly CosyVoiceWeights weights;
            readonly CosyVoiceTokenizer tokenizer;
            readonly CosyVoiceLM lm;
            readonly CosyVoiceFlow flow;
            readonly HiFTVocoder voc;
            readonly int[] voicePromptText;     // baked transcript tokens (contain <|endofprompt|>)
            readonly int[] voicePromptSpeech;   // baked FSQ prompt tokens
            ComputeBuffer melSliceBuf;

            /// <summary>RAS sampling seed (deterministic synthesis for probes; games leave -1 = random).</summary>
            public int Seed = -1;

            /// <summary>A6-max streaming fast path (default ON): single-pass causal flow (frozen
            /// 50-frame blocks solved once against a per-(step,layer) K/V cache) + windowed
            /// re-vocode. false = legacy full re-solve/re-vocode per chunk — the pre-A6 baseline
            /// for A/B (offline synthesis is identical either way).</summary>
            public bool FastStreaming
            {
                get => flow.SinglePassStreaming;
                set { flow.SinglePassStreaming = value; voc.WindowedStreaming = value; }
            }

            /// <summary>GPU-side RAS sampling (lever 7A, default OFF): token ids sampled on the
            /// GPU, only 4 bytes read back per token. Different RNG stream than the CPU sampler,
            /// so seeded runs change — the default (CPU RAS + async logits readback) is
            /// bit-identical to the pre-A6 token stream.</summary>
            public bool UseGpuSampler
            {
                get => lm.GpuSampler;
                set => lm.GpuSampler = value;
            }

            /// <summary>9b (default ON): first streaming emission at the first 25-token grid line
            /// (~16 tokens in) instead of after promptPad+25 (~41) — the dominant TTFA lever
            /// besides the flow itself. Token stream and frozen-block mel are unchanged; only
            /// chunk/seam positions move. false = the reference 25->50->100 schedule.</summary>
            public bool LowLatencyFirstChunk = true;

            // ---- A6-max Phase 4: per-utterance streaming attribution (non-overlapping walls).
            // Chunk-synthesis work is timed inside the overlap pump, so StreamLmMs = token-loop
            // wall MINUS chunk work = LM issue + sampler waits that no chunk work could fill.
            public float StreamPrefillMs { get; private set; }
            public float StreamLmMs { get; private set; }
            public float StreamChunkMs { get; private set; }
            public float StreamFinalizeMs { get; private set; }
            // Phase 5: chunk-pump time split by WHEN it ran — during the token loop (overlapped
            // with decode) vs after it (pure serial tail). after ≈ 0 means the CPU-side overlap
            // is exhausted; any residual wall gap is then single-queue GPU serialization (D3D11
            // has no async compute — LM and flow GPU work EXECUTE sequentially regardless of
            // how the CPU schedules them, so the wall floor is TOTAL GPU time + unhidden CPU).
            public float StreamChunkDuringMs { get; private set; }
            public float StreamChunkAfterMs { get; private set; }
            /// <summary>Per-token pump budget (ms) for the in-flight chunk. Each token also caps
            /// at 8 pump steps so readback spin-waits cannot burn their patience in one token.</summary>
            public float ChunkPumpBudgetMs = 3f;
            /// <summary>Seams the vocoder cross-faded this utterance (probe evidence that the
            /// fade engaged on every boundary incl. first and finalize).</summary>
            public int SeamsBlended => voc.SeamsBlended;
            public float LmMs { get; private set; }
            public float FlowMs => flow.FlowMs;
            public float VocoderMs => voc.VocoderMs;
            public int LastTokenCount { get; private set; }

            public override bool IsReady => weights.IsReady;
            public override ModelResidency Residency => weights.Residency;
            public override long TotalWeightBytes => weights.BytesTotal;
            public override long UploadedWeightBytes => weights.BytesUploaded;
            public override long LoadBudgetBytesPerFrame
            {
                get => weights.BudgetBytesPerFrame;
                set => weights.BudgetBytesPerFrame = value;
            }

            public CosyVoiceTTS(
                string paramsPath = "Assets/Resources/Weights/weights_cosyvoice3_fp16",
                string voice = "default",
                bool beginLoad = true)
            {
                this.paramsPath = paramsPath;
                weights = new CosyVoiceWeights(paramsPath, beginLoad);

                // Unknown voice (e.g. a Kokoro voicepack name left on the NPC after switching
                // engines) must NOT brick the engine into silence — fall back to a baked voice
                // with a loud warning instead of throwing out of the ctor.
                if (!weights.Has($"voices/{voice}/prompt_text_tokens"))
                {
                    string fallback = weights.Has("voices/velmire/prompt_text_tokens") ? "velmire" : "default";
                    ConsoleMessage.Warning($"CosyVoice3: baked voice '{voice}' not found in {paramsPath} " +
                                           $"(expected voices/{voice}/...) — falling back to '{fallback}'. " +
                                           "Bake new voices with validation/make_voice.py.");
                    voice = fallback;
                }

                tokenizer = new CosyVoiceTokenizer();
                lm = new CosyVoiceLM(weights);
                flow = new CosyVoiceFlow(weights, voice);
                voc = new HiFTVocoder(weights);
                voicePromptText = weights.ReadInts($"voices/{voice}/prompt_text_tokens");
                voicePromptSpeech = weights.ReadInts($"voices/{voice}/prompt_speech_tokens");
            }

            /// <summary>Synchronous full load — EDITOR/VALIDATION ONLY (probes cannot pump frames).
            /// Games use Prefetch()/SlowPrefetch().</summary>
            public void LoadBlocking() => weights.LoadBlocking();

            protected override void StartPrefetch(long bytesPerFrame)
            {
                weights.BudgetBytesPerFrame = bytesPerFrame;
                weights.BeginLoad();
            }

            public override void Defetch(DefetchMode mode)
                => weights.Defetch(mode == DefetchMode.Slow ? weights.BudgetBytesPerFrame : 0);

            /// <summary>One tiny synthesis to compile/warm every kernel path (call once after
            /// IsReady, e.g. behind a loading screen).</summary>
            public override IEnumerator Warmup()
            {
                while (!weights.IsReady) yield return null;
                float[] _ = null;
                var e = Synthesize("Hi.", w => _ = w);
                while (e.MoveNext()) yield return e.Current;
            }

            /// <summary>Per-token tap (fires as the LM emits speech tokens) — A5 streaming hook.</summary>
            public Action<int> OnSpeechToken;

            public override IEnumerator Synthesize(string text, Action<float[]> onWav)
            {
                if (!weights.IsReady)
                {
                    ConsoleMessage.Warning("CosyVoiceTTS.Synthesize called before weights are resident (Prefetch first).");
                    onWav?.Invoke(null);
                    yield break;
                }

                // LM text = baked prompt transcript ++ utterance tokens (SPEC §1/§4)
                int[] utt = tokenizer.EncodeIds(text);
                int[] textTokens = new int[voicePromptText.Length + utt.Length];
                voicePromptText.CopyTo(textTokens, 0);
                utt.CopyTo(textTokens, voicePromptText.Length);

                var sw = System.Diagnostics.Stopwatch.StartNew();
                var tokens = new List<int>();
                int seed = Seed >= 0 ? Seed : UnityEngine.Random.Range(0, int.MaxValue);
                var gen = lm.GenerateYielding(textTokens, voicePromptSpeech, tokens, utt.Length, OnSpeechToken, seed);
                while (gen.MoveNext()) yield return gen.Current;
                sw.Stop();
                LmMs = (float)sw.Elapsed.TotalMilliseconds;
                LastTokenCount = tokens.Count;
                TokensPerSecond = tokens.Count / Mathf.Max((float)sw.Elapsed.TotalSeconds, 1e-3f);
                if (tokens.Count == 0) { onWav?.Invoke(null); yield break; }

                // tokens -> mel (rows [pm, pm+frames) of the flow's x buffer)
                ComputeBuffer mel = null; int pm = 0, frames = 0;
                var fl = flow.SynthesizeMelYielding(tokens.ToArray(), (m, p, n) => { mel = m; pm = p; frames = n; });
                while (fl.MoveNext()) yield return fl.Current;

                // slice the generated rows to a vocoder-owned buffer (CPU hop is fine offline;
                // A5 streams per chunk and keeps this on-GPU)
                float[] melHost = new float[frames * CosyVoiceConfig.MEL_DIM];
                mel.GetData(melHost, 0, pm * CosyVoiceConfig.MEL_DIM, melHost.Length);
                if (melSliceBuf == null || melSliceBuf.count < melHost.Length)
                {
                    melSliceBuf?.Release();
                    melSliceBuf = new ComputeBuffer(melHost.Length, 4, ComputeBufferType.Structured);
                }
                melSliceBuf.SetData(melHost);

                float[] wav = null;
                var vo = voc.VocodeYielding(melSliceBuf, frames, w => wav = w);
                while (vo.MoveNext()) yield return vo.Current;
                onWav?.Invoke(wav);
            }

            // ---------------- A5: token-level streaming synthesis --------------------------------
            // ONE LM pass; every token_hop new tokens (25 -> 50 -> 100, +3 lookahead held back)
            // a flow chunk under the 50-frame chunk mask, then the vocoder emits only the NEW
            // samples (CosyVoice3Model.tts/token2wav schedule, cli/model.py:346-374). The first
            // hop is padded so prompt+hop aligns to the 50-frame chunk grid.
            // A6-max (FastStreaming, default ON): the flow solves ONLY the new 50-frame blocks
            // against its per-(step,layer) K/V cache and the vocoder re-runs just an overlap
            // window — both were O(prefix) per chunk before (2.62x redundancy, research §2.5).
            public IEnumerator SynthesizeStreaming(string text, Action<float[]> onSamples)
            {
                if (!weights.IsReady) { onSamples?.Invoke(null); yield break; }

                int[] utt = tokenizer.EncodeIds(text);
                int[] textTokens = new int[voicePromptText.Length + utt.Length];
                voicePromptText.CopyTo(textTokens, 0);
                utt.CopyTo(textTokens, voicePromptText.Length);

                lm.ResetCache();
                flow.ResetStream();   // forget any single-pass state of the previous utterance
                var swPrefill = System.Diagnostics.Stopwatch.StartNew();
                int L = lm.BuildPrefillEmbeds(textTokens, voicePromptSpeech);
                var pf = lm.PrefillYielding(L);
                while (pf.MoveNext()) yield return pf.Current;
                swPrefill.Stop();
                StreamPrefillMs = (float)swPrefill.Elapsed.TotalMilliseconds;
                StreamLmMs = StreamChunkMs = StreamFinalizeMs = 0f;

                int minLen = (int)(utt.Length * CosyVoiceConfig.MIN_TOKEN_TEXT_RATIO);
                int maxLen = (int)(utt.Length * CosyVoiceConfig.MAX_TOKEN_TEXT_RATIO);
                var rng = new System.Random(Seed >= 0 ? Seed : UnityEngine.Random.Range(0, int.MaxValue));

                var tokens = new List<int>();
                int hop = CosyVoiceConfig.CHUNK_TOKENS;
                int promptPad = (voicePromptSpeech.Length + hop - 1) / hop * hop - voicePromptSpeech.Length;
                // 9b: the legacy first hop (hop + promptPad = up to 49 tokens) made the first
                // flow window 250 frames and TTFA ~4.6 s. Low-latency mode emits at the FIRST
                // 25-token grid line instead (>= MIN_FIRST_EMIT tokens so the chunk isn't
                // degenerate); the 50-frame freeze grid is preserved, so frozen-block mel is
                // unchanged — only chunk boundaries (and seam positions) move.
                const int MIN_FIRST_EMIT = 10;
                bool lowLatFirst = LowLatencyFirstChunk;
                int firstHop = lowLatFirst
                    ? (promptPad >= MIN_FIRST_EMIT ? promptPad : promptPad + CosyVoiceConfig.CHUNK_TOKENS)
                    : hop + promptPad;
                int offset = 0, speechOffset = 0;
                bool first = true;
                float[] logits = lm.GpuSampler ? null : new float[CosyVoiceConfig.SPEECH_EMB_ROWS];
                int[] picked = lm.GpuSampler ? new int[1] : null;

                // ---- A6-max Phase 4/5: overlapped chunk synthesis. The LM decode and the flow/
                // vocoder are INDEPENDENT GPU workloads (disjoint buffers), so a chunk is kept
                // in flight and pumped from the token loop. Outputs are bit-identical (same
                // dispatches/data, only issue order interleaves); one chunk in flight at a time.
                // Phase 5: no forced drain before the next chunk — a chunk simply STARTS when
                // the previous one has finished (upTo/offset fix the chunk content, so deferring
                // the start cannot change any output sample). Each token pumps under a time
                // budget with an 8-step cap, so readback spin-waits stay cheap and never burn
                // their patience inside one token. The FIRST chunk stays synchronous (TTFA).
                IEnumerator inflight = null;
                bool loopDone = false;
                var swChunkDuring = new System.Diagnostics.Stopwatch();
                var swChunkAfter = new System.Diagnostics.Stopwatch();
                bool Pump()
                {
                    if (inflight == null) return false;
                    var w = loopDone ? swChunkAfter : swChunkDuring;
                    w.Start();
                    if (!inflight.MoveNext()) inflight = null;
                    w.Stop();
                    return true;
                }

                var swBudget = new System.Diagnostics.Stopwatch();
                var sw = System.Diagnostics.Stopwatch.StartNew();
                for (int step = 0; step < maxLen; step++)
                {
                    int tok;
                    if (lm.GpuSampler)
                    {
                        var sm = lm.SampleRasYielding(tokens.Count, tokens.Count < minLen, rng.Next(), picked);
                        while (sm.MoveNext()) { Pump(); yield return sm.Current; }
                        tok = picked[0];
                    }
                    else
                    {
                        var rd = lm.ReadLogitsYielding(logits);   // async — no per-token GPU fence
                        while (rd.MoveNext()) { Pump(); yield return rd.Current; }
                        tok = CosyVoiceLM.RasSample(logits, tokens, tokens.Count < minLen, rng);
                    }
                    if (tok >= CosyVoiceConfig.SPEECH_VOCAB) break;
                    tokens.Add(tok);
                    OnSpeechToken?.Invoke(tok);
                    var d = lm.DecodeStepYielding(tok);
                    while (d.MoveNext()) { Pump(); yield return d.Current; }

                    // budgeted pump: >= 1 heavy step (an Euler-step issue overruns the budget,
                    // that's fine) and <= 8 steps (spin-waits are microseconds — without the
                    // step cap one token could exhaust a readback's whole spin patience)
                    swBudget.Restart();
                    for (int p = 0; p < 8 && inflight != null; p++)
                    {
                        Pump();
                        if (swBudget.Elapsed.TotalMilliseconds >= ChunkPumpBudgetMs) break;
                    }

                    int thisHop = first ? firstHop : hop;
                    if (inflight == null && tokens.Count - offset >= thisHop + CosyVoiceConfig.PRE_LOOKAHEAD_LEN)
                    {
                        inflight = SynthChunk(tokens, offset + thisHop + CosyVoiceConfig.PRE_LOOKAHEAD_LEN,
                                              false, speechOffset, n => { speechOffset += n.Length; onSamples(n); });
                        if (first) while (Pump()) { }   // first chunk synchronous -> TTFA
                        offset += thisHop;
                        // legacy doubles after every chunk (38 -> 50 -> 100); low-latency keeps
                        // the second chunk at 25 (13 -> 25 -> 50 -> 100) so latency stays low
                        if (!(first && lowLatFirst))
                            hop = Math.Min(4 * CosyVoiceConfig.CHUNK_TOKENS, hop * 2);
                        first = false;
                    }
                }
                sw.Stop();
                loopDone = true;
                while (Pump()) { }   // serial tail: whatever decode could not hide
                LastTokenCount = tokens.Count;
                TokensPerSecond = tokens.Count / Mathf.Max((float)sw.Elapsed.TotalSeconds, 1e-3f);
                StreamChunkDuringMs = (float)swChunkDuring.Elapsed.TotalMilliseconds;
                StreamChunkAfterMs = (float)swChunkAfter.Elapsed.TotalMilliseconds;
                StreamChunkMs = StreamChunkDuringMs + StreamChunkAfterMs;
                StreamLmMs = (float)(sw.Elapsed.TotalMilliseconds - swChunkDuring.Elapsed.TotalMilliseconds);

                if (tokens.Count > 0)   // finalize: everything, lookahead released, tail untrimmed
                {
                    var swFin = System.Diagnostics.Stopwatch.StartNew();
                    var ch = SynthChunk(tokens, tokens.Count, true, speechOffset,
                                        n => { speechOffset += n.Length; onSamples(n); });
                    while (ch.MoveNext()) yield return ch.Current;
                    swFin.Stop();
                    StreamFinalizeMs = (float)swFin.Elapsed.TotalMilliseconds;
                }
            }

            IEnumerator SynthChunk(List<int> tokens, int upTo, bool finalize, int speechOffset, Action<float[]> onNew)
            {
                ComputeBuffer mel = null; int pm = 0, frames = 0;
                var fl = flow.SynthesizeMelStreamingYielding(tokens.GetRange(0, upTo).ToArray(), finalize,
                                                             (m, p, n) => { mel = m; pm = p; frames = n; });
                while (fl.MoveNext()) yield return fl.Current;

                // async mel readback: the old blocking GetData was a hard fence on the whole
                // flow solve — with the overlap pump these yields are filled by LM decode.
                // Spin cap 2000 (Phase 5): the budgeted pump feeds up to ~10 spins per token,
                // so 2000 ≈ 200 tokens of patience mid-loop; drain loops burn the spins in ms
                // and then hard-wait, which is exactly what a drain wants.
                float[] melHost = new float[frames * CosyVoiceConfig.MEL_DIM];
                bool melRead = false;
                if (SystemInfo.supportsAsyncGPUReadback)
                {
                    var req = UnityEngine.Rendering.AsyncGPUReadback.Request(
                        mel, melHost.Length * 4, pm * CosyVoiceConfig.MEL_DIM * 4);
                    int spins = 0;
                    while (!req.done)
                    {
                        if (++spins > 2000) { req.WaitForCompletion(); break; }
                        yield return null;
                    }
                    if (!req.hasError)
                    {
                        req.GetData<float>().CopyTo(melHost);
                        melRead = true;
                    }
                }
                if (!melRead)
                    mel.GetData(melHost, 0, pm * CosyVoiceConfig.MEL_DIM, melHost.Length);
                if (melSliceBuf == null || melSliceBuf.count < melHost.Length)
                {
                    melSliceBuf?.Release();
                    melSliceBuf = new ComputeBuffer(Math.Max(melHost.Length, 2048 * CosyVoiceConfig.MEL_DIM), 4, ComputeBufferType.Structured);
                }
                melSliceBuf.SetData(melHost, 0, 0, melHost.Length);

                float[] neu = null;
                var vo = voc.VocodeChunkYielding(melSliceBuf, frames, finalize, speechOffset, w => neu = w);
                while (vo.MoveNext()) yield return vo.Current;
                if (neu != null && neu.Length > 0) onNew(neu);
            }

            public override void Release()
            {
                lm?.Dispose();
                flow?.Dispose();
                voc?.Dispose();
                melSliceBuf?.Release();
                weights?.Dispose();
            }
        }
    }
}
