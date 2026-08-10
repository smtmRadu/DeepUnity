using System;
using System.Collections;
using System.Collections.Generic;
using UnityEngine;

namespace DeepUnity
{
    namespace PocketTTSModeling
    {
        using Cfg = PocketTTSConfig;

        // Top-level Kyutai pocket-tts: FlowLM (text->latents, autoregressive) + Mimi (latents->wav).
        // P4 offline generate. AR loop = full-forward-each-step (reuses the bit-exact RunTransformer;
        // causal-equivalent to a KV cache — KV-cache incremental decode is the P5 RTF item).
        //
        // AR loop (pocket_tts generate source):
        //   prefill context = [bos_before_voice ; voice(125) ; text_emb(S)]
        //   step n: append input token (n==0: input_linear(bos_emb); else input_linear(latent_{n-1}))
        //           -> transformer -> out_norm -> c ; out_eos(c) > -4.0 => EOS (stop at eos+frames_after_eos)
        //           -> latent = noise + flow_net(c, s=0, t=1, x=noise)   (1 Euler step, noise ~ N(0, sqrt(temp)))
        //   collect latents [T,32] -> denorm (*emb_std+emb_mean) -> Mimi.Decode -> 24kHz wav
        public class PocketTTS : IDisposable
        {
            readonly PocketTTSWeights weights;
            readonly PocketTTSFlowLM flm;
            readonly PocketTTSMimi mimi;
            readonly string weightsDir;
            PocketTTSTokenizer tokenizer;
            PocketTTSMimiEncoder encoder;    // P8: reference wav -> latents (lazy; only for CloneVoice)
            float[] _speakerProj;            // [1024,32] speaker_proj_weight (CPU matmul for audio_prompt)
            float[] embMean, embStd, voicePrompt, bbv;
            System.Random rng = new System.Random(1234);
            ComputeBuffer syncBuf;   // 1-elem buffer; a blocking GetData flushes the GPU queue for timing
            readonly float[] syncTmp = new float[1];
            ComputeBuffer wavAccum;  // #31-R3: overlapped-mimi wav assembly [maxFrames*1920] (one readback)

            void GrowWav(int n)
            {
                if (wavAccum != null && wavAccum.count >= n) return;
                wavAccum?.Release();
                wavAccum = new ComputeBuffer(Math.Max(n, 1), 4, ComputeBufferType.Structured);
            }

            // #31-R3: issue ONE mimi window of the DecodeWindowed schedule — window [s, e) with
            // ctx latents of left context, tail-restricted to the e-t0 kept frames, harvested
            // GPU-side into wavAccum at t0*1920. No readback (the caller owns the final one).
            void IssueMimiWindow(List<float[]> latents, int t0, int e)
            {
                int s = Math.Max(0, t0 - Cfg.MIMI_DECODE_CTX);
                int nWin = e - s;
                var win = new float[nWin * Cfg.LDIM];
                for (int t = s; t < e; t++)
                    Array.Copy(latents[t], 0, win, (t - s) * Cfg.LDIM, Cfg.LDIM);
                mimi.DecodeIssueTo(wavAccum, t0 * Cfg.SAMPLES_PER_LATENT, win, nWin, embMean, embStd, e - t0);
            }

            // Force the GPU to finish queued dispatches so CPU stopwatch reads reflect real GPU time
            // (FlowHead/DecodeStepKV already GetData, but PrefillKV has no readback of its own).
            void GpuSync()
            {
                if (syncBuf == null) syncBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                syncBuf.GetData(syncTmp);
            }

            public float[] LastLatentsRaw;   // [T,32] raw flow latents (pre-denorm) — for the P4 gate
            public int LastFrames;
            public float GenMs, DecodeMs;
            public float PrefillMs, LoopMs, TtfaMs;   // P5 breakdown: prompt prefill, AR loop, time-to-first-audio

            /// <summary>Streaming chunk cadence (frames between Mimi decodes). Each flush re-decodes
            /// the accumulated block [0..t] (bit-exact causal prefix) and emits only the new tail —
            /// so total decode work is O(T·frames/chunk). Larger chunk = less re-decode overhead but
            /// higher TTFA; smaller = snappier first audio. 8 frames = 0.64 s cadence @ 12.5 Hz;
            /// fine for typical NPC lines (<=~5 s). For long (10 s / 125-frame) clips prefer a
            /// larger chunk, or a true incremental Mimi-state decode (follow-up optimization).</summary>
            public int StreamChunkFrames = 8;

            /// <summary>#31-R3 lever 2 (streaming twin of ArBatchRamp): the FIRST flush fires after
            /// this many frames (then every StreamChunkFrames) — first audio reaches the ring
            /// ~0.5 s earlier at 12.5 Hz. Only the flush BOUNDARY moves: the windowed decode's
            /// kept-tail samples are position-exact, so emitted audio is unchanged. Applied on the
            /// GPU-frame path (FastKernels3) only, keeping the legacy path A/B-identical.</summary>
            public int StreamFirstChunkFrames = 2;

            /// <summary>Default frame cap of a single synthesized clause (~41 s at 12.5 Hz — EOS
            /// ends real clauses far earlier; this is the runaway bound). Shared by both synthesis
            /// entry points and by PrewarmAllocationsYielding, which must pre-size the KV for the
            /// same worst case the real clauses will ask for.</summary>
            public const int DefaultMaxFrames = 512;

            /// <summary>Text rows PrewarmAllocationsYielding budgets per clause. Calibrated with
            /// the real Unigram tokenizer (verifier 2026-07-30): a verbose two-sentence clause
            /// (clausesPerChunk = 2, the shipped setting) measures ~109 tokens, so 192 covers it
            /// with headroom. The emergency comma cut (a 1000-char run with no sentence ender)
            /// tokenizes to ~279 and is DELIBERATELY out of bound — it is pathological for NPC
            /// dialogue, and covering it would double the scratch; if it ever fires, that clause
            /// regrows once (the old cost, once).</summary>
            const int PREALLOC_TEXT_ROWS = 192;

            /// <summary>Voice-prompt rows the preallocation budgets: the CLONE CAP, not the
            /// currently bound voice — at prewarm time the bound prompt is the 125-row baked
            /// default, while cloned NPC voices bind ceil(MAX_REF_SECONDS x FRAME_RATE) = 135
            /// rows right after (verifier finding H: sizing on the default left 118 effective
            /// text rows and every voice swap under-covered).</summary>
            static readonly int PREALLOC_VOICE_ROWS = Mathf.CeilToInt(MAX_REF_SECONDS * Cfg.FRAME_RATE);

            /// <summary>#36.6: the weights residency cycle (PocketTTSWeights.LoadEpoch) the
            /// w_touch pass last COMPLETED under; -1 = never. See the latch comment inside
            /// PrewarmAllocationsYielding — the scratch preallocations below it stay unlatched
            /// because covered buffers are same-frame no-ops anyway (their idempotence IS the
            /// guarantee), while the weights re-walk was the one repeat with a real bill.</summary>
            int wTouchedEpoch = -1;

            /// <summary>Pre-allocate the flow-LM's clause-lifetime buffers at the in-game worst
            /// case (clone-cap voice prompt + PREALLOC_TEXT_ROWS text rows + DefaultMaxFrames),
            /// one real driver allocation per MoveNext — call from a prewarm coroutine while the
            /// player walks up, so the warmup synth and the first real clause allocate NOTHING
            /// mid-conversation (2026-07-30 spike hunt: 174 ms + 286 ms single-frame stalls, both
            /// allocation). Covered buffers yield nothing, so re-running is a same-frame no-op.</summary>
            public IEnumerator PrewarmAllocationsYielding()
            {
                if (!IsReady) yield break;
                if (embMean == null)
                {
                    LoadCpuTensors();   // ~258 KB disk + 128k half-float decodes: give it its own
                    yield return null;  // frame instead of sharing one with the first KV create
                }
                // #36.2 — THE WEIGHTS FIRST. Four rounds of scratch-buffer experiments (##36.1)
                // never moved the walk-up pair because the pair was never the scratch: SetData
                // only STAGES a tensor, and the driver makes it resident at its first dispatch
                // REFERENCE — the warmup synth's first prefill tick binds dozens of flow-LM
                // tensors at once (one bulk MakeResident, the ~126-160 ms frame) and its first
                // Mimi flush binds every SEANet tensor (the ~215-290 ms one). Touch each tensor
                // with a 1-element read instead — CopyBuffer into the 1-elem sync scratch, a few
                // MB of resource per frame — and the migration is paid in walk-up-sized slices
                // BEFORE the synth ever runs. Read-only by construction: weights are never
                // written by any warm pass.
                //
                // #36.6 — and ONCE PER RESIDENCY CYCLE, not once per caller. This routine runs
                // from BOTH the session warm cycle (NPCChatBase.Awake → PrewarmRoutine, ##36.5)
                // and every zone entry's PrepareVoiceRoutine, and the touch pass used to re-walk
                // the full registry each time: with the weights never defetched in between, that
                // re-walk is at best a few dozen wasted walk-up frames and at worst a REAL
                // re-migration bill — the 18:33 frame_spikes.csv showed three 40-55 ms `w_touch`
                // frames at t=11-16 (zone entry, weights resident since t≈5): on a saturated
                // 4 GB card the concurrent ~1 GB Qwen stream evicts, and ##36.1 round 4 already
                // proved a touch cannot PIN what the OS evicts afterward — first real use
                // re-pays regardless, so the re-walk buys nothing the prepare's own layer-paced
                // mini-synth dispatches don't. Latch on the weights' residency cycle instead:
                // skip while (still ready, same LoadEpoch) — no tensor has left the GPU since
                // the pass last completed — and re-run in full after any defetch→re-stream,
                // where SetData has only STAGED the fresh buffers and the pass is doing its
                // designed #36.2 job again. The latch burns ONLY on a pass that completed under
                // an unchanged epoch: a mid-pass defetch aborts through the IsReady check below
                // and the next residency cycle re-touches everything. GPU-agnostic on purpose —
                // cards with VRAM headroom never needed the re-walk (nothing evicts, the touch
                // was already invisible there), and cards without it cannot be helped by one
                // (this hunt's own evidence) — so skipping is the rare fix with no tier dial.
                if (wTouchedEpoch != weights.LoadEpoch)
                {
                    int cycle = weights.LoadEpoch;
                    var shd = DeepUnityMeta.PocketTTSCS;
                    int kCopy1 = shd.FindKernel("CopyBuffer");
                    if (syncBuf == null) syncBuf = new ComputeBuffer(1, 4, ComputeBufferType.Structured);
                    long touchedBytes = 0, touchBudget = (long)InferencePerf.TtsFirstTouchElemsPerFrame * 4;
                    foreach (var wb in weights.ResidentBuffers())
                    {
                        shd.SetInt("buffer_size", 1);
                        shd.SetBuffer(kCopy1, "buf_a", syncBuf);
                        shd.SetBuffer(kCopy1, "buf_b", wb);
                        shd.Dispatch(kCopy1, 1, 1, 1);
                        LastHeavyTick = "w_touch";   // probe attribution — which phase owns the frame
                        touchedBytes += (long)wb.count * 4;
                        if (touchedBytes >= touchBudget)
                        {
                            touchedBytes = 0;
                            yield return null;
                            if (!IsReady) yield break;
                        }
                    }
                    if (IsReady && weights.LoadEpoch == cycle) wTouchedEpoch = cycle;
                    yield return null;
                }
                // #36: the mimi scratch side. WDDM residency migrates PER RESOURCE, so the ~24 MB
                // SEANet scratches are committed here, one per frame, chunk-zeroed (#36.1 keeps
                // the measured-modest win of paying this off the synth path). Cover the widest
                // LIVE window: chunk cadence + decode context — hurry flushes and the
                // first-chunk boundary are only ever SMALLER.
                int warmT = Math.Max(StreamChunkFrames, 16) + Cfg.MIMI_DECODE_CTX;
                var me = mimi.PreallocateYielding(warmT);
                while (IsReady && me.MoveNext()) { LastHeavyTick = "mimi_pre"; yield return null; }
                // ...and the wav-assembly accumulator the flushes harvest into (#31-R3), sized at
                // the clause cap like everything above, committed the same way.
                GrowWav(DefaultMaxFrames * Cfg.SAMPLES_PER_LATENT);
                mimi.ZeroTouch(wavAccum);
                yield return null;
                int voiceFrames = Math.Max((voicePrompt?.Length ?? 0) / Cfg.DIM, PREALLOC_VOICE_ROWS);
                int maxLp = 1 + voiceFrames + PREALLOC_TEXT_ROWS;
                var e = flm.PreallocateYielding(maxLp, maxLp + DefaultMaxFrames);
                while (IsReady && e.MoveNext()) { LastHeavyTick = "flm_pre"; yield return null; }
            }

            /// <summary>Emergency-flush hook (2026-07-30), set by PocketTTSVoice while it owns the
            /// stream: returns true while audible silence is imminent (playback gated, or the ring
            /// below <c>InferencePerf.TtsPanicFloorSeconds</c>). While true, the flush schedule
            /// ignores the chunk cadence and decodes every <c>StreamHurryMinFrames</c> accumulated
            /// latents — small lumps land in the ring NOW instead of one chunk-sized lump up to a
            /// second later, which is what the 0.25 s re-gate needs to actually resume on. Emitted
            /// samples are unchanged (windowed tail-exact decode — only the flush BOUNDARY moves,
            /// same argument as StreamFirstChunkFrames). Null (probes, raw callers) = pure cadence,
            /// so probe timing and parity runs are untouched. GPU-frame path only, like firstChunk.</summary>
            public Func<bool> StreamHurry;

            /// <summary>Smallest hurry flush. A flush costs ~16 heavy ticks whatever its size (see
            /// the chunk column's docs in BackendTradeoffTable), so 1-2 latent flushes are nearly
            /// all overhead; 4 latents = 0.32 s of audio — one re-gate's worth per emergency
            /// decode — is the floor of usefulness.</summary>
            public const int StreamHurryMinFrames = 4;

            public int StreamLastTokenCount { get; private set; }   // frames actually emitted (streaming)

            /// <summary>Streaming path only (bug C): per-frame GPU readbacks via AsyncGPUReadback —
            /// the coroutine yields instead of stalling the main thread on the whole queued GPU work
            /// (the ~90 ms conversation-start freeze + ~10 ms/frame steady-state stalls). Probes set
            /// false: sync GetData keeps deterministic timing and avoids spin-waits in their
            /// editor-synchronous MoveNext loops. Data is identical either way (same dispatches).</summary>
            public bool AsyncReadback = true;

            /// <summary>#29: yield sentinel meaning "END THE FRAME NOW". The voice pump's time
            /// budget measures CPU ISSUE time (~1 ms buys ~15 ms of GPU), so plain-null yields get
            /// re-entered until an entire prefill/flush lands in ONE frame anyway. GPU-heavy ticks
            /// (a prefill layer, a Mimi-decode slice) yield THIS; PumpPipeline breaks its budget
            /// loop on it — true one-heavy-tick-per-frame pacing. Cheap yields (AR bookkeeping,
            /// readback spins) stay null and keep packing under the budget.</summary>
            public static readonly object FrameBreak = new object();

            /// <summary>#29: yield sentinel meaning "a GPU readback is in flight — no work can be
            /// issued until it lands". The pump gives these a SHORT spin window (readbacks often
            /// complete mid-frame when the queue is shallow) and then cedes the frame — spinning
            /// them for the whole CPU budget was ~5 ms/frame of pure waste (the fat SPK+AUD band
            /// in the talk-perf report).</summary>
            public static readonly object GpuWait = new object();

            /// <summary>#29: MAC budget of one GPU-heavy pipeline tick (a prefill chunk / Mimi-decode
            /// slice). Hardware-tuned by the DIAL, not by measurement: it is the Backend Tradeoff
            /// tier's row (900M on the two low tiers, 4G at Very Fast), so slower cards get finer
            /// slices and faster ones coarser ones without anything having to discover that at
            /// runtime. Read-only on purpose (2026-07-27): PocketTTSVoice.CalibrateTickBudget used to
            /// walk this between 200M and 4G chasing a 3-7 ms measured tick cost, and since slice
            /// COUNT is derived from it, its shrink branch fed straight back into the ring starvation
            /// it existed to prevent — see BackendTradeoff.cs.</summary>
            public static long GpuMacsPerTick => BackendTradeoffTable.TtsMacsPerTick;

            /// <summary>#31-P: routes every eligible matmul (FlowLM backbone GEMVs + prefill GEMMs,
            /// flow head, mimi decoder_transformer) through the coalesced/fused kernel generation
            /// (see PocketTTSCS.compute "#31-P" section + DEEPOPT.md). Default ON. false = the
            /// pre-#31 LinearBias/LinearBiasQ8 + per-op flow head, kept for A/B probes and as the
            /// rollback (PocketTTSMimi.ForceLegacyKernels additionally forces the pre-#30 convs —
            /// the two switches bisect the #30 and #31 axes independently). Tick pacing is
            /// unaffected: slicing stays MAC-budgeted and InferencePerf AutoTune re-measures
            /// real per-tick cost, so faster kernels just converge to coarser slices.</summary>
            public static bool FastKernels2 = true;

            /// <summary>#31-R2 (layered on FastKernels2): GPU-resident AR frames — the whole
            /// token->transformer->eos->flow-head->latent chain stays on the GPU (feedback via a
            /// GPU latent buffer, eos+latent written to per-frame slots), the offline loop reads
            /// results back ONCE per ArBatchFrames block instead of TWICE PER FRAME, streaming
            /// makes one combined async readback per StreamArBatchFrames block (per frame until
            /// 2026-07-30 — see that field). Plus the fused
            /// transformer step (LN folded into the GEMV staging, residual adds folded into GEMV
            /// epilogues, slice+RoPE+KV-append as one kernel): ~49 dispatches/frame vs R1's ~96.
            /// Default ON; false restores the exact R1 dispatch list (three-tier bisect:
            /// legacy / FastKernels2 / FastKernels2+3).</summary>
            public static bool FastKernels3 = true;

            /// <summary>#31-R2: frames per GPU-resident offline batch (1..16). Larger = fewer
            /// pipeline drains but more post-EOS overshoot compute (up to K-1 discarded frames,
            /// ~2% of a 130-frame clip at 8). Streaming has its own K — see StreamArBatchFrames.</summary>
            public static int ArBatchFrames = 8;

            /// <summary>Streaming twin of ArBatchFrames (2026-07-30). #31-R2 left streaming
            /// per-frame "for latency" — and that per-frame readback turned out to be exactly what
            /// caps production on the reference GTX 1650: ~73 ms/latent while the LLM decodes
            /// (≈1 latent per Unity frame at decode-era frame times), i.e. ~1.0× playback, so the
            /// ring limit-cycles off zero and every bounce is an ~80 ms blip (2026-07-30 log:
            /// 15 bursts in one reply). K chained GPU-resident frames per combined [eos|latent]
            /// readback amortize that latency K-fold. EOS overshoot compute is bounded at K-1
            /// discarded frames per clause and emitted audio is identical by the same construction
            /// as the offline block loop (a frame's latent never depends on later frames; overshoot
            /// KV rows are never attended by emitted frames; issued frames never exceed maxFrames,
            /// so KV capacity needs are unchanged). Pacing stays honest: one FrameBreak per issued
            /// frame, so the per-frame GPU load the tier's tick cap admits is unchanged — the block
            /// only removes the readback stalls between frames.</summary>
            public static int StreamArBatchFrames = 4;

            /// <summary>TTFA ramp for the first streaming blocks of EACH clause (clamped to
            /// StreamArBatchFrames; later blocks run it flat). Block 0 = 1 keeps the first
            /// readback — and with it StreamFirstChunkFrames' first flush — exactly as early as
            /// the old per-frame path. null = flat K from the first block.</summary>
            public static int[] StreamArBatchRamp = { 1, 2 };

            /// <summary>#31-R3 lever 2 (TTFA ramp): frame counts for the FIRST offline blocks (each
            /// clamped to ArBatchFrames; later blocks use ArBatchFrames). Flat K=8 regressed the
            /// TTFA proxy 57→79 ms (first readback waits for 8 frames); {2,4} lands the first
            /// eos/latent readback after ~2 frames of GPU. null = flat K (the exact R2 behavior,
            /// kept for A/B).</summary>
            public static int[] ArBatchRamp = { 2, 4 };

            /// <summary>#31-R3 lever 1: interleave mimi with the AR loop instead of running it
            /// strictly after. The offline GPU-resident path issues DecodeWindowed's EXACT window
            /// schedule (chunk 64, ctx MIMI_DECODE_CTX) as soon as each window's latents are
            /// scanned, harvests every window's kept tail GPU-side into a persistent buffer
            /// (CopySlice — pure copy), and does ONE readback at the very end: same dispatches as
            /// the sequential windowed path, only interleaved, so given the same latents the wav
            /// is BIT-IDENTICAL to DecodeWindowed(chunk 64) — probe-gated at maxAbs == 0. Mimi's
            /// fat conv kernels fill the AR chain's inter-dispatch dependency bubbles and largely
            /// leave the critical path. false = the R2 sequential tail (A/B). Offline only —
            /// streaming already interleaves decode via its flush windows.</summary>
            public static bool OverlapMimi = true;
            const int MIMI_OVERLAP_CHUNK = 64;   // = DecodeWindowed's default chunk (schedule parity)

            // ---- #31-R2 instrumentation (probe-owned; zero overhead when PerfCounting == false).
            // FlowLM funnels EVERY dispatch through its Disp() wrapper and counts blocking/async
            // readbacks + uploads at their exact sites, so the parity probe can print per-frame
            // dispatch counts and sync attribution BEFORE/AFTER the R2 path (item 1 of the brief).
            public static bool PerfCounting = false;

            // Probe-only: take the legacy ONE-FRAME-AT-A-TIME AR loop instead of the shipped
            // GPU-resident K-frame batches. Same math and same KV path (the legacy branch still
            // calls DecodeStepKV) — but it is the only branch that separates the backbone step from
            // the flow head, because the batched branch issues K frames back-to-back with no sync in
            // between and there is nothing to attribute. Pair it with OverlapMimi = false, or Mimi's
            // cost hides inside the AR loop's readback waits and DecodeMs under-reports it.
            // Its TOTAL is therefore NOT the shipped total: use the shares, scale them onto a clean
            // production-settings run. Never set outside a probe.
            public static bool ForceLegacyArLoop = false;
            public static long StatDispatches, StatBlockingReads, StatAsyncReads, StatUploads;
            public static double StatReadWaitMs;                       // ms spent inside blocking GetData
            public static double StatTokenCpuMs, StatDecodeCallMs, StatFlowCallMs;   // legacy-loop split
            public static long StatLoopStartDisp, StatLoopStartReads, StatLoopStartUps;  // post-prefill marks
            public static void StatReset()
            {
                StatDispatches = StatBlockingReads = StatAsyncReads = StatUploads = 0;
                StatReadWaitMs = StatTokenCpuMs = StatDecodeCallMs = StatFlowCallMs = 0;
                StatLoopStartDisp = StatLoopStartReads = StatLoopStartUps = 0;
            }

            /// <summary>#29 spike attribution: the last pipeline stage a pump tick worked on
            /// ("clause_start", "prefill", "ar_decode", "ar_flowhead", "mimi_decode", "flush_push",
            /// "readback_hardwait"). Written by SynthesizeStreaming's yield sites; diagnostics
            /// probes read-and-clear it per frame to attribute slow frames to a stage. Inert
            /// otherwise (one static string assign per tick).</summary>
            /// <remarks>Readers that do NOT clear it must check <see cref="LastHeavyTickFrame"/>:
            /// after a reply's final flush the field keeps its last value for the rest of the
            /// session, and the 2026-08-02 spike hunt spent a day chasing a "flush_push storm"
            /// that was mostly this staleness, not flushes.</remarks>
            public static string LastHeavyTick
            {
                get => lastHeavyTick;
                set { lastHeavyTick = value; LastHeavyTickFrame = Time.frameCount; }
            }
            static string lastHeavyTick;

            /// <summary>Time.frameCount at the last write of <see cref="LastHeavyTick"/> — a tag is
            /// only evidence about a frame it was written in (±1 for Update-order skew).</summary>
            public static int LastHeavyTickFrame { get; private set; } = -1;

            public bool IsReady => weights.IsReady;
            public long WeightBytes => weights.BytesTotal;   // resident weight footprint (fp16 vs int8 delta)

            /// <summary>#32: prompt rows the LAST SynthesizeStreaming actually prefilled — the whole
            /// prefix (1 + voiceFrames + textIds) on a cold clause / voice swap, only textIds.Length
            /// when the voice-prompt K/V was retained. Read by the prompt-cache probe and useful in
            /// the TTFA breakdown; nothing behavioural hangs off it.</summary>
            public int LastPrefillRows { get; private set; }

            /// <summary>#32: FrameBreak ticks the last clause's prefill cost, and the wall time to the
            /// end of it. TICKS is the number that matters in the game, but NOT one-for-one with frames:
            /// the pump does NOT end the frame on a FrameBreak (corrected 2026-07-28) — it counts them
            /// and breaks at maxHeavyTicks, which is 6 on Very Smooth/Smooth down to 2 on Very Fast
            /// during a clause prefill. So 24 ticks is 4-12 frames, 67-200 ms at 60 fps, not the ~400 ms
            /// an earlier version of this comment claimed. Distinct from PrefillMs, which stays the
            /// OFFLINE path's breakdown; and note LastPrefillMs is dispatch-ISSUE time, not GPU
            /// completion, because the dispatches are queued rather than awaited.</summary>
            public int LastPrefillTicks { get; private set; }
            public float LastPrefillMs { get; private set; }

            /// <summary>#32: rows of voice-prompt K/V currently retained in the flow LM (0 = cold).</summary>
            public int RetainedPromptRows => flm.RetainedPromptRows;

            public PocketTTS(string weightsDir = null)
            {
                // resolved HERE (not just in PocketTTSWeights) — the tokenizer + encoder sibling
                // files are read off this field too (player builds: StreamingAssets).
                this.weightsDir = DeepUnityMeta.ResolvePath(weightsDir ?? Cfg.WEIGHTS_DIR_FP16);
                weights = new PocketTTSWeights(this.weightsDir, beginLoad: false);
                flm = new PocketTTSFlowLM(weights);
                mimi = new PocketTTSMimi(weights);
            }

            /// <summary>SentencePiece encode text -> ids (P7; lazy-loads tokenizer.vocab.json from the
            /// weights dir). English defaults (pad=false, removeSemicolons=false). Reproduces the
            /// Python conditioner.prepare(prepare_text_prompt(...)) — parity-gated in P7 probe.</summary>
            public int[] Tokenize(string text, bool pad = false, bool removeSemicolons = false)
            {
                if (tokenizer == null)
                {
                    string vp = System.IO.Path.Combine(weightsDir, "tokenizer.vocab.json");
                    tokenizer = new PocketTTSTokenizer(vp);
                }
                return tokenizer.Encode(text, pad, removeSemicolons);
            }

            /// <summary>Game path: start streaming weights to GPU (budgeted per-frame pump). After
            /// this, poll IsReady before Synthesize. Loads the small CPU-side tensors once ready.</summary>
            public void BeginLoad()
            {
                weights.BeginLoad();
                if (weights.IsReady && embMean == null) LoadCpuTensors();
            }

            /// <summary>Load-on-approach spread over ~targetSeconds (tiny per-frame upload slices).
            /// Never SLOWS a faster stream already in flight (#36.4): the scene-start warm cycle
            /// streams at full rate from frame 0, and the zone's slow call used to overwrite that
            /// budget mid-upload — throttling the exact stream it was trying to guarantee.</summary>
            public void SlowPrefetch(float targetSeconds)
            {
                if (targetSeconds > 0.01f && weights.BytesTotal > 0)
                {
                    long slow = Math.Max(1, (long)(weights.BytesTotal / (targetSeconds * 60f)));
                    bool inFlight = !weights.IsReady && weights.BytesUploaded > 0;
                    if (!(inFlight && weights.BudgetBytesPerFrame > slow))
                        weights.BudgetBytesPerFrame = slow;
                }
                BeginLoad();
            }

            /// <summary>Conversation-open boost: lift an in-flight budgeted upload to the tier's
            /// full rate — the pump samples BudgetBytesPerFrame live, so it takes effect next
            /// frame. Until 2026-07-30 nothing ever raised a SlowPrefetch budget again, so a
            /// player who opened the dialogue during the walk-up spoke over a still-streaming
            /// voice for its whole remaining window (log: `pockettts fully streamed` landing
            /// seconds into the conversation, with the upload competing against decode + synth
            /// for the GPU — the 0.60 s of dry bursts in that reply). No-op once resident.</summary>
            public void BoostPrefetch()
            {
                weights.BudgetBytesPerFrame = Math.Max(1, BackendTradeoffTable.FetchBytesPerFrame);
                BeginLoad();
            }

            /// <summary>Release GPU weights (budgeted when slow=true). A later BeginLoad re-streams.</summary>
            public void Defetch(bool slow = true)
            {
                // #32: the KV caches outlive a defetch (they are ours, not the weight pump's) and the
                // re-streamed weights carry the same values, so the retained rows would still be
                // numerically right — drop them anyway. Residency churn is exactly where "is this
                // cache still meaningful?" stops being locally verifiable, and one extra full prefill
                // per walk-away is free next to a weight re-upload.
                flm.InvalidatePromptKV();
                weights.Defetch(slow ? weights.BudgetBytesPerFrame : 0);
            }

            // emb_mean/std + voice/bbv are tiny CPU reads (ReadFloats streams from disk directly,
            // independent of the GPU upload pump) — safe to load as soon as the manifest is parsed.
            void LoadCpuTensors()
            {
                embMean = weights.ReadFloats("flow_lm.emb_mean");
                embStd = weights.ReadFloats("flow_lm.emb_std");
                bbv = weights.ReadFloats("flow_lm.bos_before_voice");
                voicePrompt ??= weights.ReadFloats("voices/jean/audio_prompt");
                // #32: row 0 of the retained prompt is bbv, and this just replaced it with a fresh
                // array. The voicePrompt identity key cannot see that (the `??=` keeps the SAME array),
                // so invalidate explicitly — the retained rows depend on both tensors.
                flm.InvalidatePromptKV();
            }

            /// <summary>Editor/probe: synchronous blocking load of everything.</summary>
            public void LoadBlocking()
            {
                weights.LoadBlocking();   // all tensors
                LoadCpuTensors();         // emb_mean/std [32], bbv [1024], voices/jean/audio_prompt [125*1024]
            }

            public string CurrentVoice { get; private set; } = "jean";

            /// <summary>Editor/probe diagnostics: the audio_prompt currently bound [frames*1024]
            /// (baked voice or clone). Read-only snapshot for validation tooling.</summary>
            public float[] CurrentVoicePrompt => voicePrompt;

            /// <summary>Validation tooling: bind a raw audio_prompt [frames*1024] directly, past the
            /// baked voices and the clone cache. Only the export's baked voices are reachable through
            /// SetVoice (this export ships exactly one), so the #32 prompt-KV probe needs this to
            /// prove the retained-cache key FALLS BACK on a second speaker of the SAME row count —
            /// the one case a "have I prefilled before" flag would get wrong. Same PromptIsValid gate
            /// as the clone path; a rejected prompt leaves the current voice bound.</summary>
            public bool BindRawVoicePrompt(float[] prompt, string label)
            {
                if (!PromptIsValid(prompt)) return false;
                // Invalidate EXPLICITLY, at the assignment, rather than relying on the new array failing
                // ReferenceEquals (review 2026-07-28). Reference identity provably holds today — every
                // path that assigns voicePrompt allocates fresh, no indexed write into it exists — but
                // that is a property of code far from here, and the failure it would allow is a retained
                // cache serving the WRONG SPEAKER with no error. Three assignment sites, three calls; it
                // costs nothing because they only run on a real swap.
                voicePrompt = prompt;
                flm?.InvalidatePromptKV();
                CurrentVoice = string.IsNullOrEmpty(label) ? "raw" : label;
                return true;
            }

            /// <summary>Rebind the baked voice (cheap CPU read of voices/&lt;name&gt;/audio_prompt — no
            /// GPU reload). Unknown names fall back to the currently-loaded voice with a warning
            /// (only baked voices in the export are available; bake more with import_params.py pocket-tts --voice).</summary>
            // #36.2: SetVoice's ReadFloats is a cold disk read + 128k half decodes (~200 ms, the
            // `bind` frame the session warmup pays binding the DEFAULT voice — the clone path was
            // sliced separately and this was what the tag kept naming). Prepare/warmup kick this
            // on zone entry; ReadFloats is pure CPU over a read-only manifest, thread-safe.
            readonly System.Collections.Concurrent.ConcurrentDictionary<string, float[]> bakedPromptCache
                = new System.Collections.Concurrent.ConcurrentDictionary<string, float[]>();

            public void PreloadBakedVoiceAsync(string name)
            {
                if (string.IsNullOrEmpty(name) || bakedPromptCache.ContainsKey(name)) return;
                string tensor = $"voices/{name}/audio_prompt";
                if (!weights.Has(tensor)) return;
                System.Threading.Tasks.Task.Run(() =>
                { try { bakedPromptCache[name] = weights.ReadFloats(tensor); } catch { /* falls back to the sync read */ } });
            }

            public void SetVoice(string name)
            {
                if (string.IsNullOrEmpty(name) || name == CurrentVoice) return;
                if (!weights.Has($"voices/{name}/audio_prompt"))
                {
                    ConsoleMessage.Warning($"pocket-tts: baked voice '{name}' not found in {weightsDir} " +
                                           $"(only exported voices are available) — keeping '{CurrentVoice}'.");
                    return;
                }
                if (!bakedPromptCache.TryGetValue(name, out float[] vp))
                    bakedPromptCache[name] = vp = weights.ReadFloats($"voices/{name}/audio_prompt");
                voicePrompt = vp;
                flm?.InvalidatePromptKV();   // see BindRawVoicePrompt: invalidate at the assignment
                CurrentVoice = name;
            }

            // ================= P8: runtime voice cloning + cache =================
            // A reference clip -> audio_prompt [T,1024] via the Mimi encoder + speaker_proj. Cached
            // to disk by a content hash of the samples so re-cloning the same clip is a fast load
            // (~few hundred KB) instead of a full re-encode. Requires the encoder weights in the dir
            // (import_params.py pocket-tts --include-encoder); baked voices work without them.

            public bool HasEncoder => weights.Has("mimi/encoder/model/0/conv.weight")
                                   && weights.Has("mimi/downsample/conv/conv.weight");

            string CacheDir => System.IO.Path.Combine(Application.persistentDataPath, "pockettts_voices");

            /// <summary>Content hash of the reference samples — SHA-256 hex (64 chars), matching the
            /// Assets/Resources/Cache convention (the system-prompt KV folders use the same form).</summary>
            static string HashSamples(float[] s)
            {
                var buf = new byte[s.Length * 4];
                Buffer.BlockCopy(s, 0, buf, 0, buf.Length);
                using (var sha = System.Security.Cryptography.SHA256.Create())
                {
                    byte[] h = sha.ComputeHash(buf);
                    var sb = new System.Text.StringBuilder(64);
                    for (int i = 0; i < h.Length; i++) sb.Append(h[i].ToString("x2"));
                    return sb.ToString();
                }
            }

            /// <summary>Resources folder holding editor-baked voice-clone prompts as .bytes TextAssets
            /// — the SHARED content-addressed cache (same folder + SHA-256 naming as the system-prompt
            /// KV caches), the shipping tier of the clone cache.</summary>
            public const string RES_VOICE_DIR = "Cache";

            /// <summary>Where the last CloneVoice found its prompt: "resources" (editor-baked, ships
            /// in builds — the AUTHORITATIVE tier) | "persistent" (runtime disk cache) | "encoded"
            /// (computed now + cached).</summary>
            public string LastCloneSource { get; private set; }

            /// <summary>Sanity gate on an audio_prompt before it is bound or cached. A GPU device
            /// reset (TDR) mid-encode makes every dispatch silently no-op and GetData return zeros —
            /// binding that yields pure gibberish speech, and caching it POISONS the voice on disk
            /// forever (root cause of the female-voice-2 gibberish, 2026-07-22: the persistent .bin
            /// was 512 KB of exact 0.0f while the editor bake of the same clip was healthy).
            /// Healthy prompts have RMS ~0.04-0.09; reject NaN/Inf, empty, non-[T,1024] and
            /// near-silent (all-zero) buffers.</summary>
            static bool PromptIsValid(float[] p)
            {
                if (p == null || p.Length == 0 || p.Length % Cfg.DIM != 0) return false;
                double acc = 0;
                for (int i = 0; i < p.Length; i++)
                {
                    float v = p[i];
                    if (float.IsNaN(v) || float.IsInfinity(v)) return false;
                    acc += (double)v * v;
                }
                return Math.Sqrt(acc / p.Length) > 1e-4;
            }

            // Content-addressed: the key IS the SHA-256 of the capped 24 kHz wav — labels don't enter
            // the key (identical audio = identical cache entry, regardless of clip name/renames).
            static string KeyFor(float[] wav24k) => HashSamples(wav24k);

            /// <summary>The cache key CloneVoice(clip) will use — lets editor tooling check for a baked
            /// Resources entry without touching the model. Null if the clip isn't readable.</summary>
            public static string CloneKey(AudioClip clip) => CloneKey(clip, out _);

            /// <summary>Cache key + the exact crop CloneVoice(clip) will apply. The key hashes the
            /// CROPPED wav, so the cached latents cover precisely crop.croppedSeconds of audio —
            /// editor tooling shows the real cropped length from this, without touching the model.</summary>
            public static string CloneKey(AudioClip clip, out CropInfo crop)
            {
                crop = default;
                float[] mono = ClipToMono(clip);
                if (mono == null) return null;
                return KeyFor(PrepRef(mono, clip.frequency, out crop));
            }

            /// <summary>Editor precompute: encode a reference wav to the raw audio_prompt bytes the
            /// cache stores (byte-identical to WritePromptBin's file / a Resources .bytes asset).
            /// Weights (incl. the Mimi encoder) must be resident. Null on failure.</summary>
            public byte[] PrecomputePromptBytes(float[] samples, int sampleRate, string label, out string key)
                => PrecomputePromptBytes(samples, sampleRate, label, out key, out _);
            public byte[] PrecomputePromptBytes(float[] samples, int sampleRate, string label, out string key, out CropInfo crop)
            {
                key = null;
                crop = default;
                if (samples == null || samples.Length == 0) return null;
                float[] wav = PrepRef(samples, sampleRate, out crop);   // resample to 24k + cap at MAX_REF_SECONDS
                key = KeyFor(wav);   // content-addressed (label only names the voice, never the key)
                float[] prompt = EncodeToPrompt(wav);
                if (prompt != null && !PromptIsValid(prompt))
                {
                    ConsoleMessage.Warning($"pocket-tts precompute: the Mimi encoder returned a silent/invalid prompt for " +
                                           $"'{label}' (GPU device reset mid-encode?). Refusing to bake it — retry the bake.");
                    return null;
                }
                return prompt == null ? null : PromptToBytes(prompt);
            }
            public byte[] PrecomputePromptBytes(AudioClip clip, out string key)
                => PrecomputePromptBytes(clip, out key, out _);
            public byte[] PrecomputePromptBytes(AudioClip clip, out string key, out CropInfo crop)
            {
                key = null;
                crop = default;
                float[] mono = ClipToMono(clip);
                return mono == null ? null : PrecomputePromptBytes(mono, clip.frequency, clip.name, out key, out crop);
            }

            /// <summary>Clone a voice from an AudioClip (any sample rate; multi-channel down-mixed to
            /// mono; a long clip is pause-aware-cropped near MAX_REF_SECONDS — see PrepRef). Caches +
            /// binds as the current voice.</summary>
            public bool CloneVoice(AudioClip clip, string label = null)
            {
                float[] mono = ClipToMono(clip);
                if (mono == null)
                {
                    ConsoleMessage.Warning($"pocket-tts CloneVoice: clip '{(clip ? clip.name : "null")}' has no readable " +
                                           "sample data — set its import Load Type to 'Decompress On Load'. Keeping the current voice.");
                    return false;
                }
                return CloneVoice(mono, clip.frequency, label ?? clip.name);
            }

            /// <summary>#36.2: CloneVoice(clip) spread across frames — the sync form's three
            /// CPU-heavy stages (decompress-read + mono mix, resample, SHA-256; ~1 MB of managed
            /// churn EACH) were together the ~219 ms `bind` frame at zone entry. One stage per
            /// frame; the cache/bind tail is a cache-hit-cheap single frame. Same fallback
            /// contract as CloneVoice: reports false and the caller keeps/sets the baked voice.</summary>
            public IEnumerator CloneVoiceYielding(AudioClip clip, string label, Action<bool> done)
            {
                if (!IsReady || clip == null) { done?.Invoke(false); yield break; }
                // #36.3 instrumentation: after eight structural fixes the ~200-250 ms `bind`
                // frame still would not NAME itself — every stage below now reports its own CPU
                // ms AND the duration of the frame it ended (frame ≫ cpu = the cost is GPU/driver
                // drain at present, not the stage). One session log line; remove when solved.
                var stageSw = new System.Diagnostics.Stopwatch();
                var report = new System.Text.StringBuilder("[PocketTTS] clone stages cpu/frame ms: ");
                // ClipToMono WAITS on a still-decompressing clip with a blocking sleep loop (#36.2
                // round 3) — wait HERE, one frame at a time; ClipToMono's own wait then no-ops.
                if (clip.loadState == AudioDataLoadState.Unloaded) clip.LoadAudioData();
                while (clip.loadState == AudioDataLoadState.Loading) yield return null;
                LastHeavyTick = "bind";
                // #36.3 round 2: GetData on this MP3 DECODES on the spot (~190 ms of CPU in one
                // call, measured — load type and background loading change nothing about it).
                // Chunked reads pay the same decode ~1 s of audio per frame. Each chunk's buffer
                // is sized EXACTLY (GetData wraps around the clip when the buffer outruns it —
                // a same-size tail buffer would silently mix in samples from the clip's start).
                float[] mono; double cpu = 0;
                {
                    int samples = clip.samples, ch = clip.channels;
                    if (samples == 0) { done?.Invoke(false); yield break; }
                    mono = new float[samples];
                    const int CHUNK = 48000;
                    float[] buf = null;
                    bool fail = false;
                    for (int off = 0; off < samples && !fail; off += CHUNK)
                    {
                        LastHeavyTick = "bind";
                        stageSw.Restart();
                        int n = Math.Min(CHUNK, samples - off);
                        if (buf == null || buf.Length != n * ch) buf = new float[n * ch];
                        if (!clip.GetData(buf, off)) { fail = true; break; }
                        if (ch == 1) Array.Copy(buf, 0, mono, off, n);
                        else
                            for (int i = 0; i < n; i++)
                            {
                                float acc = 0f; int b = i * ch;
                                for (int c = 0; c < ch; c++) acc += buf[b + c];
                                mono[off + i] = acc / ch;
                            }
                        cpu += stageSw.Elapsed.TotalMilliseconds;
                        yield return null;
                    }
                    if (fail)
                    {
                        ConsoleMessage.Warning($"pocket-tts CloneVoice: clip '{clip.name}' has no readable " +
                                               "sample data — set its import Load Type to 'Decompress On Load'. Keeping the current voice.");
                        done?.Invoke(false); yield break;
                    }
                }
                report.Append($"mono(chunked) {cpu:0.0}/{Time.unscaledDeltaTime * 1000f:0.0} | ");
                LastHeavyTick = "bind";
                stageSw.Restart();
                float[] wav = PrepRef(mono, clip.frequency, out CropInfo crop);
                cpu = stageSw.Elapsed.TotalMilliseconds; yield return null;
                report.Append($"prep {cpu:0.0}/{Time.unscaledDeltaTime * 1000f:0.0} | ");
                LastHeavyTick = "bind";
                stageSw.Restart();
                string key = KeyFor(wav);
                cpu = stageSw.Elapsed.TotalMilliseconds; yield return null;
                report.Append($"sha {cpu:0.0}/{Time.unscaledDeltaTime * 1000f:0.0} | ");
                // async warm of the tier-1 baked cache (#36.2 round 2); the sync Load in the tail
                // then hits Unity's loaded-asset cache.
                stageSw.Restart();
                var rq = Resources.LoadAsync<TextAsset>(RES_VOICE_DIR + "/" + key);
                while (rq != null && !rq.isDone) yield return null;
                cpu = stageSw.Elapsed.TotalMilliseconds; yield return null;
                report.Append($"resload(async) {cpu:0.0}/{Time.unscaledDeltaTime * 1000f:0.0} | ");
                LastHeavyTick = "bind";
                stageSw.Restart();
                bool ok = IsReady && CloneVoicePrepared(wav, key, crop, label ?? clip.name);
                report.Append($"tail {stageSw.Elapsed.TotalMilliseconds:0.0}");
                Debug.Log(report.ToString());
                done?.Invoke(ok);
            }

            /// <summary>Max reference length for cloning. HARD TECHNICAL CEILING — do NOT raise.
            /// EXACT single-dimension GPU-dispatch boundary: the encoder's widest pass (stage-0,
            /// 64 ch x wavLen elements) is dispatched at 256 threads/group, so it stays within ONE
            /// dispatch dimension only while ceil(64*wavLen / 256) &lt;= 65535 (the D3D11 per-dimension
            /// group cap) -> wavLen &lt;= 262140 samples -> 262140 / 24000 Hz = 10.9225 s exactly. Past
            /// that, Dispatch1D falls back to a Y-spill (works to ~30 s in theory) which we DELIBERATELY
            /// never lean on. Set to 10.8 s: safely under the 10.9225 s boundary, no overshoot. It's
            /// moot for quality anyway -- the model's native audio_prompt is ~10 s (125 latent frames;
            /// every Kyutai baked voice is exactly this) and Kyutai embeds speakers from ~10 s, so more
            /// reference barely improves timbre while lengthening EVERY reply's prefill (the audio_prompt
            /// is prepended to the FlowLM each utterance). A longer clip is HARD-CUT at exactly this
            /// cap (the pause-aware detector below exists but is DISABLED — USE_PAUSE_AWARE_CROP).
            /// The cache key hashes the CROPPED wav, so bake and runtime always agree on the key.</summary>
            public const float MAX_REF_SECONDS = 10.8f;

            // Pause-aware crop: a long reference is cut at a NATURAL PAUSE near the cap instead of
            // mid-word — a chopped word in the prompt conditions the voice on a truncation artifact.
            // Never cropped shorter than MIN_CROP_SECONDS; a "pause" is >= 3 consecutive 30 ms hops
            // whose RMS sits under 15% of the clip's mean hop-RMS (stop-consonant closures are
            // shorter, real pauses are longer).
            // DISABLED (user 2026-07-22): references are cut at exactly MAX_REF_SECONDS, no pause
            // auto-detection. The detector is kept, not removed — flip this flag to re-enable it.
            // NOTE: flipping it changes the cropped wav for >cap clips, hence their cache keys.
            static readonly bool USE_PAUSE_AWARE_CROP = false;
            public const float MIN_CROP_SECONDS = 7f;
            const float PAUSE_WIN_SECONDS = 0.03f;

            /// <summary>What PrepRef did to a reference clip — surfaced so the inspector (precompute
            /// button) and the encode-time log can report the REAL cropped length. The cache key
            /// hashes the cropped wav, so the cached latents cover exactly croppedSeconds of audio.</summary>
            public struct CropInfo
            {
                public float totalSeconds;    // reference length (after resample to 24 kHz)
                public float croppedSeconds;  // length actually used (== totalSeconds when uncropped)
                public bool cropped;          // clip exceeded MAX_REF_SECONDS
                public bool atPause;          // cut landed on a natural pause (else hard cap cut)
            }

            static float[] PrepRef(float[] samples, int sampleRate) => PrepRef(samples, sampleRate, out _);

            static float[] PrepRef(float[] samples, int sampleRate, out CropInfo crop)
            {
                float[] wav = sampleRate == Cfg.SAMPLE_RATE ? samples : Resample(samples, sampleRate, Cfg.SAMPLE_RATE);
                int cap = (int)(MAX_REF_SECONDS * Cfg.SAMPLE_RATE);
                crop.totalSeconds = wav.Length / (float)Cfg.SAMPLE_RATE;
                if (wav.Length <= cap)
                {
                    crop.croppedSeconds = crop.totalSeconds;
                    crop.cropped = false;
                    crop.atPause = false;
                    return wav;
                }
                int cut = USE_PAUSE_AWARE_CROP ? FindPauseCut(wav, cap) : cap;
                crop.cropped = true;
                crop.atPause = cut < cap;
                crop.croppedSeconds = cut / (float)Cfg.SAMPLE_RATE;
                var t = new float[cut];
                Array.Copy(wav, t, cut);
                return t;
            }

            // Latest (closest-to-cap) mid-pause sample index in [MIN_CROP_SECONDS, cap), or cap
            // when the tail has no detectable pause. Deterministic — the cache key hashes the
            // cropped wav, so bake and runtime always agree.
            static int FindPauseCut(float[] wav, int cap)
            {
                int win = (int)(PAUSE_WIN_SECONDS * Cfg.SAMPLE_RATE);
                int start = (int)(MIN_CROP_SECONDS * Cfg.SAMPLE_RATE);
                int hops = (cap - start) / win;
                if (win <= 0 || hops < 3) return cap;

                double sum = 0; int n = 0;   // clip's own speech level (mean hop-RMS over the capped head)
                for (int off = 0; off + win <= cap; off += win) { sum += HopRms(wav, off, win); n++; }
                float thr = (float)(sum / Math.Max(n, 1)) * 0.15f;

                int run = 0, cut = -1;
                for (int h = 0; h < hops; h++)
                {
                    if (HopRms(wav, start + h * win, win) < thr)
                    {
                        // middle hop of the latest >=3-quiet run: inside the pause, clear of the
                        // previous word's decay tail
                        if (++run >= 3) cut = start + (h - 1) * win + win / 2;
                    }
                    else run = 0;
                }
                return cut > 0 ? cut : cap;
            }

            static float HopRms(float[] wav, int off, int n)
            {
                double acc = 0;
                for (int i = 0; i < n; i++) { float v = wav[off + i]; acc += v * v; }
                return (float)Math.Sqrt(acc / n);
            }

            /// <summary>AudioClip -> mono float[] (average mixdown). Null if the clip is null or its
            /// sample data isn't readable (Load Type must keep samples accessible, e.g. Decompress
            /// On Load — streamed clips can't be cloned).</summary>
            public static float[] ClipToMono(AudioClip clip)
            {
                if (clip == null) return null;
                // With 'Preload Audio Data' off (the importer default) a clip that hasn't played
                // yet isn't loaded — clip.samples reads 0 and GetData returns nothing, which made
                // the inspector report a "0.0s" reference (the Gowry regression). Force the load
                // and wait briefly; clips are seconds long, and a failed load exits immediately.
                if (clip.loadState != AudioDataLoadState.Loaded || clip.samples == 0)
                {
                    clip.LoadAudioData();
                    int t0 = System.Environment.TickCount;
                    while (clip.loadState == AudioDataLoadState.Loading &&
                           System.Environment.TickCount - t0 < 3000)
                        System.Threading.Thread.Sleep(5);
                }
                if (clip.loadState != AudioDataLoadState.Loaded || clip.samples == 0) return null;
                var data = new float[clip.samples * clip.channels];
                if (!clip.GetData(data, 0)) return null;
                if (clip.channels == 1) return data;
                var mono = new float[clip.samples];
                for (int i = 0; i < clip.samples; i++)
                {
                    float acc = 0f; int b = i * clip.channels;
                    for (int c = 0; c < clip.channels; c++) acc += data[b + c];
                    mono[i] = acc / clip.channels;
                }
                return mono;
            }

            static bool _kernelsPrewarmed;

            /// <summary>Weights-FREE kernel precompile — the frame-0 counterpart to the LLMs'
            /// Qwen3_5Model.PrewarmKernels (drained in NPCChatBase.Awake, hidden in the scene-load
            /// blackout). The driver compiles each compute kernel's ISA on its FIRST dispatch; this
            /// dispatches every PocketTTSCS kernel once with ZEROED size uniforms (so every thread
            /// early-outs on its `idx >= size` guard — safe with dummy buffers, no OOB) and a
            /// distinct dummy bound to every buffer property (D3D11 forbids one UAV in two slots).
            /// Needs NO weights and NO PocketTTS instance, so it runs at scene start, long before
            /// any voice streams in — unlike PocketTTSVoice.PrewarmKernels, which does a REAL "Hi."
            /// synthesis and must wait for IsReady (that pass now only warms real buffer/KV paths;
            /// the shader-compile cost it used to carry is paid here at frame 0). Static-guarded:
            /// compiles once per session.</summary>
            public static IEnumerator PrewarmKernels()
            {
                if (_kernelsPrewarmed) yield break;
                _kernelsPrewarmed = true;

                ComputeShader shader = DeepUnityMeta.PocketTTSCS;
                string[] kernels =
                {
                    "CopyBuffer", "CopySlice", "SliceCols", "ZeroBuffer", "AddResidual", "ScaleBuf",
                    "ChannelScaleAdd", "Activate", "LinearBias", "LinearBiasQ8", "Conv1D", "Conv1DTiled",
                    "ConvTranspose1D", "ConvTranspose1DTiled", "ConvTranspose1DGrouped", "LayerNormT",
                    "ApplyRoPE", "CausalAttention", "CausalAttentionLegacy", "CausalAttentionKV",
                    "AppendKV", "Modulate", "GateAdd", "RMSNormFlow",
                    // #31-P (all degenerate at zeroed uniforms: in_dim/out_dim/seq_len/norm_dim = 0)
                    "LinearBiasCoal", "LinearBiasQ8Coal", "LinearBiasGemm", "LinearBiasQ8Gemm",
                    "FlowResBlockFused", "FlowResBlockFusedQ8", "FlowFinalFused", "FlowFinalFusedQ8",
                    // #31-R2 (degenerate at in_dim/out_dim/num_heads = 0)
                    "Gemv16", "GemvQ8", "GemvLN16", "GemvLNQ8",
                    "ARQkvPrep", "AREosNorm", "AREosNormQ8", "ARCommit",
                };
                // every buffer property in PocketTTSCS.compute — distinct dummy each (one UAV per slot)
                string[] bufs =
                {
                    "AttendedValues", "K", "KCache", "Q", "V", "VCache", "W", "W_bias", "W_scales",
                    "X", "Y", "buf", "buf_a", "buf_b", "ch_scale", "inout_buf", "ln_beta", "ln_gamma",
                    "mod_vec", "norm_input", "norm_output", "rms_alpha",
                    "W2", "W_bias2", "W_scales2", "W3", "W_bias3", "W_scales3",   // #31-P fused slots
                };
                // every integer uniform that gates a thread guard — zero them so all dispatches degenerate
                string[] zeroUniforms =
                {
                    "seq_len", "in_len", "in_dim", "out_dim", "conv_kernel", "conv_stride",
                    "conv_dilation", "pad_left", "norm_dim", "num_heads", "head_dim", "buffer_size",
                    "copy_src_offset", "copy_dst_offset", "n_groups", "pos_offset", "kv_len",
                    "elem_offset", "attn_context", "mod_shift_off", "mod_scale_off", "mod_gate_off",
                    "gemv_mode",   // #31-R2
                };

                var dummies = new ComputeBuffer[bufs.Length];
                for (int i = 0; i < dummies.Length; i++)
                    dummies[i] = new ComputeBuffer(16384, 4, ComputeBufferType.Structured);   // 64 KB: covers every real-mini index below
                foreach (string u in zeroUniforms) shader.SetInt(u, 0);

                foreach (string name in kernels)
                {
                    int k = shader.FindKernel(name);
                    for (int i = 0; i < bufs.Length; i++) shader.SetBuffer(k, bufs[i], dummies[i]);
                    shader.Dispatch(k, 1, 1, 1);   // one compile per frame when pumped as a coroutine
                    yield return null;
                }

                // #36.2 second pass — REAL-mini sizes. The zeroed pass above compiles the blob,
                // but this driver defers part of the work (finalize/regalloc) until a dispatch
                // whose threads actually EXECUTE — measured as the first real prefill tick
                // costing ~146-177 ms at zone entry no matter what else was prewarmed (weights
                // touched, scratch committed, kernels "compiled"). Two rows at dim 64 keeps every
                // index formula far inside the 16 K-element dummies (W: 64x64 = 4 K; KCache:
                // kv_len*heads*hd = 128; conv out ≲ 700), executes real loads/stores of garbage,
                // and discards into the same dummies. Values are irrelevant, execution is not.
                (string u, int v)[] realMini =
                {
                    ("seq_len", 2), ("in_len", 8), ("in_dim", 64), ("out_dim", 64),
                    ("conv_kernel", 3), ("conv_stride", 1), ("conv_dilation", 1), ("pad_left", 1),
                    ("norm_dim", 64), ("num_heads", 2), ("head_dim", 32), ("buffer_size", 128),
                    ("copy_src_offset", 0), ("copy_dst_offset", 0), ("n_groups", 2),
                    ("pos_offset", 0), ("kv_len", 2), ("elem_offset", 0), ("attn_context", 0),
                    ("mod_shift_off", 0), ("mod_scale_off", 0), ("mod_gate_off", 0), ("gemv_mode", 0),
                };
                foreach (var (u, v) in realMini) shader.SetInt(u, v);
                foreach (string name in kernels)
                {
                    int k = shader.FindKernel(name);
                    for (int i = 0; i < bufs.Length; i++) shader.SetBuffer(k, bufs[i], dummies[i]);
                    shader.Dispatch(k, 2, 1, 1);
                    yield return null;
                }
                foreach (string u in zeroUniforms) shader.SetInt(u, 0);   // leave no real-mini residue

                foreach (var d in dummies) d.Release();
            }

            /// <summary>Clone a voice from a reference clip -> audio_prompt [frames*1024], cache it by
            /// content hash, and bind it as the current voice. Cache hit = instant load; miss =
            /// encode once + save. Returns false if the encoder weights aren't in this dir.</summary>
            public bool CloneVoice(float[] samples, int sampleRate, string label = null)
            {
                if (!IsReady) { ConsoleMessage.Warning("pocket-tts CloneVoice: weights not resident yet."); return false; }
                if (samples == null || samples.Length == 0) return false;

                float[] wav = PrepRef(samples, sampleRate, out CropInfo crop);   // resample to 24k + cap at MAX_REF_SECONDS
                string key = KeyFor(wav);   // content-addressed (label only names the voice, never the key)
                return CloneVoicePrepared(wav, key, crop, label);
            }

            /// <summary>#36.2: the tail of CloneVoice after the CPU-heavy prep — cache tiers +
            /// bind. Split out so <see cref="CloneVoiceYielding"/> can pay ClipToMono / PrepRef /
            /// KeyFor one frame each (together they were the measured ~219 ms `bind` walk-up
            /// frame) and land here for the (cache-hit-cheap) tail.</summary>
            bool CloneVoicePrepared(float[] wav, string key, CropInfo crop, string label)
            {
                string path = System.IO.Path.Combine(CacheDir, key + ".bin");

                // Tier order (every tier VALIDATED — see PromptIsValid):
                //   1. Editor-precomputed Resources/Cache/<key>.bytes (inspector "Precompute
                //      voice-clone cache" button / PocketTTSVoiceBaker) — bake-time-verified, ships
                //      inside builds, so a baked voice NEVER re-encodes on any machine. Checked
                //      FIRST so a corrupt runtime .bin can never shadow a healthy shipped bake.
                //   2. Persistent runtime cache <persistentDataPath>/pockettts_voices/<key>.bin —
                //      machine-written; a poisoned entry (all-zero prompt from a GPU reset
                //      mid-encode) is DELETED here so the voice self-heals.
                //   3. Fresh encode — validated BEFORE it is bound or cached, so a failed GPU
                //      encode falls back to the baked voiceName instead of caching gibberish.
                float[] prompt = null;
                var baked = Resources.Load<TextAsset>(RES_VOICE_DIR + "/" + key);
                if (baked != null)
                {
                    prompt = PromptFromBytes(baked.bytes);
                    Resources.UnloadAsset(baked);
                    if (PromptIsValid(prompt)) LastCloneSource = "resources";
                    else
                    {
                        ConsoleMessage.Warning($"pocket-tts CloneVoice: baked Resources/{RES_VOICE_DIR}/{key}.bytes is " +
                                               "silent/invalid — re-bake it (Precompute voice-clone cache). Ignoring it.");
                        prompt = null;
                    }
                }
                if (prompt == null && System.IO.File.Exists(path))
                {
                    prompt = ReadPromptBin(path);   // runtime-written cache hit
                    if (PromptIsValid(prompt)) LastCloneSource = "persistent";
                    else
                    {
                        ConsoleMessage.Warning($"pocket-tts CloneVoice: persistent cache {path} held a silent/invalid " +
                                               "prompt (a GPU device reset poisoned an earlier encode) — deleted; re-encoding.");
                        try { System.IO.File.Delete(path); } catch { }
                        prompt = null;
                    }
                }
                if (prompt == null)
                {
                    // crop info logs ONLY here — an actual encode. Cache hits (persistent or
                    // editor-baked Resources) stay silent: the crop already happened at bake
                    // time and the inspector's precompute box reports the real cropped length.
                    if (crop.cropped)
                        Debug.Log(crop.atPause
                            ? $"[PocketTTS] voice-clone reference '{label ?? key}' is {crop.totalSeconds:F1}s — cropped at a " +
                              $"natural pause to {crop.croppedSeconds:F2}s (native prompt cap {MAX_REF_SECONDS:F1}s)."
                            : $"[PocketTTS] voice-clone reference '{label ?? key}' is {crop.totalSeconds:F1}s — no pause found " +
                              $"near the cap, using the first {MAX_REF_SECONDS:F1}s.");
                    prompt = EncodeToPrompt(wav);   // encode once
                    if (prompt == null) return false;
                    if (!PromptIsValid(prompt))
                    {
                        ConsoleMessage.Warning($"pocket-tts CloneVoice: the Mimi encoder returned a silent/invalid prompt " +
                                               $"for '{label ?? key}' (GPU device reset mid-encode?). NOT cached — keeping " +
                                               "the current voice. Precompute the clone in-editor to avoid runtime encodes.");
                        return false;
                    }
                    WritePromptBin(path, prompt);   // cache for the next runtime
                    LastCloneSource = "encoded";
                }
                voicePrompt = prompt;
                flm?.InvalidatePromptKV();   // see BindRawVoicePrompt: invalidate at the assignment
                CurrentVoice = string.IsNullOrEmpty(label) ? key : label;   // readable name; key stays content-addressed
                return true;
            }

            /// <summary>wav[24k] -> unquantized Mimi latents [T*32] (encode_to_latent output; the
            /// input to speaker_proj). Lazy-loads the encoder weights. Public for the P8 parity gate.</summary>
            public float[] EncodeToLatents(float[] wav24k, out int T)
            {
                if (encoder == null)
                {
                    weights.LoadBlocking("mimi/encoder");     // lazy-load the encoder-only tensors
                    weights.LoadBlocking("mimi/downsample");
                    encoder = new PocketTTSMimiEncoder(weights);
                }
                return encoder.Encode(wav24k, out T);
            }

            /// <summary>wav[24k] -> audio_prompt [T*1024]. Mimi encoder -> latents [T,32] ->
            /// speaker_proj [1024,32] (CPU matmul). Exactly _encode_audio (parity-gated in P8).</summary>
            public float[] EncodeToPrompt(float[] wav24k)
            {
                if (!HasEncoder)
                {
                    ConsoleMessage.Warning($"pocket-tts CloneVoice: encoder weights missing in {weightsDir}. " +
                                           "Re-export with `import_params.py pocket-tts --include-encoder`.");
                    return null;
                }
                _speakerProj ??= weights.ReadFloats("flow_lm.speaker_proj_weight");   // [1024,32]
                float[] latents = EncodeToLatents(wav24k, out int T);                // [T,32]
                // audio_prompt[t,o] = sum_i latents[t,i] * speaker_proj[o,i]
                var prompt = new float[T * Cfg.DIM];
                for (int t = 0; t < T; t++)
                {
                    int lb = t * Cfg.LDIM, pb = t * Cfg.DIM;
                    for (int o = 0; o < Cfg.DIM; o++)
                    {
                        float acc = 0f; int wb = o * Cfg.LDIM;
                        for (int i = 0; i < Cfg.LDIM; i++) acc += latents[lb + i] * _speakerProj[wb + i];
                        prompt[pb + o] = acc;
                    }
                }
                return prompt;
            }

            // ---- tiny linear resampler (reference clips only; quality here is not perceptually critical
            // — the encoder distills to a 125-frame speaker embedding, not the played audio) ----
            static float[] Resample(float[] src, int srIn, int srOut)
            {
                if (srIn == srOut) return src;
                int n = (int)((long)src.Length * srOut / srIn);
                var outv = new float[n];
                double step = (double)(src.Length - 1) / Math.Max(n - 1, 1);
                for (int i = 0; i < n; i++)
                {
                    double x = i * step; int i0 = (int)x; double f = x - i0;
                    int i1 = Math.Min(i0 + 1, src.Length - 1);
                    outv[i] = (float)(src[i0] * (1 - f) + src[i1] * f);
                }
                return outv;
            }

            static byte[] PromptToBytes(float[] prompt)
            {
                var bytes = new byte[prompt.Length * 4];
                Buffer.BlockCopy(prompt, 0, bytes, 0, bytes.Length);
                return bytes;
            }
            static float[] PromptFromBytes(byte[] bytes)
            {
                var f = new float[bytes.Length / 4];
                Buffer.BlockCopy(bytes, 0, f, 0, bytes.Length);
                return f;
            }
            void WritePromptBin(string path, float[] prompt)
            {
                System.IO.Directory.CreateDirectory(CacheDir);
                System.IO.File.WriteAllBytes(path, PromptToBytes(prompt));
            }
            static float[] ReadPromptBin(string path) => PromptFromBytes(System.IO.File.ReadAllBytes(path));

            float[] Gauss(int n, float std)
            {
                var a = new float[n];
                for (int i = 0; i < n; i += 2)
                {
                    double u1 = 1.0 - rng.NextDouble(), u2 = rng.NextDouble();
                    double r = Math.Sqrt(-2.0 * Math.Log(u1)) * std;
                    a[i] = (float)(r * Math.Cos(2 * Math.PI * u2));
                    if (i + 1 < n) a[i + 1] = (float)(r * Math.Sin(2 * Math.PI * u2));
                }
                return a;
            }

            /// <summary>Generate 24kHz mono from pre-tokenized SentencePiece ids. injectNoise!=null =>
            /// deterministic (P4 parity: run exactly injectNoise.Length frames, no EOS break).
            /// useKvCache=true (P5) uses the incremental KV-cache transformer decode (O(T·L), the
            /// default for real RTF); false = the full-forward-each-step path (O(T·L²), P2-shaped).
            /// The two are bit-identical — KV decode is exactly the full causal forward, amortized.</summary>
            public float[] GenerateOffline(int[] textIds, float[][] injectNoise = null,
                                           int maxFrames = DefaultMaxFrames, int framesAfterEos = 2,
                                           bool useKvCache = true)
            {
                var swAll = System.Diagnostics.Stopwatch.StartNew();
                int dim = Cfg.DIM;
                float[] textEmb = flm.EmbedLookup(textIds);   // [S*1024]
                int voiceFrames = voicePrompt.Length / dim;

                // prompt prefix = [bbv ; voice ; text]
                int Lp = 1 + voiceFrames + textIds.Length;
                var prefix = new float[Lp * dim];
                Array.Copy(bbv, 0, prefix, 0, dim);
                Array.Copy(voicePrompt, 0, prefix, dim, voiceFrames * dim);
                Array.Copy(textEmb, 0, prefix, (1 + voiceFrames) * dim, textIds.Length * dim);

                bool deterministic = injectNoise != null;
                int frames = deterministic ? injectNoise.Length : maxFrames;
                var latents = new List<float[]>(frames);
                int eosStep = -1;

                // full-forward path keeps the whole growing sequence; KV path only holds the caches.
                List<float> seq = null;
                PrefillMs = 0f; TtfaMs = 0f;
                if (useKvCache)
                {
                    flm.ResetKV();
                    var swPre = System.Diagnostics.Stopwatch.StartNew();
                    flm.PrefillKV(prefix, Lp, Lp + frames);   // caches now hold the prompt
                    GpuSync();                                // force completion so the timing is real
                    PrefillMs = (float)swPre.Elapsed.TotalMilliseconds;
                }
                else
                {
                    seq = new List<float>((Lp + frames) * dim);
                    seq.AddRange(prefix);
                }

                if (PerfCounting)   // #31-R2 item 1: attribute the AR loop separately from prefill
                {
                    StatLoopStartDisp = StatDispatches;
                    StatLoopStartReads = StatBlockingReads;
                    StatLoopStartUps = StatUploads;
                }
                var swLoop = System.Diagnostics.Stopwatch.StartNew();
                bool overlapActive = false;   // #31-R3: mimi windows interleaved with the AR blocks
                int mimiIssued = 0;           // latents already routed into an issued mimi window
                if (useKvCache && flm.CanRunGpuFrames() && !ForceLegacyArLoop)
                {
                    // ===== #31-R2: GPU-resident K-frame batches — ZERO per-frame syncs =====
                    // K frames issue back-to-back (feedback stays on-GPU), then ONE blocking
                    // readback of the K [eos|latent] slots. The CPU EOS scan reproduces the legacy
                    // semantics exactly: eos checked BEFORE a frame's latent is emitted, stop at
                    // eosStep+framesAfterEos, overshoot latents (computed but past the stop) are
                    // DISCARDED here, before mimi — emitted audio is identical by construction
                    // (frame f's latent never depends on later frames; extra KV rows are never
                    // attended by emitted frames). Non-deterministic runs consume RNG for the whole
                    // issued block (overshoot noise) — inconsequential (sampling), and injectNoise
                    // parity runs are indexed per absolute frame, unaffected.
                    // #31-R3 lever 2: the FIRST blocks follow ArBatchRamp (e.g. 2, 4) so the first
                    // eos/latent readback lands after ~2 frames of GPU, not K.
                    // #31-R3 lever 1: every completed MIMI_OVERLAP_CHUNK of scanned latents issues
                    // its mimi window immediately (no readback) — the window executes on the GPU
                    // interleaved with the NEXT AR blocks and is drained by their eos readbacks.
                    int K = Math.Max(1, Math.Min(ArBatchFrames, 16));
                    int stride = Cfg.LDIM + 1;
                    var noiseRows = new float[K][];
                    var slotCpu = new float[K * stride];
                    bool done = false;
                    overlapActive = OverlapMimi;
                    if (overlapActive) GrowWav(frames * Cfg.SAMPLES_PER_LATENT);
                    int blockIdx = 0;
                    for (int n0 = 0; n0 < frames && !done; blockIdx++)
                    {
                        int kThis = (ArBatchRamp != null && blockIdx < ArBatchRamp.Length)
                                    ? Math.Max(1, Math.Min(ArBatchRamp[blockIdx], K)) : K;
                        int blk = Math.Min(kThis, frames - n0);
                        for (int f = 0; f < blk; f++)
                            noiseRows[f] = deterministic ? injectNoise[n0 + f]
                                                         : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                        flm.UploadNoiseBlock(noiseRows, blk);              // one upload per block
                        for (int f = 0; f < blk; f++)
                            flm.DecodeFrameGpuIssue(f, n0 + f);            // no readbacks inside
                        flm.ReadEosLatBlock(blk, slotCpu);                 // the block's ONE sync point
                        for (int f = 0; f < blk; f++)
                        {
                            int n = n0 + f;
                            float eos = slotCpu[f * stride];
                            if (eos > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                            if (eosStep >= 0 && n >= eosStep + framesAfterEos) { done = true; break; }
                            var lat = new float[Cfg.LDIM];
                            Array.Copy(slotCpu, f * stride + 1, lat, 0, Cfg.LDIM);
                            latents.Add(lat);
                        }
                        // TTFA proxy is BLOCK-granular here (ArBatchRamp keeps block 0 tiny); the
                        // per-frame streaming path owns the real user-facing TTFA.
                        if (TtfaMs == 0f && latents.Count > 0)
                            TtfaMs = PrefillMs + (float)swLoop.Elapsed.TotalMilliseconds;
                        n0 += blk;
                        // R3 lever 1: issue every COMPLETE window over the scanned latents now —
                        // its GPU work hides behind the next blocks' eos-readback waits.
                        while (overlapActive && latents.Count - mimiIssued >= MIMI_OVERLAP_CHUNK)
                        {
                            IssueMimiWindow(latents, mimiIssued, mimiIssued + MIMI_OVERLAP_CHUNK);
                            mimiIssued += MIMI_OVERLAP_CHUNK;
                        }
                    }
                }
                else
                {
                    var swSec = PerfCounting ? new System.Diagnostics.Stopwatch() : null;
                    for (int n = 0; n < frames; n++)
                    {
                        swSec?.Restart();
                        float[] token = (n == 0) ? flm.BosLatentEmbedding() : flm.InputLinear(latents[n - 1]);
                        if (swSec != null) StatTokenCpuMs += swSec.Elapsed.TotalMilliseconds;
                        float[] c;
                        swSec?.Restart();
                        if (useKvCache)
                        {
                            c = flm.DecodeStepKV(token);          // append token, attend over cache -> out_norm(c)
                        }
                        else
                        {
                            seq.AddRange(token);
                            int L = seq.Count / dim;
                            var tfOut = flm.RunTransformer(seq.ToArray(), L);
                            c = flm.OutNormLastRow(tfOut, L);
                        }
                        if (swSec != null) StatDecodeCallMs += swSec.Elapsed.TotalMilliseconds;

                        // EOS applies in ALL modes (bit-exact c -> same eos step as the reference). The
                        // reference calls flow at the loop top then breaks BEFORE queue.put, so the
                        // post-EOS frames' noise exists in flow_noise_all but their latents are never
                        // emitted — breaking here (before FlowHead/collect) matches its put-count exactly.
                        float eos = flm.OutEos(c);
                        if (eos > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                        if (eosStep >= 0 && n >= eosStep + framesAfterEos) break;

                        float[] noise = deterministic ? injectNoise[n] : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                        swSec?.Restart();
                        float[] vel = flm.FlowHead(c, noise, 0f, 1f);
                        if (swSec != null) StatFlowCallMs += swSec.Elapsed.TotalMilliseconds;
                        float[] lat = new float[Cfg.LDIM];
                        for (int i = 0; i < Cfg.LDIM; i++) lat[i] = noise[i] + vel[i];   // 1 Euler step
                        latents.Add(lat);
                        // TTFA (offline proxy): prefill + first latent ready. True end-to-end TTFA (incl.
                        // the first Mimi frame + audio buffer fill) is measured by the streaming path (next step).
                        if (n == 0) { GpuSync(); TtfaMs = PrefillMs + (float)swLoop.Elapsed.TotalMilliseconds; }
                    }
                }
                // #31-R3: with overlap, the last eos readback already fenced the AR chain; only
                // freshly-issued mimi windows are still queued and belong to DecodeMs (the single
                // final readback drains them). NOTE the split's meaning under overlap: earlier
                // windows execute during the loop's readback waits, so LoopMs absorbs hidden mimi
                // work and DecodeMs shrinks — TOTAL is the comparable number in [perf] A/Bs.
                if (!overlapActive) GpuSync();
                LoopMs = (float)swLoop.Elapsed.TotalMilliseconds;

                int T = latents.Count;
                LastFrames = T;
                float[] raw = new float[T * Cfg.LDIM];
                for (int t = 0; t < T; t++) Array.Copy(latents[t], 0, raw, t * Cfg.LDIM, Cfg.LDIM);
                LastLatentsRaw = raw;
                GenMs = (float)swAll.Elapsed.TotalMilliseconds;

                // denorm (boundary op) happens inside Mimi.Decode when embMean/embStd passed.
                // Long generations route through the windowed chunked decode: same audio (windowed
                // is exact past the receptive field, see MIMI_DECODE_CTX) with BOUNDED dispatch
                // sizes and scratch memory (a single 512-frame block decode would want ~1 GB scratch).
                var swDec = System.Diagnostics.Stopwatch.StartNew();
                float[] wav;
                if (overlapActive)
                {
                    // finish the schedule (final ragged window) + ONE readback of the assembly.
                    // Windows and their dispatch parameters are IDENTICAL to DecodeWindowed(chunk
                    // 64) on the same latents -> bit-identical wav (probe R3 gate); for T <= 64
                    // that single window IS the plain full decode (bit-identical to the old path).
                    while (mimiIssued < T)
                    {
                        int e = Math.Min(mimiIssued + MIMI_OVERLAP_CHUNK, T);
                        IssueMimiWindow(latents, mimiIssued, e);
                        mimiIssued = e;
                    }
                    wav = new float[T * Cfg.SAMPLES_PER_LATENT];
                    if (wav.Length > 0) wavAccum.GetData(wav, 0, 0, wav.Length);
                }
                else
                {
                    wav = T <= 128 ? mimi.Decode(raw, T, embMean, embStd)
                                   : DecodeWindowed(raw, T);
                }
                DecodeMs = (float)swDec.Elapsed.TotalMilliseconds;
                return wav;
            }

            /// <summary>Windowed chunked Mimi decode of a long latent block [T*32]: each chunk is
            /// decoded fresh with MIMI_DECODE_CTX latents of left context and only its new samples
            /// kept. Exact vs a full decode once CTX >= the decoder's receptive field (~34.5 latents;
            /// verified corr 1.00000000, maxabs ~3e-5 fp noise), and bounds dispatch + scratch to the
            /// window size regardless of T. T <= CTX+chunk short-circuits to the plain full decode.</summary>
            public float[] DecodeWindowed(float[] rawLatents, int T, float[] mean = null, float[] std = null, int chunk = 64)
            {
                mean ??= embMean; std ??= embStd;
                int ctx = Cfg.MIMI_DECODE_CTX;
                if (T <= ctx + chunk) return mimi.Decode(rawLatents, T, mean, std);
                var wav = new float[T * Cfg.SAMPLES_PER_LATENT];
                for (int t0 = 0; t0 < T; t0 += chunk)
                {
                    int e = Math.Min(t0 + chunk, T);
                    int s = Math.Max(0, t0 - ctx);
                    int n = e - s;
                    var win = new float[n * Cfg.LDIM];
                    Array.Copy(rawLatents, s * Cfg.LDIM, win, 0, n * Cfg.LDIM);
                    // #30: only the kept tail is computed (context latents feed the transformer /
                    // receptive-field margins only) — samples before the tail are garbage.
                    float[] wv = mimi.Decode(win, n, mean, std, tailLatents: e - t0);
                    int newN = (e - t0) * Cfg.SAMPLES_PER_LATENT;
                    Array.Copy(wv, wv.Length - newN, wav, t0 * Cfg.SAMPLES_PER_LATENT, newN);
                }
                return wav;
            }

            // ---------------- P5.2: real-time streaming synthesis (per-frame AR + windowed decode) ----
            // Mirrors CosyVoiceTTS.SynthesizeStreaming(text, onSamples). Pocket is autoregressive
            // like CosyVoice's LM, so it streams per FRAME: KV-prefill the prompt, then each frame
            // DecodeStepKV -> FlowHead -> one latent. Every StreamChunkFrames, decode a WINDOW of
            // [MIMI_DECODE_CTX context + new frames] latents and push ONLY the new tail samples.
            //
            // No Mimi streaming state is needed: the decoder is fully causal with a BOUNDED left
            // receptive field (~34.5 latents, see MIMI_DECODE_CTX) and relative (RoPE) attention, so
            // a fresh window decode reproduces the new samples exactly (verified corr 1.00000000,
            // maxabs ~3e-5 fp noise vs full decode on a 210-frame utterance). While T <= CTX+chunk
            // the window starts at 0 = literally the full-prefix decode = BIT-exact (P5 gate).
            // The window bounds BOTH per-chunk cost (true O(1) streaming — the old full-prefix
            // re-decode was O(T^2) and blew the 65535-group dispatch limit past ~136 latents) and
            // dispatch/scratch sizes for arbitrarily long replies.
            //
            // onSamples receives each new sample block on the coroutine (main) thread; the caller
            // (PocketTTSVoice) pushes them into its ring buffer. Yields between frames/chunks so the
            // GPU work spreads across ticks and never blocks a single frame.
            public IEnumerator SynthesizeStreaming(int[] textIds, Action<float[]> onSamples,
                                                    int maxFrames = DefaultMaxFrames, int framesAfterEos = 2,
                                                    float[][] injectNoise = null)
            {
                if (!IsReady) { onSamples?.Invoke(null); yield break; }
                if (embMean == null) LoadCpuTensors();   // lazy after async BeginLoad

                bool deterministic = injectNoise != null;   // probe parity: inject reference noise
                if (deterministic) maxFrames = injectNoise.Length;

                LastHeavyTick = "clause_start";
                int dim = Cfg.DIM;
                float[] textEmb = flm.EmbedLookup(textIds);
                int voiceFrames = voicePrompt.Length / dim;
                int Lv = 1 + voiceFrames;                      // [bbv | voicePrompt] — speaker conditioning
                int Lp = Lv + textIds.Length;

                var swAll = System.Diagnostics.Stopwatch.StartNew();
                // #32: the prompt rows are identical in CONTENT and in POSITION on every clause, so
                // their K/V is retained across clauses and only the text rows are prefilled. This
                // attacks the measured 392-604 ms synth->first-audio dead window in the [TTFA] line,
                // during which playback drains and the ring starves.
                // Measured on the GTX 1650 box (125-frame prompt => Lv 126, prompt-cache probe
                // 2026-07-28, fp16): prefill rows 140 -> 14 (14-token clause) and 151 -> 25
                // (25-token clause), i.e. the same 126 prompt rows skipped either way => 11.4 -> 1.9
                // GMAC, whole-clause synth 541 -> 467 ms and 1016 -> 953 ms (63-74 ms of compute), and
                // the FrameBreak ticks the pump pays for the prefill 24 -> 0-1 (at the clause-start
                // allowance of 2-6 heavy ticks/frame that is ~4 frames of pacing gone on top).
                // Not the WHOLE window — the AR frames up to the first flush and that flush's Mimi
                // decode are the rest — but it is the part that was pure waste.
                //
                // Only the speaker conditioning is retained — never the previous clause's text or
                // latents. Continuing [voice][text1][latents1][text2] would be out of distribution
                // for an utterance-at-a-time TTS, the inter-clause pause is injected as literal zeros
                // into the ring (so the model's prosody timeline would diverge from the audio one),
                // and every clause ends with EOS + tail frames — i.e. the model already decided to
                // stop. None of that applies to the prompt rows.
                bool reusePrompt = flm.CanReusePromptKV(voicePrompt, Lv, Lp + maxFrames);
                LastPrefillRows = reusePrompt ? textIds.Length : Lp;
                // Every multi-frame loop below re-checks IsReady: on play-mode exit / assembly
                // reload the shared engine is disposed (weight buffers nulled) while this
                // coroutine may still get one more MoveNext — resuming into a dispatch then threw
                // ArgumentNullException from SetBuffer (LinearRows:178, seen 2026-07-22). Abort
                // quietly instead; PocketTTSWeights.Dispose drops IsReady.
                int prefillTicks = 0;
                if (reusePrompt)
                {
                    // Text rows go through the PER-ROW decode path (same kernels as the block
                    // prefill for a single token -> bit-exact; see AppendRowsKVYielding).
                    flm.BeginFromRetainedPromptKV();
                    flm.NotePromptKVReuse();   // drives the bounded self-heal — see CanReusePromptKV
                    var tp = flm.AppendRowsKVYielding(textEmb, textIds.Length);
                    while (IsReady && tp.MoveNext()) { prefillTicks++; LastHeavyTick = "prefill_text"; yield return FrameBreak; }
                    if (!IsReady) { onSamples?.Invoke(null); yield break; }
                }
                else
                {
                    var prefix = new float[Lp * dim];
                    Array.Copy(bbv, 0, prefix, 0, dim);
                    Array.Copy(voicePrompt, 0, prefix, dim, voiceFrames * dim);
                    Array.Copy(textEmb, 0, prefix, Lv * dim, textIds.Length * dim);
                    flm.ResetKV();
                    // bug C + #29: prefill yields per layer AND each tick ends the frame (FrameBreak) —
                    // the pump's CPU-time budget would otherwise re-enter all 6 ticks in one frame.
                    var pf = flm.PrefillKVYielding(prefix, Lp, Lp + maxFrames);
                    while (IsReady && pf.MoveNext())
                    {
                        prefillTicks++;
                        // #36.3: keep the fine pf:L<i>.<sec> tag the prefill just set — the
                        // blanket overwrite was hiding WHICH section owned a slow tick.
                        if (LastHeavyTick == null || !LastHeavyTick.StartsWith("pf:")) LastHeavyTick = "prefill";
                        yield return FrameBreak;
                    }
                    if (!IsReady) { onSamples?.Invoke(null); yield break; }
                    // The prefix STARTS with the prompt, so rows [0,Lv) of the caches are already the
                    // prompt's K/V — retaining them is bookkeeping only, no extra compute. Keyed on
                    // the voicePrompt array's identity: SetVoice/CloneVoice both assign a fresh array,
                    // so the next clause on another voice fails the key and re-prefills.
                    flm.RetainPromptKV(voicePrompt, Lv);
                }
                LastPrefillTicks = prefillTicks;
                LastPrefillMs = (float)swAll.Elapsed.TotalMilliseconds;

                var latents = new List<float[]>(maxFrames);
                var rawAll = new List<float>(maxFrames * Cfg.LDIM);   // raw flow latents, growing
                int eosStep = -1;
                int emittedFrames = 0;                                // latents already turned into audio
                int chunk = Mathf.Max(1, StreamChunkFrames);
                TtfaMs = 0f;
                float[] c = new float[Cfg.DIM];                       // per-frame readback targets (reused)
                float[] vel = new float[Cfg.LDIM];

                // #31-R2: GPU-resident frame — one combined [eos|latent] readback replaces the
                // legacy pair (c then velocity) and kills the token/cond uploads + CPU input_linear.
                // 2026-07-30: readbacks are now one per StreamArBatchFrames BLOCK, not per frame
                // (see that field's docs). The flush block is UNCHANGED; the only extra work is
                // up to K-1 discarded overshoot frames past the stop frame.
                bool gpuAr = flm.CanRunGpuFrames();
                // #StreamArBatch state: frames arrive in K-sized blocks (one readback per block),
                // the outer loop scans them one per iteration so EOS/flush semantics are untouched.
                int steadyK = gpuAr ? Mathf.Clamp(StreamArBatchFrames, 1, 8) : 1;
                int slotStride = Cfg.LDIM + 1;
                float[] slotBlk = gpuAr ? new float[steadyK * slotStride] : null;
                float[][] noiseBlk = gpuAr ? new float[steadyK][] : null;
                int blkCount = 0, blkNext = 0, blkIdx = 0;   // buffered frames, scan cursor, block #
                // #31-R3: first flush after StreamFirstChunkFrames (GPU-frame path), then the
                // normal cadence — earlier first audio, identical samples (windowed tail-exact).
                int firstChunk = gpuAr ? Mathf.Clamp(StreamFirstChunkFrames, 1, chunk) : chunk;

                for (int n = 0; n < maxFrames; n++)
                {
                    if (!IsReady) { onSamples?.Invoke(null); yield break; }   // disposed mid-utterance
                    bool stop;
                    if (gpuAr)
                    {
                        if (blkNext >= blkCount)
                        {
                            // Issue the next block: one noise upload, kThis chained GPU-resident
                            // frames (feedback on-GPU, no readbacks inside), then ONE combined
                            // [eos|latent] readback for the whole block. The per-frame readback
                            // this replaces was the production cap behind the 2026-07-30 blips.
                            int kThis = (StreamArBatchRamp != null && blkIdx < StreamArBatchRamp.Length)
                                        ? Mathf.Clamp(StreamArBatchRamp[blkIdx], 1, steadyK) : steadyK;
                            int blk = Math.Min(kThis, maxFrames - n);
                            for (int f = 0; f < blk; f++)
                                noiseBlk[f] = deterministic ? injectNoise[n + f]
                                                            : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                            flm.UploadNoiseBlock(noiseBlk, blk);
                            for (int f = 0; f < blk; f++)
                            {
                                // one FrameBreak per issued frame, BETWEEN issues (blk == 1 goes
                                // straight to its readback — exactly the old per-frame pacing):
                                // the tier's tick cap still bounds per-frame GPU load.
                                if (f > 0) { LastHeavyTick = "ar_frame"; yield return FrameBreak; }
                                if (!IsReady) { onSamples?.Invoke(null); yield break; }
                                flm.DecodeFrameGpuIssue(f, n + f);
                            }
                            var rb = flm.ReadEosLatYielding(blk, slotBlk, AsyncReadback);
                            while (IsReady && rb.MoveNext()) { LastHeavyTick = "ar_frame"; yield return rb.Current; }
                            if (!IsReady) { onSamples?.Invoke(null); yield break; }
                            blkCount = blk; blkNext = 0; blkIdx++;
                        }
                        float eosG = slotBlk[blkNext * slotStride];
                        if (eosG > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                        stop = eosStep >= 0 && n >= eosStep + framesAfterEos;
                        if (!stop)
                        {
                            float[] lat = new float[Cfg.LDIM];
                            Array.Copy(slotBlk, blkNext * slotStride + 1, lat, 0, Cfg.LDIM);
                            latents.Add(lat);
                            rawAll.AddRange(lat);
                        }
                        blkNext++;
                    }
                    else
                    {
                    float[] token = (n == 0) ? flm.BosLatentEmbedding() : flm.InputLinear(latents[n - 1]);
                    // bug C: async readbacks — yield while the GPU drains instead of a blocking GetData
                    var ds = flm.DecodeStepKVYielding(token, c, AsyncReadback);
                    while (IsReady && ds.MoveNext()) { LastHeavyTick = "ar_decode"; yield return ds.Current; }
                    if (!IsReady) { onSamples?.Invoke(null); yield break; }
                    float eos = flm.OutEos(c);
                    if (eos > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                    stop = eosStep >= 0 && n >= eosStep + framesAfterEos;
                    if (!stop)
                    {
                        float[] noise = deterministic ? injectNoise[n] : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                        var fh = flm.FlowHeadYielding(c, noise, 0f, 1f, vel, AsyncReadback);
                        while (IsReady && fh.MoveNext()) { LastHeavyTick = "ar_flowhead"; yield return fh.Current; }
                        if (!IsReady) { onSamples?.Invoke(null); yield break; }
                        float[] lat = new float[Cfg.LDIM];
                        for (int i = 0; i < Cfg.LDIM; i++) lat[i] = noise[i] + vel[i];
                        latents.Add(lat);
                        rawAll.AddRange(lat);
                    }
                    }

                    // emit a chunk when enough new frames accumulated, or at end-of-stream.
                    // Schedule form (== the old `% chunk` cadence when firstChunk == chunk): next
                    // flush at emitted + (first ? firstChunk : chunk) scanned frames.
                    // StreamHurry (gpuAr path, like firstChunk): while audible silence is imminent
                    // the cadence is suspended and any StreamHurryMinFrames pending decode NOW —
                    // only delivery timing moves, samples are boundary-invariant.
                    int pending = latents.Count - emittedFrames;
                    bool flush = stop || (pending > 0 &&
                        (latents.Count >= emittedFrames + (emittedFrames == 0 ? firstChunk : chunk) ||
                         (gpuAr && pending >= StreamHurryMinFrames && StreamHurry != null && StreamHurry())));
                    if (flush && latents.Count > emittedFrames)
                    {
                        // windowed decode: [ctx context ; new frames] — new samples are exact
                        // (receptive field < ctx) and per-chunk cost/dispatch stay O(window).
                        // #29: the decode chain is SLICED (each slice = FrameBreak = one frame) and
                        // the wav readback is ASYNC — the old single-tick sync flush was a 100-155 ms
                        // main-thread stall every 0.64 s. The ring prebuffer absorbs the 1-3 frame
                        // delivery latency.
                        int T = latents.Count;
                        int newFrames = T - emittedFrames;
                        int s = Math.Max(0, T - newFrames - Cfg.MIMI_DECODE_CTX);
                        int nWin = T - s;
                        var win = new float[nWin * Cfg.LDIM];
                        rawAll.CopyTo(s * Cfg.LDIM, win, 0, nWin * Cfg.LDIM);
                        float[] wv = new float[nWin * Cfg.SAMPLES_PER_LATENT];
                        // #30: tail-restricted — only the newFrames tail of the window is decoded
                        // (bit-exact for the kept samples; the context region stays garbage).
                        var de = mimi.DecodeYielding(win, nWin, embMean, embStd, wv, AsyncReadback, newFrames);
                        // GPU-issue slices end the frame (FrameBreak); readback waits surface as
                        // GpuWait so the pump can catch a mid-frame completion cheaply.
                        while (IsReady && de.MoveNext())
                        {
                            LastHeavyTick = "mimi_decode";
                            yield return ReferenceEquals(de.Current, GpuWait) ? GpuWait : FrameBreak;
                        }
                        if (!IsReady) { onSamples?.Invoke(null); yield break; }
                        int newN = newFrames * Cfg.SAMPLES_PER_LATENT;
                        float[] tail = new float[newN];
                        Array.Copy(wv, wv.Length - newN, tail, 0, newN);
                        emittedFrames = T;
                        if (TtfaMs == 0f) TtfaMs = (float)swAll.Elapsed.TotalMilliseconds;
                        LastHeavyTick = "flush_push";
                        onSamples?.Invoke(tail);
                        yield return null;   // let the ring drain
                    }
                    if (stop) break;
                    yield return null;       // one frame of AR per tick — never blocks the main thread
                }
                StreamLastTokenCount = latents.Count;
                GenMs = (float)swAll.Elapsed.TotalMilliseconds;
                onSamples?.Invoke(null);     // sentinel: stream complete
            }

            /// <summary>Decode the first T frames of a raw (pre-denorm) flow-latent block [T*32] to
            /// wav. Causal: the result is the exact prefix of decoding any longer block (verified
            /// bit-exact). Used by the streaming path / stream probe to re-decode growing prefixes.</summary>
            public float[] DecodePrefix(float[] rawLatents, int T, float[] mean = null, float[] std = null)
                => mimi.Decode(rawLatents, T, mean ?? embMean, std ?? embStd);

            public AudioClip ToClip(float[] wav, string name = "pockettts")
            {
                var clip = AudioClip.Create(name, wav.Length, 1, Cfg.SAMPLE_RATE, false);
                clip.SetData(wav, 0);
                return clip;
            }

            public void Dispose() { syncBuf?.Release(); wavAccum?.Release(); encoder?.Dispose(); flm?.Dispose(); mimi?.Dispose(); weights?.Dispose(); }
        }
    }
}
