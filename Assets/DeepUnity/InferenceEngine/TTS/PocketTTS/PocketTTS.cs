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
            /// makes one combined async readback per frame instead of two. Plus the fused
            /// transformer step (LN folded into the GEMV staging, residual adds folded into GEMV
            /// epilogues, slice+RoPE+KV-append as one kernel): ~49 dispatches/frame vs R1's ~96.
            /// Default ON; false restores the exact R1 dispatch list (three-tier bisect:
            /// legacy / FastKernels2 / FastKernels2+3).</summary>
            public static bool FastKernels3 = true;

            /// <summary>#31-R2: frames per GPU-resident offline batch (1..16). Larger = fewer
            /// pipeline drains but more post-EOS overshoot compute (up to K-1 discarded frames,
            /// ~2% of a 130-frame clip at 8). Streaming always runs per-frame (latency).</summary>
            public static int ArBatchFrames = 8;

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
            public static string LastHeavyTick;

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

            /// <summary>Load-on-approach spread over ~targetSeconds (tiny per-frame upload slices).</summary>
            public void SlowPrefetch(float targetSeconds)
            {
                if (targetSeconds > 0.01f && weights.BytesTotal > 0)
                    weights.BudgetBytesPerFrame = Math.Max(1, (long)(weights.BytesTotal / (targetSeconds * 60f)));
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
            public void SetVoice(string name)
            {
                if (string.IsNullOrEmpty(name) || name == CurrentVoice) return;
                if (!weights.Has($"voices/{name}/audio_prompt"))
                {
                    ConsoleMessage.Warning($"pocket-tts: baked voice '{name}' not found in {weightsDir} " +
                                           $"(only exported voices are available) — keeping '{CurrentVoice}'.");
                    return;
                }
                voicePrompt = weights.ReadFloats($"voices/{name}/audio_prompt");
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
                    dummies[i] = new ComputeBuffer(256, 4, ComputeBufferType.Structured);
                foreach (string u in zeroUniforms) shader.SetInt(u, 0);

                foreach (string name in kernels)
                {
                    int k = shader.FindKernel(name);
                    for (int i = 0; i < bufs.Length; i++) shader.SetBuffer(k, bufs[i], dummies[i]);
                    shader.Dispatch(k, 1, 1, 1);   // one compile per frame when pumped as a coroutine
                    yield return null;
                }

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
                                           int maxFrames = 512, int framesAfterEos = 2,
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
                if (useKvCache && flm.CanRunGpuFrames())
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
                                                    int maxFrames = 512, int framesAfterEos = 2,
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
                    while (IsReady && pf.MoveNext()) { prefillTicks++; LastHeavyTick = "prefill"; yield return FrameBreak; }
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
                // Per-frame pacing and the flush block are UNCHANGED (pump semantics intact); the
                // only extra work is one discarded flow-head pass on the final stop frame.
                bool gpuAr = flm.CanRunGpuFrames();
                float[] slot1 = gpuAr ? new float[Cfg.LDIM + 1] : null;
                float[][] noiseRow1 = gpuAr ? new float[1][] : null;
                // #31-R3: first flush after StreamFirstChunkFrames (GPU-frame path), then the
                // normal cadence — earlier first audio, identical samples (windowed tail-exact).
                int firstChunk = gpuAr ? Mathf.Clamp(StreamFirstChunkFrames, 1, chunk) : chunk;

                for (int n = 0; n < maxFrames; n++)
                {
                    if (!IsReady) { onSamples?.Invoke(null); yield break; }   // disposed mid-utterance
                    bool stop;
                    if (gpuAr)
                    {
                        noiseRow1[0] = deterministic ? injectNoise[n] : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                        flm.UploadNoiseBlock(noiseRow1, 1);
                        flm.DecodeFrameGpuIssue(0, n);
                        var rb = flm.ReadEosLatYielding(1, slot1, AsyncReadback);
                        while (IsReady && rb.MoveNext()) { LastHeavyTick = "ar_frame"; yield return rb.Current; }
                        if (!IsReady) { onSamples?.Invoke(null); yield break; }
                        float eosG = slot1[0];
                        if (eosG > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                        stop = eosStep >= 0 && n >= eosStep + framesAfterEos;
                        if (!stop)
                        {
                            float[] lat = new float[Cfg.LDIM];
                            Array.Copy(slot1, 1, lat, 0, Cfg.LDIM);
                            latents.Add(lat);
                            rawAll.AddRange(lat);
                        }
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
