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
            /// slice). NOT hardware-tuned: this is only the starting guess — PocketTTSVoice
            /// self-calibrates it at runtime from real frame feedback (a heavy tick should cost
            /// ~3-7 ms over the scene's baseline frame, whatever the GPU). Slower cards converge
            /// to finer slices, faster cards to coarser ones.</summary>
            public static long GpuMacsPerTick = 900_000_000;

            /// <summary>#29 spike attribution: the last pipeline stage a pump tick worked on
            /// ("clause_start", "prefill", "ar_decode", "ar_flowhead", "mimi_decode", "flush_push",
            /// "readback_hardwait"). Written by SynthesizeStreaming's yield sites; diagnostics
            /// probes read-and-clear it per frame to attribute slow frames to a stage. Inert
            /// otherwise (one static string assign per tick).</summary>
            public static string LastHeavyTick;

            public bool IsReady => weights.IsReady;
            public long WeightBytes => weights.BytesTotal;   // resident weight footprint (fp16 vs int8 delta)

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
                => weights.Defetch(slow ? weights.BudgetBytesPerFrame : 0);

            // emb_mean/std + voice/bbv are tiny CPU reads (ReadFloats streams from disk directly,
            // independent of the GPU upload pump) — safe to load as soon as the manifest is parsed.
            void LoadCpuTensors()
            {
                embMean = weights.ReadFloats("flow_lm.emb_mean");
                embStd = weights.ReadFloats("flow_lm.emb_std");
                bbv = weights.ReadFloats("flow_lm.bos_before_voice");
                voicePrompt ??= weights.ReadFloats("voices/jean/audio_prompt");
            }

            /// <summary>Editor/probe: synchronous blocking load of everything.</summary>
            public void LoadBlocking()
            {
                weights.LoadBlocking();   // all tensors
                LoadCpuTensors();         // emb_mean/std [32], bbv [1024], voices/jean/audio_prompt [125*1024]
            }

            public string CurrentVoice { get; private set; } = "jean";

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

            /// <summary>Where the last CloneVoice found its prompt: "persistent" (runtime disk cache) |
            /// "resources" (editor-baked, ships in builds) | "encoded" (computed now + cached).</summary>
            public string LastCloneSource { get; private set; }

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

            /// <summary>Max reference length for cloning — 10 s = 125 latent frames, the model's NATIVE
            /// audio_prompt length (every Kyutai baked voice is exactly this). Longer references only
            /// slow each reply's prefill and overflow the encoder's 1D dispatch limit (~10.9 s at the
            /// 24 kHz stage), so a longer clip is cropped at a natural pause near this cap (PrepRef;
            /// hard 10 s cut only if the 7-10 s window has no pause). The cache key hashes the CROPPED
            /// wav, so bake and runtime always agree on the same key.</summary>
            public const float MAX_REF_SECONDS = 10f;

            // Pause-aware crop: a long reference is cut at a NATURAL PAUSE near the cap instead of
            // mid-word at exactly 10.0 s — a chopped word in the prompt conditions the voice on a
            // truncation artifact. Never cropped shorter than MIN_CROP_SECONDS; a "pause" is
            // >= 3 consecutive 30 ms hops whose RMS sits under 15% of the clip's mean hop-RMS
            // (stop-consonant closures are shorter, real pauses are longer).
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
                int cut = FindPauseCut(wav, cap);
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

                float[] prompt;
                if (System.IO.File.Exists(path))
                {
                    prompt = ReadPromptBin(path);   // runtime-written cache hit
                    LastCloneSource = "persistent";
                }
                else
                {
                    // Editor-precomputed cache (inspector "Precompute voice-clone cache" button /
                    // PocketTTSVoiceBaker): a raw-float TextAsset at Resources/Cache/<key> (shared content-addressed cache)
                    // — ships inside builds, so a baked voice NEVER re-encodes on any machine.
                    var baked = Resources.Load<TextAsset>(RES_VOICE_DIR + "/" + key);
                    if (baked != null)
                    {
                        prompt = PromptFromBytes(baked.bytes);
                        Resources.UnloadAsset(baked);
                        LastCloneSource = "resources";
                    }
                    else
                    {
                        // crop info logs ONLY here — an actual encode. Cache hits (persistent or
                        // editor-baked Resources) stay silent: the crop already happened at bake
                        // time and the inspector's precompute box reports the real cropped length.
                        if (crop.cropped)
                            Debug.Log(crop.atPause
                                ? $"[PocketTTS] voice-clone reference '{label ?? key}' is {crop.totalSeconds:F1}s — cropped at a " +
                                  $"natural pause to {crop.croppedSeconds:F2}s (native prompt cap {MAX_REF_SECONDS:F0}s)."
                                : $"[PocketTTS] voice-clone reference '{label ?? key}' is {crop.totalSeconds:F1}s — no pause found " +
                                  $"near the cap, using the first {MAX_REF_SECONDS:F0}s.");
                        prompt = EncodeToPrompt(wav);   // encode once
                        if (prompt == null) return false;
                        WritePromptBin(path, prompt);   // cache for the next runtime
                        LastCloneSource = "encoded";
                    }
                }
                voicePrompt = prompt;
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

                var swLoop = System.Diagnostics.Stopwatch.StartNew();
                for (int n = 0; n < frames; n++)
                {
                    float[] token = (n == 0) ? flm.BosLatentEmbedding() : flm.InputLinear(latents[n - 1]);
                    float[] c;
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

                    // EOS applies in ALL modes (bit-exact c -> same eos step as the reference). The
                    // reference calls flow at the loop top then breaks BEFORE queue.put, so the
                    // post-EOS frames' noise exists in flow_noise_all but their latents are never
                    // emitted — breaking here (before FlowHead/collect) matches its put-count exactly.
                    float eos = flm.OutEos(c);
                    if (eos > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                    if (eosStep >= 0 && n >= eosStep + framesAfterEos) break;

                    float[] noise = deterministic ? injectNoise[n] : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                    float[] vel = flm.FlowHead(c, noise, 0f, 1f);
                    float[] lat = new float[Cfg.LDIM];
                    for (int i = 0; i < Cfg.LDIM; i++) lat[i] = noise[i] + vel[i];   // 1 Euler step
                    latents.Add(lat);
                    // TTFA (offline proxy): prefill + first latent ready. True end-to-end TTFA (incl.
                    // the first Mimi frame + audio buffer fill) is measured by the streaming path (next step).
                    if (n == 0) { GpuSync(); TtfaMs = PrefillMs + (float)swLoop.Elapsed.TotalMilliseconds; }
                }
                GpuSync();
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
                float[] wav = T <= 128 ? mimi.Decode(raw, T, embMean, embStd)
                                       : DecodeWindowed(raw, T);
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
                    float[] wv = mimi.Decode(win, n, mean, std);
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
                int Lp = 1 + voiceFrames + textIds.Length;
                var prefix = new float[Lp * dim];
                Array.Copy(bbv, 0, prefix, 0, dim);
                Array.Copy(voicePrompt, 0, prefix, dim, voiceFrames * dim);
                Array.Copy(textEmb, 0, prefix, (1 + voiceFrames) * dim, textIds.Length * dim);

                var swAll = System.Diagnostics.Stopwatch.StartNew();
                flm.ResetKV();
                // bug C + #29: prefill yields per layer AND each tick ends the frame (FrameBreak) —
                // the pump's CPU-time budget would otherwise re-enter all 6 ticks in one frame.
                var pf = flm.PrefillKVYielding(prefix, Lp, Lp + maxFrames);
                while (pf.MoveNext()) { LastHeavyTick = "prefill"; yield return FrameBreak; }

                var latents = new List<float[]>(maxFrames);
                var rawAll = new List<float>(maxFrames * Cfg.LDIM);   // raw flow latents, growing
                int eosStep = -1;
                int emittedFrames = 0;                                // latents already turned into audio
                int chunk = Mathf.Max(1, StreamChunkFrames);
                TtfaMs = 0f;
                float[] c = new float[Cfg.DIM];                       // per-frame readback targets (reused)
                float[] vel = new float[Cfg.LDIM];

                for (int n = 0; n < maxFrames; n++)
                {
                    float[] token = (n == 0) ? flm.BosLatentEmbedding() : flm.InputLinear(latents[n - 1]);
                    // bug C: async readbacks — yield while the GPU drains instead of a blocking GetData
                    var ds = flm.DecodeStepKVYielding(token, c, AsyncReadback);
                    while (ds.MoveNext()) { LastHeavyTick = "ar_decode"; yield return ds.Current; }
                    float eos = flm.OutEos(c);
                    if (eos > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                    bool stop = eosStep >= 0 && n >= eosStep + framesAfterEos;
                    if (!stop)
                    {
                        float[] noise = deterministic ? injectNoise[n] : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                        var fh = flm.FlowHeadYielding(c, noise, 0f, 1f, vel, AsyncReadback);
                        while (fh.MoveNext()) { LastHeavyTick = "ar_flowhead"; yield return fh.Current; }
                        float[] lat = new float[Cfg.LDIM];
                        for (int i = 0; i < Cfg.LDIM; i++) lat[i] = noise[i] + vel[i];
                        latents.Add(lat);
                        rawAll.AddRange(lat);
                    }

                    // emit a chunk when enough new frames accumulated, or at end-of-stream
                    bool flush = stop || (latents.Count > 0 && latents.Count % chunk == 0);
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
                        var de = mimi.DecodeYielding(win, nWin, embMean, embStd, wv, AsyncReadback);
                        // GPU-issue slices end the frame (FrameBreak); readback waits surface as
                        // GpuWait so the pump can catch a mid-frame completion cheaply.
                        while (de.MoveNext())
                        {
                            LastHeavyTick = "mimi_decode";
                            yield return ReferenceEquals(de.Current, GpuWait) ? GpuWait : FrameBreak;
                        }
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

            public void Dispose() { syncBuf?.Release(); encoder?.Dispose(); flm?.Dispose(); mimi?.Dispose(); weights?.Dispose(); }
        }
    }
}
