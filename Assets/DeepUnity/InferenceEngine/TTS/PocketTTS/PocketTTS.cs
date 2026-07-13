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

            public bool IsReady => weights.IsReady;
            public long WeightBytes => weights.BytesTotal;   // resident weight footprint (fp16 vs int8 delta)

            public PocketTTS(string weightsDir = null)
            {
                this.weightsDir = weightsDir ?? Cfg.WEIGHTS_DIR_FP16;
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
            /// (only baked voices in the export are available; bake more with import_pocket_tts.py --voice).</summary>
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
            // (import_pocket_tts.py --include-encoder); baked voices work without them.

            public bool HasEncoder => weights.Has("mimi/encoder/model/0/conv.weight")
                                   && weights.Has("mimi/downsample/conv/conv.weight");

            string CacheDir => System.IO.Path.Combine(Application.persistentDataPath, "pockettts_voices");

            /// <summary>Content hash of the reference samples (FNV-1a over the raw floats) — the cache key.</summary>
            static string HashSamples(float[] s)
            {
                ulong h = 1469598103934665603UL;
                var buf = new byte[s.Length * 4];
                Buffer.BlockCopy(s, 0, buf, 0, buf.Length);
                foreach (byte bb in buf) { h ^= bb; h *= 1099511627789UL; }
                return h.ToString("x16");
            }

            /// <summary>Resources folder (under any Assets/Resources/) holding editor-baked voice-clone
            /// prompts as .bytes TextAssets — the shipping tier of the clone cache.</summary>
            public const string RES_VOICE_DIR = "PocketTTSVoices";

            /// <summary>Where the last CloneVoice found its prompt: "persistent" (runtime disk cache) |
            /// "resources" (editor-baked, ships in builds) | "encoded" (computed now + cached).</summary>
            public string LastCloneSource { get; private set; }

            static string KeyFor(float[] wav24k, string label)
            {
                string tag = string.IsNullOrEmpty(label) ? "" : Sanitize(label) + "_";
                return tag + HashSamples(wav24k);
            }
            static string Sanitize(string s)
            {
                var sb = new System.Text.StringBuilder(s.Length);
                foreach (char c in s) sb.Append(char.IsLetterOrDigit(c) || c == '-' || c == '_' ? c : '_');
                return sb.ToString();
            }

            /// <summary>The cache key CloneVoice(clip) will use — lets editor tooling check for a baked
            /// Resources entry without touching the model. Null if the clip isn't readable.</summary>
            public static string CloneKey(AudioClip clip)
            {
                float[] mono = ClipToMono(clip);
                if (mono == null) return null;
                return KeyFor(PrepRef(mono, clip.frequency), clip.name);
            }

            /// <summary>Editor precompute: encode a reference wav to the raw audio_prompt bytes the
            /// cache stores (byte-identical to WritePromptBin's file / a Resources .bytes asset).
            /// Weights (incl. the Mimi encoder) must be resident. Null on failure.</summary>
            public byte[] PrecomputePromptBytes(float[] samples, int sampleRate, string label, out string key)
            {
                key = null;
                if (samples == null || samples.Length == 0) return null;
                float[] wav = PrepRef(samples, sampleRate);   // resample to 24k + cap at MAX_REF_SECONDS
                key = KeyFor(wav, label);
                float[] prompt = EncodeToPrompt(wav);
                return prompt == null ? null : PromptToBytes(prompt);
            }
            public byte[] PrecomputePromptBytes(AudioClip clip, out string key)
            {
                key = null;
                float[] mono = ClipToMono(clip);
                return mono == null ? null : PrecomputePromptBytes(mono, clip.frequency, clip.name, out key);
            }

            /// <summary>Clone a voice from an AudioClip (any sample rate; multi-channel down-mixed to
            /// mono; only the first MAX_REF_SECONDS are used). Caches + binds as the current voice.</summary>
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
            /// 24 kHz stage), so the first 10 s are used and the rest ignored. The cache key hashes the
            /// CAPPED wav, so bake and runtime always agree on the same key.</summary>
            public const float MAX_REF_SECONDS = 10f;

            static float[] PrepRef(float[] samples, int sampleRate)
            {
                float[] wav = sampleRate == Cfg.SAMPLE_RATE ? samples : Resample(samples, sampleRate, Cfg.SAMPLE_RATE);
                int cap = (int)(MAX_REF_SECONDS * Cfg.SAMPLE_RATE);
                if (wav.Length > cap)
                {
                    Debug.Log($"[PocketTTS] voice-clone reference is {wav.Length / (float)Cfg.SAMPLE_RATE:F1}s — " +
                              $"using the first {MAX_REF_SECONDS:F0}s (the model's native prompt length).");
                    var t = new float[cap];
                    Array.Copy(wav, t, cap);
                    wav = t;
                }
                return wav;
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

                float[] wav = PrepRef(samples, sampleRate);   // resample to 24k + cap at MAX_REF_SECONDS
                string key = KeyFor(wav, label);
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
                    // PocketTTSVoiceBaker): a raw-float TextAsset at Resources/PocketTTSVoices/<key>
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
                        prompt = EncodeToPrompt(wav);   // encode once
                        if (prompt == null) return false;
                        WritePromptBin(path, prompt);   // cache for the next runtime
                        LastCloneSource = "encoded";
                    }
                }
                voicePrompt = prompt;
                CurrentVoice = key;
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
                                           "Re-export with `import_pocket_tts.py --include-encoder`.");
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
                // bug C: prefill yields per layer (~15 ms/tick instead of one ~90 ms burst)
                var pf = flm.PrefillKVYielding(prefix, Lp, Lp + maxFrames);
                while (pf.MoveNext()) yield return null;

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
                    while (ds.MoveNext()) yield return null;
                    float eos = flm.OutEos(c);
                    if (eos > Cfg.EOS_THRESHOLD && eosStep < 0) eosStep = n;
                    bool stop = eosStep >= 0 && n >= eosStep + framesAfterEos;
                    if (!stop)
                    {
                        float[] noise = deterministic ? injectNoise[n] : Gauss(Cfg.LDIM, Mathf.Sqrt(Cfg.TEMPERATURE));
                        var fh = flm.FlowHeadYielding(c, noise, 0f, 1f, vel, AsyncReadback);
                        while (fh.MoveNext()) yield return null;
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
                        int T = latents.Count;
                        int newFrames = T - emittedFrames;
                        int s = Math.Max(0, T - newFrames - Cfg.MIMI_DECODE_CTX);
                        int nWin = T - s;
                        var win = new float[nWin * Cfg.LDIM];
                        rawAll.CopyTo(s * Cfg.LDIM, win, 0, nWin * Cfg.LDIM);
                        float[] wv = mimi.Decode(win, nWin, embMean, embStd);
                        int newN = newFrames * Cfg.SAMPLES_PER_LATENT;
                        float[] tail = new float[newN];
                        Array.Copy(wv, wv.Length - newN, tail, 0, newN);
                        emittedFrames = T;
                        if (TtfaMs == 0f) TtfaMs = (float)swAll.Elapsed.TotalMilliseconds;
                        onSamples?.Invoke(tail);
                        yield return null;   // spread decode + let the ring drain
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
