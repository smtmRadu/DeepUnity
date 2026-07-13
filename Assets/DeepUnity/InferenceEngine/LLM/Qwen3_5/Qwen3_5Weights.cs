using System;
using System.Collections;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // FP16 weights packed two halves per uint, stored on the GPU as ComputeBuffer<uint>.
        // Per-layer arrays are sized to NUM_LAYERS; entries are null for the wrong layer type.
        //
        // Loading is streamed and allocation-flat:
        //   - The constructor only builds a file manifest (no GPU buffers, no reads) — the boot frame
        //     does near-zero work.
        //   - Background readers (at most MAX_IO_JOBS in flight) read each .bin into a pooled byte[]
        //     of exactly the file size; arrays are reused across files, so the managed heap stays flat
        //     instead of churning ~2x the model size and stalling the main thread on GC collections.
        //   - The main-thread UploadPump creates each ComputeBuffer lazily right before its first
        //     upload and SetDatas in slices, both charged to a per-frame byte budget — neither the
        //     GPU allocation burst nor the copies can land on a single frame.
        //   - After a job's upload its byte[] returns to the pool and its IO slot is released, so at
        //     most MAX_IO_JOBS files of data exist in managed memory at any moment (~120 MB peak).
        public class Qwen3_5Weights : IDisposable
        {
            // Tied embedding (also serves as lm_head) + final RMSNorm. Backed by 1-element slots so
            // the UploadPump can create them lazily like the per-layer buffers.
            public ComputeBuffer embedLmHead => _embedSlot[0];
            public ComputeBuffer finalNormGamma => _finalNormSlot[0];
            readonly ComputeBuffer[] _embedSlot = new ComputeBuffer[1];
            readonly ComputeBuffer[] _finalNormSlot = new ComputeBuffer[1];

            // Per-layer common
            public ComputeBuffer[] inputLnGamma;
            public ComputeBuffer[] postAttnLnGamma;
            public ComputeBuffer[] mlpGate, mlpUp, mlpDown;

            // Full-attention layers (null on linear layers)
            public ComputeBuffer[] W_Q;        // [hidden, heads_q * head_dim * 2]   (Q + gate)
            public ComputeBuffer[] W_K;        // [hidden, heads_kv * head_dim]
            public ComputeBuffer[] W_V;        // [hidden, heads_kv * head_dim]
            public ComputeBuffer[] W_O;        // [heads_q * head_dim, hidden]
            public ComputeBuffer[] qNormGamma; // [head_dim]
            public ComputeBuffer[] kNormGamma; // [head_dim]

            // Linear (Gated DeltaNet) layers (null on full layers)
            public ComputeBuffer[] W_inProjQKV;     // [hidden, key_dim*2 + value_dim]
            public ComputeBuffer[] W_inProjZ;       // [hidden, value_dim]
            public ComputeBuffer[] W_inProjA;       // [hidden, num_v_heads]
            public ComputeBuffer[] W_inProjB;       // [hidden, num_v_heads]
            public ComputeBuffer[] convWeight;      // [conv_dim, kernel_size]   depthwise
            public ComputeBuffer[] dtBias;          // [num_v_heads]
            public ComputeBuffer[] ALog;            // [num_v_heads]
            public ComputeBuffer[] linearNormGamma; // [head_v_dim]
            public ComputeBuffer[] W_outProj;       // [value_dim, hidden]

            // INT8 mode only (see import_params.py --quant int8): one fp16 scale per output row of
            // each quantized matrix; the weight buffers above then hold int8 packed 4-per-uint.
            // All null in FP16 mode. Norm gammas / conv / dt_bias / A_log / in_proj_a/b stay FP16.
            public readonly LLMQuant Quant;
            public ComputeBuffer embedScales => _embedScalesSlot[0];
            readonly ComputeBuffer[] _embedScalesSlot = new ComputeBuffer[1];
            public ComputeBuffer[] W_QScales, W_KScales, W_VScales, W_OScales;
            public ComputeBuffer[] W_inProjQKVScales, W_inProjZScales, W_outProjScales;
            public ComputeBuffer[] mlpGateScales, mlpUpScales, mlpDownScales;

            public bool IsReady { get; private set; }

            // Boot timings (ms), filled across construction + upload, read once by the consolidated
            // "model booted up" log emitted from Qwen3_5ForCausalLM.InitializeChat.
            public double allocMs, bootTokenizerMs, bootKernelsMs, bootCacheMs, bootRopeMs, bootScratchMs;
            public double uploadMs, worstUploadMs, ropeAsyncMs, warmupMs;
            public int uploadFrames;

            readonly int numLayers, hidden, headDim, headsQ, headsKV, intermediate, vocab;
            readonly int keyDim, valueDim, qkvLinDim, convDim, kernelSize, numVHeads, headVDim;
            readonly Qwen3_5LayerType[] layerTypes;

            const int EMBED_NUM_CHUNKS = 16;
            // Per-frame main-thread GPU budget in BYTES — the boot-vs-framedrop knob now lives on
            // the shared LLM base (LLM.UploadBudgetBytes) so it can be swept between boots; read
            // live each frame in UploadPump below.
            // Max files simultaneously in flight (being read or sitting in the upload queue). This
            // bounds boot-time managed memory to ~MAX_IO_JOBS * largest-file (~30 MB embed shards).
            const int MAX_IO_JOBS = 4;

            // One entry per .bin file. slot[slotIndex] is created lazily by the pump; the embedding
            // shards all target the same slot at different byte offsets.
            struct FileJob
            {
                public string path;
                public ComputeBuffer[] slot;
                public int slotIndex;
                public int bufferHalfCount;  // size of the (whole) target buffer, in fp16 halves
                public int fileHalfCount;    // size of this file, in fp16 halves
                public int dstByteOffset;    // byte offset within the target buffer
            }

            struct UploadJob
            {
                public ComputeBuffer[] slot;
                public int slotIndex;
                public int bufferHalfCount;
                public byte[] data;          // pooled; returned to the pool after upload
                public int dstByteOffset;
            }

            readonly List<FileJob> _manifest = new List<FileJob>();
            readonly ConcurrentQueue<UploadJob> _uploads = new ConcurrentQueue<UploadJob>();
            readonly SemaphoreSlim _ioGate = new SemaphoreSlim(MAX_IO_JOBS);
            // Exact-size byte[] pool keyed by length — weight files come in ~a dozen distinct sizes,
            // so a handful of arrays get reused for all ~270 files.
            readonly ConcurrentDictionary<int, ConcurrentStack<byte[]>> _pool
                = new ConcurrentDictionary<int, ConcurrentStack<byte[]>>();
            volatile bool _allReadsEnqueued;
            volatile bool _disposed;
            int _jobsUploaded;
            string _rootPath;      // weights folder — identity for the standardized [GPU] log lines
            long _bytesUploaded;
            long _bytesTotal;

            /// <summary>Total manifest bytes / bytes already resident — surfaced through
            /// LLM.TotalWeightBytes so game code can pace the stream (NPC walk-up prefetch).</summary>
            public long BytesTotal => _bytesTotal;
            public long BytesUploaded => _bytesUploaded;

            public Qwen3_5Weights(string paramsPath, LLMQuant quant = LLMQuant.FP16)
            {
                paramsPath = DeepUnityMeta.ResolvePath(paramsPath);   // player builds: StreamingAssets
                // The exported params are large and may not be checked into the repo — point the
                // user at the exporter scripts instead of letting 270 file-not-found errors rain.
                if (!Directory.Exists(paramsPath))
                    throw new DirectoryNotFoundException(
                        $"Qwen3.5 weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/InferenceEngine/import_params.py — e.g. `python import_params.py Qwen/Qwen3.5-0.8B " +
                        "--quant fp16|int8|int4` downloads the checkpoint and exports the params folder under " +
                        "Assets/Resources/Weights/.");

                Quant = quant;
                hidden = Qwen3_5Config.HIDDEN_SIZE;
                headDim = Qwen3_5Config.HEAD_DIM;
                headsQ = Qwen3_5Config.HEADS_Q;
                headsKV = Qwen3_5Config.HEADS_KV;
                intermediate = Qwen3_5Config.MLP_INTERMEDIATE_SIZE;
                vocab = Qwen3_5Config.VOCAB_SIZE;
                numLayers = Qwen3_5Config.NUM_LAYERS;
                layerTypes = Qwen3_5Config.layer_types;

                keyDim    = Qwen3_5Config.LINEAR_KEY_DIM;
                valueDim  = Qwen3_5Config.LINEAR_VALUE_DIM;
                qkvLinDim = keyDim * 2 + valueDim;
                convDim   = Qwen3_5Config.LINEAR_CONV_DIM;
                kernelSize = Qwen3_5Config.LINEAR_CONV_KERNEL_DIM;
                numVHeads = Qwen3_5Config.LINEAR_NUM_VALUE_HEADS;
                headVDim  = Qwen3_5Config.LINEAR_VALUE_HEAD_DIM;

                // Only managed bookkeeping here — every ComputeBuffer is created lazily by the
                // UploadPump under the frame budget (the old version created 1.5 GB of GPU buffers
                // in this single frame).
                var swAlloc = System.Diagnostics.Stopwatch.StartNew();

                inputLnGamma    = new ComputeBuffer[numLayers];
                postAttnLnGamma = new ComputeBuffer[numLayers];
                mlpGate = new ComputeBuffer[numLayers];
                mlpUp   = new ComputeBuffer[numLayers];
                mlpDown = new ComputeBuffer[numLayers];

                W_Q = new ComputeBuffer[numLayers];
                W_K = new ComputeBuffer[numLayers];
                W_V = new ComputeBuffer[numLayers];
                W_O = new ComputeBuffer[numLayers];
                qNormGamma = new ComputeBuffer[numLayers];
                kNormGamma = new ComputeBuffer[numLayers];

                W_inProjQKV = new ComputeBuffer[numLayers];
                W_inProjZ = new ComputeBuffer[numLayers];
                W_inProjA = new ComputeBuffer[numLayers];
                W_inProjB = new ComputeBuffer[numLayers];
                convWeight = new ComputeBuffer[numLayers];
                dtBias = new ComputeBuffer[numLayers];
                ALog = new ComputeBuffer[numLayers];
                linearNormGamma = new ComputeBuffer[numLayers];
                W_outProj = new ComputeBuffer[numLayers];

                W_QScales = new ComputeBuffer[numLayers];
                W_KScales = new ComputeBuffer[numLayers];
                W_VScales = new ComputeBuffer[numLayers];
                W_OScales = new ComputeBuffer[numLayers];
                W_inProjQKVScales = new ComputeBuffer[numLayers];
                W_inProjZScales = new ComputeBuffer[numLayers];
                W_outProjScales = new ComputeBuffer[numLayers];
                mlpGateScales = new ComputeBuffer[numLayers];
                mlpUpScales = new ComputeBuffer[numLayers];
                mlpDownScales = new ComputeBuffer[numLayers];

                BuildManifest(paramsPath);
                foreach (var j in _manifest) _bytesTotal += (long)j.fileHalfCount * 2;
                _rootPath = paramsPath;

                allocMs = swAlloc.Elapsed.TotalMilliseconds;
                Qwen3_5ForCausalLM.Trace($"[w.alloc:{allocMs:0.0}] ");

                // Start the drain coroutine before the reads so it idles until the first file lands,
                // then kick off the background file reads (which enqueue upload jobs as they complete).
                ResidencyLog.Loading(ResidencyLog.Label(_rootPath), _bytesTotal, LLM.UploadBudgetBytes);
                Qwen3_5ForCausalLM.Trace($"[w.log:{swAlloc.Elapsed.TotalMilliseconds:0.0}] ");
                DeepUnityDispatcher.Run(UploadPump());
                Qwen3_5ForCausalLM.Trace($"[w.pump:{swAlloc.Elapsed.TotalMilliseconds:0.0}] ");
                // kicked on the THREAD POOL: an async method's synchronous prefix (task fan-out + the
                // first MAX_IO_JOBS reads, which pass the io-gate synchronously) otherwise runs the
                // first file reads on the MAIN thread — measured 80-280 ms = the zone-entry freeze.
                _ = Task.Run(() => LoadAllAsync());
                Qwen3_5ForCausalLM.Trace($"[w.iokick:{swAlloc.Elapsed.TotalMilliseconds:0.0}] ");
            }

            void Add(string path, ComputeBuffer[] slot, int slotIndex, int bufferHalves,
                     int fileHalves = -1, int dstByteOffset = 0)
            {
                _manifest.Add(new FileJob
                {
                    path = path,
                    slot = slot,
                    slotIndex = slotIndex,
                    bufferHalfCount = bufferHalves,
                    fileHalfCount = fileHalves < 0 ? bufferHalves : fileHalves,
                    dstByteOffset = dstByteOffset,
                });
            }

            // One big matmul weight, sized in the manifest's 2-byte "half" unit:
            //   FP16: <base>.bin       rows*cols halves
            //   INT8: <base>.int8.bin  rows*cols bytes  -> rows*cols/2 halves (4 int8 per uint)
            //         + .scales.bin    one fp16 scale per output row
            //   INT4: <base>.int4.bin  rows*cols/2 bytes -> rows*cols/4 halves (8 nibbles per uint)
            //         + .scales.bin    one fp16 scale per 32-weight group = rows*cols/32 halves
            void AddW(string basePath, ComputeBuffer[] slot, ComputeBuffer[] scaleSlot, int i, int rows, int cols)
            {
                switch (Quant)
                {
                    case LLMQuant.FP16:
                        Add(basePath + ".bin", slot, i, rows * cols);
                        break;
                    case LLMQuant.INT8:
                        Add(basePath + ".int8.bin", slot, i, rows * cols / 2);
                        Add(basePath + ".scales.bin", scaleSlot, i, rows);
                        break;
                    case LLMQuant.INT4:
                        Add(basePath + ".int4.bin", slot, i, rows * cols / 4);
                        Add(basePath + ".scales.bin", scaleSlot, i, rows * cols / 32);
                        break;
                }
            }

            void BuildManifest(string p)
            {
                // Tied embedding/lm_head is ALWAYS fp16 in every quant mode (it doubles as the
                // lm_head; quantizing it poisons every logit and collapses small models — only the
                // transformer-block linear weights are quantized). 16 row-aligned fp16 shards into
                // one buffer at consecutive offsets, no scales.
                int totalHalves = vocab * hidden;
                int perChunk = totalHalves / EMBED_NUM_CHUNKS; // exactly divisible for 0.8B
                for (int i = 0; i < EMBED_NUM_CHUNKS; i++)
                    Add($"{p}/embed_tokens/part_{i}.bin", _embedSlot, 0, totalHalves, perChunk, i * perChunk * 2);

                Add(p + "/norm.bin", _finalNormSlot, 0, hidden);

                int qProjOut = headsQ * headDim * 2; // 4096 (Q + gate)
                int kvProjOut = headsKV * headDim;   // 512
                int oIn = headsQ * headDim;

                for (int i = 0; i < numLayers; i++)
                {
                    string lp = $"{p}/layer_{i}";
                    Add(lp + "/input_layernorm.bin",          inputLnGamma,    i, hidden);
                    Add(lp + "/post_attention_layernorm.bin", postAttnLnGamma, i, hidden);
                    AddW(lp + "/mlp_gate_proj", mlpGate, mlpGateScales, i, intermediate, hidden);
                    AddW(lp + "/mlp_up_proj",   mlpUp,   mlpUpScales,   i, intermediate, hidden);
                    AddW(lp + "/mlp_down_proj", mlpDown, mlpDownScales, i, hidden, intermediate);

                    if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                    {
                        AddW(lp + "/self_attn_q_proj", W_Q, W_QScales, i, qProjOut, hidden);
                        AddW(lp + "/self_attn_k_proj", W_K, W_KScales, i, kvProjOut, hidden);
                        AddW(lp + "/self_attn_v_proj", W_V, W_VScales, i, kvProjOut, hidden);
                        AddW(lp + "/self_attn_o_proj", W_O, W_OScales, i, hidden, oIn);
                        Add(lp + "/self_attn_q_norm.bin", qNormGamma, i, headDim);
                        Add(lp + "/self_attn_k_norm.bin", kNormGamma, i, headDim);
                    }
                    else
                    {
                        AddW(lp + "/linear_in_proj_qkv", W_inProjQKV, W_inProjQKVScales, i, qkvLinDim, hidden);
                        AddW(lp + "/linear_in_proj_z",   W_inProjZ,   W_inProjZScales,   i, valueDim, hidden);
                        Add(lp + "/linear_in_proj_a.bin",   W_inProjA,   i, hidden * numVHeads);
                        Add(lp + "/linear_in_proj_b.bin",   W_inProjB,   i, hidden * numVHeads);
                        Add(lp + "/linear_conv1d.bin",      convWeight,  i, convDim * kernelSize);
                        Add(lp + "/linear_dt_bias.bin",     dtBias,      i, numVHeads);
                        Add(lp + "/linear_A_log.bin",       ALog,        i, numVHeads);
                        Add(lp + "/linear_norm.bin",        linearNormGamma, i, headVDim);
                        AddW(lp + "/linear_out_proj",    W_outProj, W_outProjScales, i, hidden, valueDim);
                    }
                }
            }

            // FP16 packed: 2 halves per 4-byte uint.
            static ComputeBuffer HalfBuf(int halfCount)
            {
                if ((halfCount & 1) != 0)
                    throw new ArgumentException($"HalfBuf needs even count, got {halfCount}");
                return new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            }

            byte[] Rent(int size)
                => _pool.TryGetValue(size, out var stack) && stack.TryPop(out var arr) ? arr : new byte[size];

            void ReturnToPool(byte[] arr)
                => _pool.GetOrAdd(arr.Length, _ => new ConcurrentStack<byte[]>()).Push(arr);

            async Task LoadAllAsync()
            {
                // ConfigureAwait(false) everywhere: continuations must never marshal back to Unity's
                // main-thread SynchronizationContext, or the file reads would land on the game loop.
                try
                {
                    var tasks = new Task[_manifest.Count];
                    for (int i = 0; i < _manifest.Count; i++)
                        tasks[i] = ReadOneAsync(_manifest[i]);
                    await Task.WhenAll(tasks).ConfigureAwait(false);
                }
                catch (Exception e)
                {
                    Debug.LogException(e);
                }
                finally
                {
                    _allReadsEnqueued = true; // once the queue drains, UploadPump knows nothing more is coming
                }
            }

            async Task ReadOneAsync(FileJob f)
            {
                await _ioGate.WaitAsync().ConfigureAwait(false); // released by UploadPump after this job uploads
                try
                {
                    int byteLen = f.fileHalfCount * 2;
                    byte[] data = Rent(byteLen);
                    using (var fs = new FileStream(f.path, FileMode.Open, FileAccess.Read, FileShare.Read,
                                                   64 * 1024, FileOptions.SequentialScan))
                    {
                        if (fs.Length != byteLen)
                            throw new IOException($"Bad size {fs.Length}, expected {byteLen} for {f.path}");
                        int off = 0;
                        while (off < byteLen)
                        {
                            int n = fs.Read(data, off, byteLen - off);
                            if (n <= 0) throw new IOException($"Unexpected EOF at {off}/{byteLen} in {f.path}");
                            off += n;
                        }
                    }
                    _uploads.Enqueue(new UploadJob
                    {
                        slot = f.slot,
                        slotIndex = f.slotIndex,
                        bufferHalfCount = f.bufferHalfCount,
                        data = data,
                        dstByteOffset = f.dstByteOffset,
                    });
                }
                catch
                {
                    _ioGate.Release(); // job never reaches the pump; free its slot here
                    throw;
                }
            }

            // Single main-thread consumer: drains the upload queue under the per-frame byte budget.
            // Creates each target buffer lazily right before its first upload (charged at full size),
            // slices every SetData (byte[] overload — offsets/counts are in bytes), then recycles the
            // job's array and IO slot. Flips IsReady once everything is uploaded.
            // #31 probes: lets Qwen3_5Model.LoadBlockingForProbe drive a fresh pump synchronously in
            // edit mode (the ctor-started dispatcher coroutine never ticks without a player loop;
            // both drain the same concurrent queue, so running a second pump is harmless).
            internal IEnumerator EditorUploadPump() => UploadPump();

            IEnumerator UploadPump()
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var slice = new System.Diagnostics.Stopwatch();
                int frames = 0;
                double worstSliceMs = 0;
                long budget = LLM.UploadBudgetBytes;

                while (true)
                {
                    if (_disposed) yield break; // model released mid-load (e.g. play mode exited)

                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        if (job.slot[job.slotIndex] == null)
                        {
                            if (budget <= 0) { yield return null; frames++; if (_disposed) yield break; budget = LLM.UploadBudgetBytes; }
                            // big allocations right after a release stall the driver mid-cleanup
                            // (measured 250-550 ms single-frame spikes) — give it one present first
                            if ((long)job.bufferHalfCount * 2 > 48_000_000) { yield return null; frames++; if (_disposed) yield break; }
                            slice.Restart();
                            job.slot[job.slotIndex] = HalfBuf(job.bufferHalfCount);
                            double cms = slice.Elapsed.TotalMilliseconds; // spikes here = driver allocation
                            if (cms > worstSliceMs) worstSliceMs = cms;
                            budget -= (long)job.bufferHalfCount * 2;
                        }
                        ComputeBuffer target = job.slot[job.slotIndex];

                        int src = 0, len = job.data.Length;
                        while (src < len)
                        {
                            if (_disposed) yield break;
                            if (budget <= 0)
                            {
                                yield return null;               // hand the frame back to rendering
                                frames++;
                                // released while we slept on the budget — the buffer under
                                // `target` is gone; touching it throws (zone-exit mid-stream)
                                if (_disposed) yield break;
                                budget = LLM.UploadBudgetBytes;
                            }
                            int count = (int)Math.Min(budget, len - src);
                            count &= ~3;                 // SetData needs stride(4)-aligned byte counts
                            if (count == 0) count = 4;   // never stall on a sub-word budget
                            slice.Restart();
                            target.SetData(job.data, src, job.dstByteOffset + src, count);
                            double ms = slice.Elapsed.TotalMilliseconds; // spikes here = first-touch GPU commit
                            if (ms > worstSliceMs) worstSliceMs = ms;
                            src += count;
                            budget -= count;
                        }

                        ReturnToPool(job.data);
                        _ioGate.Release();                       // frees a slot for the next background read
                        _jobsUploaded++;
                        _bytesUploaded += len;
                    }
                    else if (_allReadsEnqueued && _uploads.IsEmpty)
                    {
                        break;                                   // no more reads in flight and queue drained => done
                    }
                    else
                    {
                        yield return null;                       // reads still in flight; check again next frame
                        frames++;
                        budget = LLM.UploadBudgetBytes;
                    }
                }

                _pool.Clear(); // drop the pooled read arrays — one small collection behind the load, not during play

                if (_jobsUploaded != _manifest.Count)
                    ConsoleMessage.Warning($"Qwen3.5 weights: only {_jobsUploaded}/{_manifest.Count} weight files uploaded " +
                                           "(missing or failed reads — see earlier exceptions). Model output will be invalid.");

                uploadFrames = frames;
                uploadMs = sw.Elapsed.TotalMilliseconds;
                worstUploadMs = worstSliceMs;
                ResidencyLog.Resident(ResidencyLog.Label(_rootPath), _bytesUploaded,
                                      uploadMs, uploadFrames, worstUploadMs);
                IsReady = true;
            }

            // Every weight buffer this loader owns (shared by the sync and budgeted teardowns).
            IEnumerable<ComputeBuffer> OwnedBuffers()
            {
                yield return _embedSlot[0];
                yield return _finalNormSlot[0];
                yield return _embedScalesSlot[0];
                for (int i = 0; i < numLayers; i++)
                {
                    yield return inputLnGamma[i]; yield return postAttnLnGamma[i];
                    yield return mlpGate[i]; yield return mlpUp[i]; yield return mlpDown[i];
                    yield return W_Q[i]; yield return W_K[i]; yield return W_V[i]; yield return W_O[i];
                    yield return qNormGamma[i]; yield return kNormGamma[i];
                    yield return W_inProjQKV[i]; yield return W_inProjZ[i];
                    yield return W_inProjA[i]; yield return W_inProjB[i];
                    yield return convWeight[i]; yield return dtBias[i]; yield return ALog[i];
                    yield return linearNormGamma[i]; yield return W_outProj[i];
                    yield return W_QScales[i]; yield return W_KScales[i]; yield return W_VScales[i];
                    yield return W_OScales[i]; yield return W_inProjQKVScales[i];
                    yield return W_inProjZScales[i]; yield return W_outProjScales[i];
                    yield return mlpGateScales[i]; yield return mlpUpScales[i]; yield return mlpDownScales[i];
                }
            }

            bool _buffersFreed;
            void BeginDispose()
            {
                _disposed = true; // stops the UploadPump before buffers vanish under it
                if (_bytesUploaded > 0)
                {
                    ResidencyLog.Released(ResidencyLog.Label(_rootPath), _bytesUploaded);
                    _bytesUploaded = 0;
                }
            }

            public void Dispose()
            {
                BeginDispose();
                if (_buffersFreed) return;
                _buffersFreed = true;
                foreach (var b in OwnedBuffers()) b?.Release();
            }

            /// <summary>Budgeted teardown: ~bytesPerFrame of buffers freed per frame — the driver
            /// digests the release incrementally instead of stalling the NEXT model's big
            /// allocations (measured 250-550 ms single-frame spikes after monolithic frees).</summary>
            public IEnumerator DisposeSlow(long bytesPerFrame)
            {
                BeginDispose();
                if (_buffersFreed) yield break;
                _buffersFreed = true;
                long spent = 0;
                foreach (var b in OwnedBuffers())
                {
                    if (b == null) continue;
                    spent += (long)b.count * b.stride;
                    b.Release();
                    if (spent >= bytesPerFrame) { spent = 0; yield return null; }
                }
            }
        }
    }
}
