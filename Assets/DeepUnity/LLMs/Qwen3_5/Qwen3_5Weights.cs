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

            public Qwen3_5Weights(string paramsPath, LLMQuant quant = LLMQuant.FP16)
            {
                // The exported params are large and may not be checked into the repo — point the
                // user at the exporter scripts instead of letting 270 file-not-found errors rain.
                if (!Directory.Exists(paramsPath))
                    throw new DirectoryNotFoundException(
                        $"Qwen3.5 weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/LLMs/import_params.py — e.g. `python import_params.py Qwen/Qwen3.5-0.8B " +
                        "--quant fp16|int8|int4` downloads the checkpoint and exports the params folder under " +
                        "Assets/Resources/DeepUnity/LLMs/Qwen3_5/.");

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

                allocMs = swAlloc.Elapsed.TotalMilliseconds;

                // Start the drain coroutine before the reads so it idles until the first file lands,
                // then kick off the background file reads (which enqueue upload jobs as they complete).
                DeepUnityDispatcher.Run(UploadPump());
                _ = LoadAllAsync();
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
                // Embedding: 16 shards into one buffer at consecutive offsets (row-aligned chunks
                // in every mode, plus one scales file covering the whole matrix when quantized).
                string embedExt = Quant == LLMQuant.INT8 ? ".int8.bin"
                                : Quant == LLMQuant.INT4 ? ".int4.bin" : ".bin";
                int divisor = Quant == LLMQuant.INT8 ? 2 : Quant == LLMQuant.INT4 ? 4 : 1;
                int totalHalves = vocab * hidden / divisor;
                int perChunk = totalHalves / EMBED_NUM_CHUNKS; // exactly divisible for 0.8B
                for (int i = 0; i < EMBED_NUM_CHUNKS; i++)
                    Add($"{p}/embed_tokens/part_{i}{embedExt}", _embedSlot, 0, totalHalves, perChunk, i * perChunk * 2);
                if (Quant == LLMQuant.INT8)
                    Add($"{p}/embed_tokens/scales.bin", _embedScalesSlot, 0, vocab);
                else if (Quant == LLMQuant.INT4)
                    Add($"{p}/embed_tokens/scales.bin", _embedScalesSlot, 0, vocab * hidden / 32);

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
                            if (budget <= 0) { yield return null; frames++; budget = LLM.UploadBudgetBytes; }
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
                                budget = LLM.UploadBudgetBytes;
                            }
                            int count = (int)Math.Min(budget, len - src);
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
                IsReady = true;
            }

            public void Dispose()
            {
                _disposed = true; // stops the UploadPump before buffers vanish under it

                _embedSlot[0]?.Release();
                _finalNormSlot[0]?.Release();
                _embedScalesSlot[0]?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    inputLnGamma[i]?.Release();
                    postAttnLnGamma[i]?.Release();
                    mlpGate[i]?.Release();
                    mlpUp[i]?.Release();
                    mlpDown[i]?.Release();
                    W_Q[i]?.Release();
                    W_K[i]?.Release();
                    W_V[i]?.Release();
                    W_O[i]?.Release();
                    qNormGamma[i]?.Release();
                    kNormGamma[i]?.Release();
                    W_inProjQKV[i]?.Release();
                    W_inProjZ[i]?.Release();
                    W_inProjA[i]?.Release();
                    W_inProjB[i]?.Release();
                    convWeight[i]?.Release();
                    dtBias[i]?.Release();
                    ALog[i]?.Release();
                    linearNormGamma[i]?.Release();
                    W_outProj[i]?.Release();
                    W_QScales[i]?.Release();
                    W_KScales[i]?.Release();
                    W_VScales[i]?.Release();
                    W_OScales[i]?.Release();
                    W_inProjQKVScales[i]?.Release();
                    W_inProjZScales[i]?.Release();
                    W_outProjScales[i]?.Release();
                    mlpGateScales[i]?.Release();
                    mlpUpScales[i]?.Release();
                    mlpDownScales[i]?.Release();
                }
            }
        }
    }
}
