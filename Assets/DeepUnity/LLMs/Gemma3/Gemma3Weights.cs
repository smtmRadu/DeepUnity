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
    namespace Gemma3Modeling
    {
        // FP16 weights packed two halves per uint, stored on the GPU as ComputeBuffer<uint>.
        //
        // Loading is streamed and allocation-flat (same design as Qwen3_5Weights — see
        // Assets/DeepUnity/LLMs/OPTIMIZATIONS.md for the full story):
        //   - The constructor only builds a file manifest (no GPU buffers, no reads).
        //   - Background readers (at most MAX_IO_JOBS in flight) read each .bin into a pooled
        //     byte[] of exactly the file size; arrays are reused across files so the managed heap
        //     stays flat — the previous version allocated ~2x the model size in temporary arrays
        //     and stalled the main thread on GC collections and whole-buffer SetData calls
        //     (including the entire 335 MB embedding in ONE SetData).
        //   - The main-thread UploadPump creates each ComputeBuffer lazily right before its first
        //     upload and SetDatas in slices, both charged to a per-frame byte budget.
        //   - Concatenated buffers (W_QKV = q|k|v, mlpWeights = gate|up|down, the lm_head shards)
        //     are simply manifest entries targeting the same buffer at different byte offsets.
        public class Gemma3Weights : IDisposable
        {
            public ComputeBuffer embedLmHead => _embedSlot[0];
            public ComputeBuffer finalNormGamma => _finalNormSlot[0];
            readonly ComputeBuffer[] _embedSlot = new ComputeBuffer[1];
            readonly ComputeBuffer[] _finalNormSlot = new ComputeBuffer[1];

            public ComputeBuffer[] W_QKV;
            public ComputeBuffer[] W_O;
            public ComputeBuffer[] mlpWeights;
            public ComputeBuffer[] qNormGamma;
            public ComputeBuffer[] kNormGamma;
            public ComputeBuffer[] inputLnGamma;
            public ComputeBuffer[] postAttnLnGamma;
            public ComputeBuffer[] preFfnLnGamma;
            public ComputeBuffer[] postFfnLnGamma;

            // Quantized modes only (see import_params.py --quant int8|int4). INT8: one fp16 scale
            // per output row; the concatenated W_QKV / mlpWeights buffers hold int8 packed 4-per-uint.
            // INT4 (GGUF Q4_0): weights packed 8-per-uint + one fp16 scale per 32-weight group, the
            // scales concatenated in the same q|k|v / gate|up|down order as the weights. All null in
            // FP16 mode. Norm gammas stay FP16 in every mode.
            public readonly LLMQuant Quant;
            public ComputeBuffer embedScales => _embedScalesSlot[0];   // [vocab]; tied lm_head shares it
            readonly ComputeBuffer[] _embedScalesSlot = new ComputeBuffer[1];
            public ComputeBuffer[] W_QKVScales;   // [qkv_proj_dim]
            public ComputeBuffer[] W_OScales;     // [hidden]
            public ComputeBuffer[] mlpScales;     // [2*intermediate + hidden]

            public bool IsReady { get; private set; }

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;

            // Per-frame main-thread GPU budget in BYTES (lazy buffer creation + SetData slices) —
            // the boot-vs-framedrop knob now lives on the shared LLM base (LLM.UploadBudgetBytes)
            // so it can be swept between boots; read live each frame in UploadPump below.
            // Max files simultaneously in flight (being read or queued) — bounds boot RAM.
            const int MAX_IO_JOBS = 4;

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
            readonly ConcurrentDictionary<int, ConcurrentStack<byte[]>> _pool
                = new ConcurrentDictionary<int, ConcurrentStack<byte[]>>();
            volatile bool _allReadsEnqueued;
            volatile bool _disposed;
            int _jobsUploaded;

            public Gemma3Weights(string paramsPath, LLMQuant quant = LLMQuant.FP16)
            {
                // The exported params are large and may not be checked into the repo — point the
                // user at the exporter script instead of letting hundreds of file errors rain.
                if (!System.IO.Directory.Exists(paramsPath))
                    throw new System.IO.DirectoryNotFoundException(
                        $"Gemma3 weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/LLMs/import_params.py — e.g. `python import_params.py google/gemma-3-270m-it " +
                        "--quant fp16|int8|int4` downloads the checkpoint and exports the params folder under " +
                        "Assets/Resources/DeepUnity/LLMs/Gemma3/.");

                Quant = quant;
                numLayers = Gemma3Config.NUM_LAYERS;
                hiddenSize = Gemma3Config.HIDDEN_SIZE;
                headDim = Gemma3Config.HEAD_DIM;
                headsQ = Gemma3Config.HEADS_Q;
                headsKV = Gemma3Config.HEADS_KV;
                intermediateSize = Gemma3Config.MLP_INTERMEDIATE_SIZE;
                vocabSize = Gemma3Config.VOCAB_SIZE;

                float exp = Gemma3Config.ATTN_EXPANSION_FACTOR;
                innerEmbDim = (int)(hiddenSize * exp);
                qkvProjDim = innerEmbDim + 2 * (innerEmbDim * headsKV / headsQ);

                W_QKV = new ComputeBuffer[numLayers];
                W_O = new ComputeBuffer[numLayers];
                mlpWeights = new ComputeBuffer[numLayers];
                qNormGamma = new ComputeBuffer[numLayers];
                kNormGamma = new ComputeBuffer[numLayers];
                inputLnGamma = new ComputeBuffer[numLayers];
                postAttnLnGamma = new ComputeBuffer[numLayers];
                preFfnLnGamma = new ComputeBuffer[numLayers];
                postFfnLnGamma = new ComputeBuffer[numLayers];

                W_QKVScales = new ComputeBuffer[numLayers];
                W_OScales = new ComputeBuffer[numLayers];
                mlpScales = new ComputeBuffer[numLayers];

                BuildManifest(paramsPath);

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

            // One group of row-partitioned matmul weights written into ONE weight buffer + ONE
            // scale buffer, concatenated by output row (q|k|v, gate|up|down, or a single matrix).
            // Sizes are in the manifest's 2-byte "half" unit:
            //   FP16: <bp>.bin       rows*cols halves
            //   INT8: <bp>.int8.bin  rows*cols bytes -> rows*cols/2 halves (4 int8 per uint)
            //         + <bp>.scales.bin  one fp16 scale per output row
            //   INT4: <bp>.int4.bin  rows*cols/2 bytes -> rows*cols/4 halves (8 nibbles per uint)
            //         + <bp>.scales.bin  one fp16 scale per 32-weight group = rows*cols/32 halves
            // dstByteOffsets walk the parts in order. The shader's wScale row index is the part's
            // row offset within this concatenation (q|k|v / gate|up|down) for INT8; INT4 folds
            // wScale to 1.0 and indexes the group scale by the flat weight index >> 5, which lines
            // up because both weights and scales are concatenated in the same part order.
            void AddConcatW(ComputeBuffer[] wSlot, ComputeBuffer[] sSlot, int i, (string bp, int rows, int cols)[] parts)
            {
                int totElems = 0, totRows = 0;
                foreach (var pp in parts) { totElems += pp.rows * pp.cols; totRows += pp.rows; }

                if (Quant == LLMQuant.FP16)
                {
                    int bytePos = 0;
                    foreach (var pp in parts)
                    {
                        int elems = pp.rows * pp.cols;
                        Add(pp.bp + ".bin", wSlot, i, totElems, elems, bytePos);
                        bytePos += elems * 2;               // fp16 = 2 bytes/elem
                    }
                    return;
                }

                if (Quant == LLMQuant.INT8)
                {
                    // weights packed 4-per-uint (1 byte/elem) + one fp16 scale per output row.
                    int wPos = 0;
                    foreach (var pp in parts)
                    {
                        int elems = pp.rows * pp.cols;
                        Add(pp.bp + ".int8.bin", wSlot, i, totElems / 2, elems / 2, wPos);
                        wPos += elems;                      // int8 = 1 byte/elem
                    }
                    int sPos = 0;
                    foreach (var pp in parts)
                    {
                        Add(pp.bp + ".scales.bin", sSlot, i, totRows, pp.rows, sPos);
                        sPos += pp.rows * 2;                // fp16 scale = 2 bytes/row
                    }
                    return;
                }

                // INT4: weights packed 8-per-uint (0.5 byte/elem) + one fp16 scale per 32-weight
                // group. Every part's cols are a multiple of 32, so groups never straddle a part.
                int wPos4 = 0;
                foreach (var pp in parts)
                {
                    int elems = pp.rows * pp.cols;
                    Add(pp.bp + ".int4.bin", wSlot, i, totElems / 4, elems / 4, wPos4);
                    wPos4 += elems / 2;                     // int4 = 0.5 byte/elem
                }
                int sPos4 = 0;
                foreach (var pp in parts)
                {
                    int groups = pp.rows * pp.cols / 32;
                    Add(pp.bp + ".scales.bin", sSlot, i, totElems / 32, groups, sPos4);
                    sPos4 += groups * 2;                    // fp16 scale = 2 bytes/group
                }
            }

            void BuildManifest(string p)
            {
                // Tied embedding/lm_head shards into one buffer at running offsets. Unified
                // convention (import_params.py): embed_tokens/part_{0..15}[.int8].bin, 16 equal
                // row-aligned shards (+ one scales.bin covering the whole matrix when int8).
                // Legacy export: lm_head/part_{0..13}.bin, 14 torch.chunk shards (uneven, fp16
                // only). Detect by which folder exists.
                if (Directory.Exists(Path.Combine(p, "embed_tokens")))
                {
                    // Tied embedding/lm_head is ALWAYS fp16 in every quant mode (it's the lm_head;
                    // quantizing it poisons every logit). 16 row-aligned fp16 shards, no scales.
                    int totalHalves = vocabSize * hiddenSize;
                    int perChunk = totalHalves / 16;
                    for (int i = 0; i < 16; i++)
                        Add($"{p}/embed_tokens/part_{i}.bin", _embedSlot, 0, totalHalves, perChunk, i * perChunk * 2);
                }
                else
                {
                    int embedHalves = vocabSize * hiddenSize;
                    int[] partSizes =
                    {
                        11_983_726, 11_983_726, 11_983_726, 11_983_726,
                        11_983_726, 11_983_726, 11_983_726, 11_983_726,
                        11_983_726, 11_983_726, 11_983_726, 11_983_726,
                        11_983_726, 11_983_722
                    };
                    int offset = 0;
                    for (int i = 0; i < partSizes.Length; i++)
                    {
                        Add($"{p}/lm_head/part_{i}.bin", _embedSlot, 0, embedHalves, partSizes[i], offset);
                        offset += partSizes[i] * 2;
                    }
                }

                Add(p + "/norm.bin", _finalNormSlot, 0, hiddenSize);

                int qOut = innerEmbDim;                          // q_proj output rows
                int kvOut = innerEmbDim * headsKV / headsQ;      // k/v_proj output rows

                for (int i = 0; i < numLayers; i++)
                {
                    string lp = $"{p}/layer_{i}";

                    // q|k|v concatenated into one W_QKV buffer (+ q|k|v scales when int8)
                    AddConcatW(W_QKV, W_QKVScales, i, new[]
                    {
                        (lp + "/self_attn_q_proj", qOut,  hiddenSize),
                        (lp + "/self_attn_k_proj", kvOut, hiddenSize),
                        (lp + "/self_attn_v_proj", kvOut, hiddenSize),
                    });

                    AddConcatW(W_O, W_OScales, i, new[]
                    {
                        (lp + "/self_attn_o_proj", hiddenSize, innerEmbDim),
                    });

                    // gate|up|down concatenated into one mlpWeights buffer (+ scales when int8)
                    AddConcatW(mlpWeights, mlpScales, i, new[]
                    {
                        (lp + "/mlp_gate_proj", intermediateSize, hiddenSize),
                        (lp + "/mlp_up_proj",   intermediateSize, hiddenSize),
                        (lp + "/mlp_down_proj", hiddenSize,       intermediateSize),
                    });

                    Add(lp + "/self_attn_q_norm.bin", qNormGamma, i, headDim);
                    Add(lp + "/self_attn_k_norm.bin", kNormGamma, i, headDim);
                    Add(lp + "/input_layernorm.bin",            inputLnGamma,    i, hiddenSize);
                    Add(lp + "/post_attention_layernorm.bin",   postAttnLnGamma, i, hiddenSize);
                    Add(lp + "/pre_feedforward_layernorm.bin",  preFfnLnGamma,   i, hiddenSize);
                    Add(lp + "/post_feedforward_layernorm.bin", postFfnLnGamma,  i, hiddenSize);
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
                    _allReadsEnqueued = true;
                }
            }

            async Task ReadOneAsync(FileJob f)
            {
                await _ioGate.WaitAsync().ConfigureAwait(false); // released by UploadPump after upload
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
                    _ioGate.Release();
                    throw;
                }
            }

            // Single main-thread consumer: drains the upload queue under the per-frame byte budget.
            // Creates each target buffer lazily right before its first upload (charged at full size),
            // slices every SetData (byte[] overload — offsets/counts are in bytes), then recycles the
            // job's array and IO slot. Flips IsReady once everything is uploaded.
            IEnumerator UploadPump()
            {
                long budget = LLM.UploadBudgetBytes;

                while (true)
                {
                    if (_disposed) yield break; // model released mid-load (e.g. play mode exited)

                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        if (job.slot[job.slotIndex] == null)
                        {
                            if (budget <= 0) { yield return null; budget = LLM.UploadBudgetBytes; }
                            job.slot[job.slotIndex] = HalfBuf(job.bufferHalfCount);
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
                                budget = LLM.UploadBudgetBytes;
                            }
                            int count = (int)Math.Min(budget, len - src);
                            target.SetData(job.data, src, job.dstByteOffset + src, count);
                            src += count;
                            budget -= count;
                        }

                        ReturnToPool(job.data);
                        _ioGate.Release();
                        _jobsUploaded++;
                    }
                    else if (_allReadsEnqueued && _uploads.IsEmpty)
                    {
                        break;
                    }
                    else
                    {
                        yield return null;
                        budget = LLM.UploadBudgetBytes;
                    }
                }

                _pool.Clear();

                if (_jobsUploaded != _manifest.Count)
                    ConsoleMessage.Warning($"Gemma3 weights: only {_jobsUploaded}/{_manifest.Count} weight files uploaded " +
                                           "(missing or failed reads — see earlier exceptions). Model output will be invalid.");
                else
                    ConsoleMessage.Info($"Gemma3-270m {Quant} weights streamed to GPU.");

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
                    W_QKV[i]?.Release(); W_O[i]?.Release(); mlpWeights[i]?.Release();
                    qNormGamma[i]?.Release(); kNormGamma[i]?.Release();
                    inputLnGamma[i]?.Release(); postAttnLnGamma[i]?.Release();
                    preFfnLnGamma[i]?.Release(); postFfnLnGamma[i]?.Release();
                    W_QKVScales[i]?.Release(); W_OScales[i]?.Release(); mlpScales[i]?.Release();
                }
            }
        }
    }
}
