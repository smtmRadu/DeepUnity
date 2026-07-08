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
    namespace MiniCPM5Modeling
    {
        // FP16 weights packed two halves per uint, stored on the GPU as ComputeBuffer<uint>.
        // Streamed, allocation-flat loading — same design as Gemma3Weights/Qwen3_5Weights (see
        // Assets/DeepUnity/LLM/OPTIMIZATIONS.md).
        //
        // MiniCPM5 (vanilla llama) differences from Gemma3:
        //   - UNTIED lm_head: `lmHead` is its own vocab x hidden matrix (16 shards under lm_head/),
        //     loaded next to the embedding. Both ALWAYS stay fp16 (quantizing the logit matrix
        //     collapses small models — same rule as the tied families).
        //   - No qk-norm gammas, no pre/post-feedforward norms: just input_layernorm +
        //     post_attention_layernorm per layer.
        public class MiniCPM5Weights : IDisposable
        {
            public ComputeBuffer embed => _embedSlot[0];
            public ComputeBuffer lmHead => _lmHeadSlot[0];
            public ComputeBuffer finalNormGamma => _finalNormSlot[0];
            readonly ComputeBuffer[] _embedSlot = new ComputeBuffer[1];
            readonly ComputeBuffer[] _lmHeadSlot = new ComputeBuffer[1];
            readonly ComputeBuffer[] _finalNormSlot = new ComputeBuffer[1];

            public ComputeBuffer[] W_QKV;
            public ComputeBuffer[] W_O;
            public ComputeBuffer[] mlpWeights;
            public ComputeBuffer[] inputLnGamma;
            public ComputeBuffer[] postAttnLnGamma;

            // Quantized modes only (import_params.py --quant int8|int4); layout identical to
            // Gemma3Weights. Norm gammas + embedding + lm_head stay FP16 in every mode.
            public readonly LLMQuant Quant;
            public ComputeBuffer[] W_QKVScales;   // [qkv_proj_dim]
            public ComputeBuffer[] W_OScales;     // [hidden]
            public ComputeBuffer[] mlpScales;     // [2*intermediate + hidden]

            public bool IsReady { get; private set; }

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;

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

            public MiniCPM5Weights(string paramsPath, LLMQuant quant = LLMQuant.FP16)
            {
                if (!Directory.Exists(paramsPath))
                    throw new DirectoryNotFoundException(
                        $"MiniCPM5 weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/LLM/import_params.py — e.g. `python import_params.py openbmb/MiniCPM5-1B " +
                        "--quant fp16|int8|int4` downloads the checkpoint and exports the params folder under " +
                        "Assets/Resources/DeepUnity/LLM/MiniCPM5/.");

                Quant = quant;
                numLayers = MiniCPM5Config.NUM_LAYERS;
                hiddenSize = MiniCPM5Config.HIDDEN_SIZE;
                headDim = MiniCPM5Config.HEAD_DIM;
                headsQ = MiniCPM5Config.HEADS_Q;
                headsKV = MiniCPM5Config.HEADS_KV;
                intermediateSize = MiniCPM5Config.MLP_INTERMEDIATE_SIZE;
                vocabSize = MiniCPM5Config.VOCAB_SIZE;

                innerEmbDim = headsQ * headDim;                          // 2048 (≠ hidden 1536)
                qkvProjDim = innerEmbDim + 2 * (headsKV * headDim);      // q | k | v rows

                W_QKV = new ComputeBuffer[numLayers];
                W_O = new ComputeBuffer[numLayers];
                mlpWeights = new ComputeBuffer[numLayers];
                inputLnGamma = new ComputeBuffer[numLayers];
                postAttnLnGamma = new ComputeBuffer[numLayers];

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
            // Identical convention to Gemma3Weights.AddConcatW — see there for the size math.
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
                // Embedding and (untied) lm_head: 16 equal row-aligned fp16 shards each, in their
                // own folders (import_params.py convention). ALWAYS fp16, no scales.
                int totalHalves = vocabSize * hiddenSize;
                int perChunk = totalHalves / 16;
                for (int i = 0; i < 16; i++)
                    Add($"{p}/embed_tokens/part_{i}.bin", _embedSlot, 0, totalHalves, perChunk, i * perChunk * 2);
                for (int i = 0; i < 16; i++)
                    Add($"{p}/lm_head/part_{i}.bin", _lmHeadSlot, 0, totalHalves, perChunk, i * perChunk * 2);

                Add(p + "/norm.bin", _finalNormSlot, 0, hiddenSize);

                int qOut = innerEmbDim;             // q_proj output rows (heads_q * head_dim)
                int kvOut = headsKV * headDim;      // k/v_proj output rows

                for (int i = 0; i < numLayers; i++)
                {
                    string lp = $"{p}/layer_{i}";

                    // q|k|v concatenated into one W_QKV buffer (+ q|k|v scales when quantized)
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

                    // gate|up|down concatenated into one mlpWeights buffer (+ scales when quantized)
                    AddConcatW(mlpWeights, mlpScales, i, new[]
                    {
                        (lp + "/mlp_gate_proj", intermediateSize, hiddenSize),
                        (lp + "/mlp_up_proj",   intermediateSize, hiddenSize),
                        (lp + "/mlp_down_proj", hiddenSize,       intermediateSize),
                    });

                    Add(lp + "/input_layernorm.bin",          inputLnGamma,    i, hiddenSize);
                    Add(lp + "/post_attention_layernorm.bin", postAttnLnGamma, i, hiddenSize);
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
            // See Gemma3Weights.UploadPump for the design notes.
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
                    ConsoleMessage.Warning($"MiniCPM5 weights: only {_jobsUploaded}/{_manifest.Count} weight files uploaded " +
                                           "(missing or failed reads — see earlier exceptions). Model output will be invalid.");
                else
                    ConsoleMessage.Info($"MiniCPM5-1B {Quant} weights streamed to GPU.");

                IsReady = true;
            }

            public void Dispose()
            {
                _disposed = true; // stops the UploadPump before buffers vanish under it

                _embedSlot[0]?.Release();
                _lmHeadSlot[0]?.Release();
                _finalNormSlot[0]?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    W_QKV[i]?.Release(); W_O[i]?.Release(); mlpWeights[i]?.Release();
                    inputLnGamma[i]?.Release(); postAttnLnGamma[i]?.Release();
                    W_QKVScales[i]?.Release(); W_OScales[i]?.Release(); mlpScales[i]?.Release();
                }
            }
        }
    }
}
