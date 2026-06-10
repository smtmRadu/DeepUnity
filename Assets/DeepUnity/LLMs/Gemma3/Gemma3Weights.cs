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

            public bool IsReady { get; private set; }

            readonly int numLayers, hiddenSize, headDim, headsQ, headsKV;
            readonly int innerEmbDim, qkvProjDim, intermediateSize, vocabSize;

            // Per-frame main-thread GPU budget in BYTES (lazy buffer creation + SetData slices).
            const int UPLOAD_BUDGET_BYTES = 24 * 1024 * 1024;
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

            public Gemma3Weights(string paramsPath)
            {
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

            void BuildManifest(string p)
            {
                // lm_head shards (uneven sizes) into one tied embedding buffer at running offsets.
                int[] partSizes =
                {
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_726, 11_983_726, 11_983_726,
                    11_983_726, 11_983_722
                };
                int embedHalves = vocabSize * hiddenSize;
                int offset = 0;
                for (int i = 0; i < partSizes.Length; i++)
                {
                    Add($"{p}/lm_head/part_{i}.bin", _embedSlot, 0, embedHalves, partSizes[i], offset);
                    offset += partSizes[i] * 2;
                }

                Add(p + "/norm.bin", _finalNormSlot, 0, hiddenSize);

                int qHalves = hiddenSize * innerEmbDim;
                int kvHalves = hiddenSize * innerEmbDim * headsKV / headsQ;
                int qkvHalves = hiddenSize * qkvProjDim;
                int oHalves = innerEmbDim * hiddenSize;
                int mlpPart = hiddenSize * intermediateSize;

                for (int i = 0; i < numLayers; i++)
                {
                    string lp = $"{p}/layer_{i}";

                    // q|k|v concatenated into one W_QKV buffer
                    Add(lp + "/self_attn_q_proj.bin", W_QKV, i, qkvHalves, qHalves, 0);
                    Add(lp + "/self_attn_k_proj.bin", W_QKV, i, qkvHalves, kvHalves, qHalves * 2);
                    Add(lp + "/self_attn_v_proj.bin", W_QKV, i, qkvHalves, kvHalves, (qHalves + kvHalves) * 2);

                    Add(lp + "/self_attn_o_proj.bin", W_O, i, oHalves);

                    // gate|up|down concatenated into one mlpWeights buffer
                    Add(lp + "/mlp_gate_proj.bin", mlpWeights, i, mlpPart * 3, mlpPart, 0);
                    Add(lp + "/mlp_up_proj.bin",   mlpWeights, i, mlpPart * 3, mlpPart, mlpPart * 2);
                    Add(lp + "/mlp_down_proj.bin", mlpWeights, i, mlpPart * 3, mlpPart, mlpPart * 4);

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
                long budget = UPLOAD_BUDGET_BYTES;

                while (true)
                {
                    if (_disposed) yield break; // model released mid-load (e.g. play mode exited)

                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        if (job.slot[job.slotIndex] == null)
                        {
                            if (budget <= 0) { yield return null; budget = UPLOAD_BUDGET_BYTES; }
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
                                budget = UPLOAD_BUDGET_BYTES;
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
                        budget = UPLOAD_BUDGET_BYTES;
                    }
                }

                _pool.Clear();

                if (_jobsUploaded != _manifest.Count)
                    ConsoleMessage.Warning($"Gemma3 weights: only {_jobsUploaded}/{_manifest.Count} weight files uploaded " +
                                           "(missing or failed reads — see earlier exceptions). Model output will be invalid.");
                else
                    ConsoleMessage.Info("Gemma3 weights streamed to GPU.");

                IsReady = true;
            }

            public void Dispose()
            {
                _disposed = true; // stops the UploadPump before buffers vanish under it

                _embedSlot[0]?.Release();
                _finalNormSlot[0]?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    W_QKV[i]?.Release(); W_O[i]?.Release(); mlpWeights[i]?.Release();
                    qNormGamma[i]?.Release(); kNormGamma[i]?.Release();
                    inputLnGamma[i]?.Release(); postAttnLnGamma[i]?.Release();
                    preFfnLnGamma[i]?.Release(); postFfnLnGamma[i]?.Release();
                }
            }
        }
    }
}
