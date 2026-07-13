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
    namespace ChatterboxModeling
    {
        // Manifest-driven weight store for the Chatterbox-Turbo export (fp16 .bin files +
        // manifest.tsv written by import_params.py --arch chatterbox).
        //
        // Same streamed/allocation-flat design as Gemma3Weights (background readers -> pooled
        // byte[] -> main-thread UploadPump under LLM.UploadBudgetBytes/frame), but GENERIC:
        // every tensor is a manifest line `name\tfile\tdtype\tnumel\tshape`, exposed via
        // Get(name). fp16 tensors land packed 2-per-uint (shader readH); q8 tensors (int8
        // matmul weights, --quant int8) land packed 4-per-uint (readQ8) with their fp16 per-row
        // scales as a sibling `<name>.scales` f16 entry; i32 tensors land as raw int buffers
        // (token ids). No hardcoded tensor list — the exporter is the schema.
        public class ChatterboxWeights : IDisposable
        {
            public bool IsReady { get; private set; }

            const int MAX_IO_JOBS = 4;

            struct Entry
            {
                public ComputeBuffer[] slot;   // 1-element slot (lazy creation on upload)
                public string dtype;           // "f16" | "i32" | "q8"
                public int numel;
                public int[] shape;
                public string file;
            }

            static int BytesPerElem(string dtype) => dtype == "i32" ? 4 : dtype == "q8" ? 1 : 2;

            struct UploadJob
            {
                public ComputeBuffer[] slot;
                public string dtype;
                public int numel;
                public byte[] data;
            }

            readonly Dictionary<string, Entry> _entries = new Dictionary<string, Entry>();
            readonly ConcurrentQueue<UploadJob> _uploads = new ConcurrentQueue<UploadJob>();
            readonly SemaphoreSlim _ioGate = new SemaphoreSlim(MAX_IO_JOBS);
            readonly ConcurrentDictionary<int, ConcurrentStack<byte[]>> _pool
                = new ConcurrentDictionary<int, ConcurrentStack<byte[]>>();
            readonly string _root;
            volatile bool _allReadsEnqueued;
            volatile bool _disposed;
            int _jobsUploaded, _jobsTotal;

            public ChatterboxWeights(string paramsPath)
            {
                if (!Directory.Exists(paramsPath))
                    throw new DirectoryNotFoundException(
                        $"Chatterbox weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/InferenceEngine/import_params.py — `python import_params.py ResembleAI/chatterbox-turbo` " +
                        "exports it under Assets/Resources/Weights/.");
                _root = paramsPath;

                string manifest = Path.Combine(paramsPath, "manifest.tsv");
                if (!File.Exists(manifest))
                    throw new FileNotFoundException($"manifest.tsv missing in {paramsPath} (re-run the exporter).");

                foreach (string line in File.ReadAllLines(manifest))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] p = line.Split('\t');
                    // name, file, dtype, numel, shape-csv
                    string[] sh = p[4].Split(',');
                    int[] shape = new int[sh.Length];
                    for (int i = 0; i < sh.Length; i++) shape[i] = int.Parse(sh[i]);
                    _entries[p[0]] = new Entry
                    {
                        slot = new ComputeBuffer[1],
                        dtype = p[2],
                        numel = int.Parse(p[3]),
                        shape = shape,
                        file = p[1],
                    };
                }
                _jobsTotal = _entries.Count;

                _loadWall.Restart();
                _loadStartFrame = UnityEngine.Time.frameCount;
                ResidencyLog.Loading(ResidencyLog.Label(_root), 0, LLM.UploadBudgetBytes);
                DeepUnityDispatcher.Run(UploadPump());
                // kicked on the THREAD POOL: an async method's synchronous prefix (task fan-out + the
                // first MAX_IO_JOBS reads, which pass the io-gate synchronously) otherwise runs the
                // first file reads on the MAIN thread — measured 80-280 ms = the zone-entry freeze.
                _ = Task.Run(() => LoadAllAsync());
            }

            readonly System.Diagnostics.Stopwatch _loadWall = new System.Diagnostics.Stopwatch();
            int _loadStartFrame;

            /// <summary>GPU buffer of an exported tensor (manifest name, e.g. "t3/layer_0/qkv.w").
            /// fp16 tensors: packed 2-per-uint; q8: packed 4-per-uint; i32: raw ints.
            /// Null until its upload completed.</summary>
            public ComputeBuffer Get(string name)
            {
                if (!_entries.TryGetValue(name, out Entry e))
                    throw new KeyNotFoundException($"Chatterbox manifest has no tensor '{name}'.");
                return e.slot[0];
            }

            public int[] Shape(string name) => _entries[name].shape;
            public bool Has(string name) => _entries.ContainsKey(name);

            /// <summary>Synchronous CPU read of a small tensor's file (conds: prompt tokens, speaker emb).
            /// i32 files -> int[]; f16 files -> float[] (widened). Only for tiny conds/bias data.</summary>
            public float[] ReadFloats(string name)
            {
                Entry e = _entries[name];
                byte[] raw = File.ReadAllBytes(Path.Combine(_root, e.file));
                float[] r = new float[e.numel];
                if (e.dtype == "i32")
                    for (int i = 0; i < e.numel; i++) r[i] = BitConverter.ToInt32(raw, i * 4);
                else
                    for (int i = 0; i < e.numel; i++) r[i] = Mathf.HalfToFloat(BitConverter.ToUInt16(raw, i * 2));
                return r;
            }

            public int[] ReadInts(string name)
            {
                Entry e = _entries[name];
                byte[] raw = File.ReadAllBytes(Path.Combine(_root, e.file));
                int[] r = new int[e.numel];
                for (int i = 0; i < e.numel; i++) r[i] = BitConverter.ToInt32(raw, i * 4);
                return r;
            }

            byte[] Rent(int size)
                => _pool.TryGetValue(size, out var stack) && stack.TryPop(out var arr) ? arr : new byte[size];

            void ReturnToPool(byte[] arr)
                => _pool.GetOrAdd(arr.Length, _ => new ConcurrentStack<byte[]>()).Push(arr);

            async Task LoadAllAsync()
            {
                try
                {
                    var tasks = new List<Task>(_entries.Count);
                    foreach (var kv in _entries)
                        tasks.Add(ReadOneAsync(kv.Value));
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

            async Task ReadOneAsync(Entry en)
            {
                await _ioGate.WaitAsync().ConfigureAwait(false);
                try
                {
                    string path = Path.Combine(_root, en.file);
                    int byteLen = en.numel * BytesPerElem(en.dtype);
                    byte[] data = Rent(byteLen);
                    using (var fs = new FileStream(path, FileMode.Open, FileAccess.Read, FileShare.Read,
                                                   64 * 1024, FileOptions.SequentialScan))
                    {
                        if (fs.Length != byteLen)
                            throw new IOException($"Bad size {fs.Length}, expected {byteLen} for {path}");
                        int off = 0;
                        while (off < byteLen)
                        {
                            int n = fs.Read(data, off, byteLen - off);
                            if (n <= 0) throw new IOException($"Unexpected EOF at {off}/{byteLen} in {path}");
                            off += n;
                        }
                    }
                    _uploads.Enqueue(new UploadJob { slot = en.slot, dtype = en.dtype, numel = en.numel, data = data });
                }
                catch
                {
                    _ioGate.Release();
                    throw;
                }
            }

            static ComputeBuffer MakeBuffer(string dtype, int numel)
            {
                if (dtype == "i32")
                    return new ComputeBuffer(numel, 4, ComputeBufferType.Structured);
                if (dtype == "q8")   // int8 packed 4-per-uint
                    return new ComputeBuffer((numel + 3) / 4, 4, ComputeBufferType.Structured);
                // fp16 packed 2-per-uint; odd counts round up (exporter pads file? no — file is exact,
                // last uint's upper half stays zero via a padded SetData below)
                return new ComputeBuffer((numel + 1) / 2, 4, ComputeBufferType.Structured);
            }

            IEnumerator UploadPump()
            {
                long budget = LLM.UploadBudgetBytes;

                while (true)
                {
                    if (_disposed) yield break;

                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        if (job.slot[0] == null)
                        {
                            if (budget <= 0) { yield return null; budget = LLM.UploadBudgetBytes; }
                            job.slot[0] = MakeBuffer(job.dtype, job.numel);
                            budget -= (long)job.slot[0].count * 4;
                        }
                        ComputeBuffer target = job.slot[0];

                        int byteLen = job.numel * BytesPerElem(job.dtype);
                        // fp16/q8 with a non-word-multiple numel: upload all full 4-byte words,
                        // then patch the tail word.
                        int even = byteLen & ~3;
                        int src = 0;
                        while (src < even)
                        {
                            if (_disposed) yield break;
                            if (budget <= 0)
                            {
                                yield return null;
                                budget = LLM.UploadBudgetBytes;
                            }
                            int count = (int)Math.Min(budget, even - src);
                            count &= ~3;
                            if (count == 0) count = 4;
                            target.SetData(job.data, src, src, count);
                            src += count;
                            budget -= count;
                        }
                        if (even < byteLen)   // 1-3 tail bytes: pad into a full zero-extended word
                        {
                            byte[] tail = new byte[4];
                            for (int i = 0; even + i < byteLen; i++) tail[i] = job.data[even + i];
                            target.SetData(tail, 0, even, 4);
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

                if (_jobsUploaded != _jobsTotal)
                    ConsoleMessage.Warning($"Chatterbox weights: only {_jobsUploaded}/{_jobsTotal} tensors uploaded " +
                                           "(missing or failed reads — see earlier exceptions). Output will be invalid.");
                else
                    ResidencyLog.Resident(ResidencyLog.Label(_root), 0,   // legacy loader: no byte tracking (WS-F)
                                          _loadWall.Elapsed.TotalMilliseconds,
                                          UnityEngine.Time.frameCount - _loadStartFrame);

                IsReady = true;
            }

            public void Dispose()
            {
                _disposed = true;
                foreach (var kv in _entries)
                    kv.Value.slot[0]?.Release();
            }
        }
    }
}
