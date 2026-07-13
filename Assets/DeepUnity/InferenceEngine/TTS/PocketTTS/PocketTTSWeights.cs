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
    namespace PocketTTSModeling
    {
        // Manifest-driven weight store for the Kyutai pocket-tts export (fp16/int8 .bin files +
        // manifest.tsv written by TTS/PocketTTS/validation/import_pocket_tts.py). Byte-for-byte the
        // same generic loader as CosyVoiceWeights (the exporter is the schema — no hardcoded tensor
        // list): implements the ModelBase residency contract loader-side (BeginLoad budgeted upload
        // pump, Defetch safe at any point via load-epoch, LoadBlocking for editor parity probes).
        // fp16 packed 2-per-uint (readH); q8 packed 4-per-uint (readQ8) + fp16 per-row `<name>.scales`.
        public class PocketTTSWeights : IDisposable
        {
            public bool IsReady { get; private set; }
            public ModelResidency Residency { get; private set; } = ModelResidency.Unloaded;

            public long BudgetBytesPerFrame = LLM.UploadBudgetBytes;
            public long BytesTotal { get; private set; }
            public long BytesUploaded { get; private set; }

            const int MAX_IO_JOBS = 4;

            struct Entry
            {
                public ComputeBuffer[] slot;
                public string dtype;
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
                public int epoch;
            }

            readonly Dictionary<string, Entry> _entries = new Dictionary<string, Entry>();
            readonly ConcurrentQueue<UploadJob> _uploads = new ConcurrentQueue<UploadJob>();
            readonly SemaphoreSlim _ioGate = new SemaphoreSlim(MAX_IO_JOBS);
            readonly ConcurrentDictionary<int, ConcurrentStack<byte[]>> _pool
                = new ConcurrentDictionary<int, ConcurrentStack<byte[]>>();
            readonly string _root;
            volatile bool _allReadsEnqueued;
            volatile bool _disposed;
            volatile int _epoch;
            int _jobsUploaded, _jobsTotal;

            public PocketTTSWeights(string paramsPath, bool beginLoad = true)
            {
                if (!Directory.Exists(paramsPath))
                    throw new DirectoryNotFoundException(
                        $"pocket-tts weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/InferenceEngine/TTS/PocketTTS/validation/import_pocket_tts.py " +
                        "(`python import_pocket_tts.py --quant fp16`) → exports under Assets/Resources/Weights/.");
                _root = paramsPath;

                string manifest = Path.Combine(paramsPath, "manifest.tsv");
                if (!File.Exists(manifest))
                    throw new FileNotFoundException($"manifest.tsv missing in {paramsPath} (re-run the exporter).");

                foreach (string line in File.ReadAllLines(manifest))
                {
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    string[] p = line.Split('\t');
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
                    BytesTotal += (long)int.Parse(p[3]) * BytesPerElem(p[2]);
                }
                _jobsTotal = _entries.Count;

                if (beginLoad) BeginLoad();
            }

            bool _reloadAfterDefetch;
            System.Diagnostics.Stopwatch _loadWall = new System.Diagnostics.Stopwatch();
            int _loadStartFrame;

            public void BeginLoad()
            {
                if (_disposed) return;
                if (Residency == ModelResidency.Defetching) { _reloadAfterDefetch = true; return; }
                if (Residency != ModelResidency.Unloaded) return;
                Residency = ModelResidency.Prefetching;
                _allReadsEnqueued = false;
                _jobsUploaded = 0;
                _loadWall.Restart();
                _loadStartFrame = UnityEngine.Time.frameCount;
                ResidencyLog.Loading(ResidencyLog.Label(_root), BytesTotal, BudgetBytesPerFrame);
                DeepUnityDispatcher.Run(UploadPump(_epoch));
                _ = Task.Run(() => LoadAllAsync(_epoch));
            }

            /// <summary>Synchronous unbudgeted load of tensors whose name starts with prefix (null =
            /// all). EDITOR/VALIDATION ONLY (parity probes run in one invoke, cannot pump frames).</summary>
            public void LoadBlocking(string prefix = null)
            {
                foreach (var kv in _entries)
                {
                    if (prefix != null && !kv.Key.StartsWith(prefix, StringComparison.Ordinal)) continue;
                    Entry e = kv.Value;
                    if (e.slot[0] != null) continue;
                    byte[] raw = File.ReadAllBytes(Path.Combine(_root, e.file));
                    int byteLen = e.numel * BytesPerElem(e.dtype);
                    if (raw.Length != byteLen)
                        throw new IOException($"Bad size {raw.Length}, expected {byteLen} for {e.file}");
                    e.slot[0] = MakeBuffer(e.dtype, e.numel);
                    int even = byteLen & ~3;
                    if (even > 0) e.slot[0].SetData(raw, 0, 0, even);
                    if (even < byteLen)
                    {
                        byte[] tail = new byte[4];
                        for (int i = 0; even + i < byteLen; i++) tail[i] = raw[even + i];
                        e.slot[0].SetData(tail, 0, even, 4);
                    }
                    BytesUploaded += byteLen;
                }
                if (prefix == null) { IsReady = true; Residency = ModelResidency.Ready; }
            }

            public void Defetch(long slowBytesPerFrame = 0)
            {
                if (_disposed || Residency == ModelResidency.Unloaded
                              || Residency == ModelResidency.Defetching) return;
                _epoch++;
                IsReady = false;
                Residency = ModelResidency.Defetching;
                ResidencyLog.Defetching(ResidencyLog.Label(_root), slowBytesPerFrame);
                DeepUnityDispatcher.Run(DefetchPump(slowBytesPerFrame));
            }

            IEnumerator DefetchPump(long slowBytesPerFrame)
            {
                while (_ioGate.CurrentCount < MAX_IO_JOBS || !_uploads.IsEmpty)
                {
                    while (_uploads.TryDequeue(out UploadJob j))
                    {
                        ReturnToPool(j.data);
                        _ioGate.Release();
                    }
                    yield return null;
                    if (_disposed) yield break;
                }

                long budget = slowBytesPerFrame;
                foreach (var kv in _entries)
                {
                    var slot = kv.Value.slot;
                    if (slot[0] == null) continue;
                    if (slowBytesPerFrame > 0)
                    {
                        budget -= (long)slot[0].count * 4;
                        if (budget <= 0)
                        {
                            yield return null;
                            if (_disposed) yield break;
                            budget = slowBytesPerFrame;
                        }
                    }
                    slot[0].Release();
                    slot[0] = null;
                }

                long freed = BytesUploaded;
                BytesUploaded = 0;
                Residency = ModelResidency.Unloaded;
                ResidencyLog.Released(ResidencyLog.Label(_root), freed);

                if (_reloadAfterDefetch && !_disposed)
                {
                    _reloadAfterDefetch = false;
                    BeginLoad();
                }
            }

            /// <summary>GPU buffer of an exported tensor (manifest name). Null until upload done.</summary>
            public ComputeBuffer Get(string name)
            {
                if (!_entries.TryGetValue(name, out Entry e))
                    throw new KeyNotFoundException($"pocket-tts manifest has no tensor '{name}'.");
                return e.slot[0];
            }

            public int[] Shape(string name) => _entries[name].shape;
            public bool Has(string name) => _entries.ContainsKey(name);

            /// <summary>Synchronous CPU read of a small tensor as fp32 (voice audio_prompt,
            /// emb_mean/std, and the CPU-side matmul weights input_linear/out_eos). i32 -> int
            /// widened; f16 -> float widened; q8 -> DEQUANTIZED (packed int8 4-per-uint × per-row
            /// fp16 scale from the sibling '<name>.scales'), so a CPU-read weight works the same
            /// whether the dir is fp16 or int8. Small tensors only.</summary>
            public float[] ReadFloats(string name)
            {
                Entry e = _entries[name];
                byte[] raw = File.ReadAllBytes(Path.Combine(_root, e.file));
                float[] r = new float[e.numel];
                if (e.dtype == "i32")
                {
                    for (int i = 0; i < e.numel; i++) r[i] = BitConverter.ToInt32(raw, i * 4);
                }
                else if (e.dtype == "q8")
                {
                    // per-output-row symmetric int8: element [o,i] = int8[o*cols+i] * scale[o].
                    // shape is [rows, cols]; scales sibling has one fp16 per row.
                    int rows = e.shape.Length >= 1 ? e.shape[0] : 1;
                    int cols = e.numel / Math.Max(rows, 1);
                    float[] scales = ReadFloats(name + ".scales");   // [rows] fp16 -> float
                    for (int o = 0; o < rows; o++)
                    {
                        float s = scales[o];
                        int b = o * cols;
                        for (int i = 0; i < cols; i++)
                            r[b + i] = (sbyte)raw[b + i] * s;   // int8.bin is raw signed bytes
                    }
                }
                else
                {
                    for (int i = 0; i < e.numel; i++) r[i] = Mathf.HalfToFloat(BitConverter.ToUInt16(raw, i * 2));
                }
                return r;
            }

            byte[] Rent(int size)
                => _pool.TryGetValue(size, out var stack) && stack.TryPop(out var arr) ? arr : new byte[size];

            void ReturnToPool(byte[] arr)
                => _pool.GetOrAdd(arr.Length, _ => new ConcurrentStack<byte[]>()).Push(arr);

            async Task LoadAllAsync(int epoch)
            {
                try
                {
                    var tasks = new List<Task>(_entries.Count);
                    foreach (var kv in _entries)
                        tasks.Add(ReadOneAsync(kv.Value, epoch));
                    await Task.WhenAll(tasks).ConfigureAwait(false);
                }
                catch (Exception e) { Debug.LogException(e); }
                finally { _allReadsEnqueued = true; }
            }

            async Task ReadOneAsync(Entry en, int epoch)
            {
                await _ioGate.WaitAsync().ConfigureAwait(false);
                try
                {
                    if (epoch != _epoch) { _ioGate.Release(); return; }
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
                    _uploads.Enqueue(new UploadJob { slot = en.slot, dtype = en.dtype,
                                                     numel = en.numel, data = data, epoch = epoch });
                }
                catch { _ioGate.Release(); throw; }
            }

            static ComputeBuffer MakeBuffer(string dtype, int numel)
            {
                if (dtype == "i32")
                    return new ComputeBuffer(numel, 4, ComputeBufferType.Structured);
                if (dtype == "q8")
                    return new ComputeBuffer((numel + 3) / 4, 4, ComputeBufferType.Structured);
                return new ComputeBuffer((numel + 1) / 2, 4, ComputeBufferType.Structured);
            }

            IEnumerator UploadPump(int epoch)
            {
                long budget = BudgetBytesPerFrame;
                while (true)
                {
                    if (_disposed || epoch != _epoch) yield break;

                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        if (job.epoch != _epoch)
                        {
                            ReturnToPool(job.data);
                            _ioGate.Release();
                            continue;
                        }
                        bool abort = false;

                        if (job.slot[0] == null)
                        {
                            while (budget <= 0)
                            {
                                yield return null;
                                if (_disposed || epoch != _epoch) { abort = true; break; }
                                budget = BudgetBytesPerFrame;
                            }
                            if (!abort)
                            {
                                job.slot[0] = MakeBuffer(job.dtype, job.numel);
                                budget -= (long)job.slot[0].count * 4;
                            }
                        }

                        int byteLen = job.numel * BytesPerElem(job.dtype);
                        int even = byteLen & ~3;
                        int src = 0;
                        while (!abort && src < even)
                        {
                            if (_disposed || epoch != _epoch) { abort = true; break; }
                            while (budget <= 0)
                            {
                                yield return null;
                                if (_disposed || epoch != _epoch) { abort = true; break; }
                                budget = BudgetBytesPerFrame;
                            }
                            if (abort) break;
                            int count = (int)Math.Min(budget, even - src);
                            count &= ~3;
                            if (count == 0) count = 4;
                            job.slot[0].SetData(job.data, src, src, count);
                            src += count;
                            budget -= count;
                            BytesUploaded += count;
                        }
                        if (!abort && even < byteLen)
                        {
                            byte[] tail = new byte[4];
                            for (int i = 0; even + i < byteLen; i++) tail[i] = job.data[even + i];
                            job.slot[0].SetData(tail, 0, even, 4);
                            BytesUploaded += byteLen - even;
                        }

                        ReturnToPool(job.data);
                        _ioGate.Release();
                        if (abort) yield break;
                        _jobsUploaded++;
                    }
                    else if (_allReadsEnqueued && _uploads.IsEmpty) { break; }
                    else { yield return null; budget = BudgetBytesPerFrame; }
                }

                _pool.Clear();
                if (_jobsUploaded != _jobsTotal)
                    ConsoleMessage.Warning($"pocket-tts weights: only {_jobsUploaded}/{_jobsTotal} tensors uploaded.");
                else
                    ResidencyLog.Resident(ResidencyLog.Label(_root), BytesTotal,
                                          _loadWall.Elapsed.TotalMilliseconds,
                                          UnityEngine.Time.frameCount - _loadStartFrame);
                IsReady = true;
                Residency = ModelResidency.Ready;
            }

            public void Dispose()
            {
                _disposed = true;
                foreach (var kv in _entries)
                {
                    kv.Value.slot[0]?.Release();
                    kv.Value.slot[0] = null;
                }
            }
        }
    }
}
