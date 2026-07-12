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
    namespace CosyVoiceModeling
    {
        // Manifest-driven weight store for the Fun-CosyVoice3-0.5B export (fp16/int8 .bin files +
        // manifest.tsv written by import_params.py cosyvoice3-0.5b).
        //
        // Implements the ModelBase RESIDENCY contract loader-side:
        //   - BeginLoad(): background reader tasks -> pooled byte[] -> main-thread UploadPump
        //     that never exceeds BudgetBytesPerFrame of SetData work per frame (hitch-free boot).
        //     The budget is LIVE — SlowPrefetch/Boost/Pause take effect next frame.
        //   - Defetch(slowBytesPerFrame): SAFE AT ANY POINT, incl. mid-load: bumps the load epoch
        //     (stale IO results are discarded), drains queued uploads back to the pool, then
        //     releases resident GPU buffers instantly (0) or budgeted across frames (>0).
        //     Afterwards the store is Unloaded and BeginLoad() can start fresh.
        //
        // Generic: every tensor is a manifest line `name\tfile\tdtype\tnumel\tshape`, exposed via
        // Get(name). fp16 packed 2-per-uint (readH); q8 packed 4-per-uint (readQ8) + fp16 per-row
        // `<name>.scales`; i32 raw ints. No hardcoded tensor list — the exporter is the schema.
        public class CosyVoiceWeights : IDisposable
        {
            public bool IsReady { get; private set; }
            public ModelResidency Residency { get; private set; } = ModelResidency.Unloaded;

            /// <summary>Per-frame SetData budget — LIVE-adjustable (ModelBase forwards here).</summary>
            public long BudgetBytesPerFrame = LLM.UploadBudgetBytes;
            /// <summary>Total/resident weight bytes for ModelBase.LoadProgress.</summary>
            public long BytesTotal { get; private set; }
            public long BytesUploaded { get; private set; }

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
                public int epoch;              // stale jobs (from a defetched load) are discarded
            }

            readonly Dictionary<string, Entry> _entries = new Dictionary<string, Entry>();
            readonly ConcurrentQueue<UploadJob> _uploads = new ConcurrentQueue<UploadJob>();
            readonly SemaphoreSlim _ioGate = new SemaphoreSlim(MAX_IO_JOBS);
            readonly ConcurrentDictionary<int, ConcurrentStack<byte[]>> _pool
                = new ConcurrentDictionary<int, ConcurrentStack<byte[]>>();
            readonly string _root;
            volatile bool _allReadsEnqueued;
            volatile bool _disposed;
            volatile int _epoch;               // incremented by Defetch to invalidate in-flight IO
            int _jobsUploaded, _jobsTotal;

            public CosyVoiceWeights(string paramsPath, bool beginLoad = true)
            {
                if (!Directory.Exists(paramsPath))
                    throw new DirectoryNotFoundException(
                        $"CosyVoice weights folder not found: '{paramsPath}'. Generate it with " +
                        "Assets/DeepUnity/LLM/import_params.py — `python import_params.py cosyvoice3-0.5b` " +
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
                    BytesTotal += (long)int.Parse(p[3]) * BytesPerElem(p[2]);
                }
                _jobsTotal = _entries.Count;

                if (beginLoad) BeginLoad();
            }

            bool _reloadAfterDefetch;

            /// <summary>Start (or restart after a Defetch) the budgeted background load.
            /// No-op while Prefetching/Ready. Called DURING a defetch, the request is remembered
            /// and the load restarts automatically when the defetch completes (never lost).</summary>
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
                // kicked on the THREAD POOL: an async method's synchronous prefix (task fan-out + the
                // first MAX_IO_JOBS reads, which pass the io-gate synchronously) otherwise runs the
                // first file reads on the MAIN thread — measured 80-280 ms = the zone-entry freeze.
                _ = Task.Run(() => LoadAllAsync(_epoch));
            }

            /// <summary>Synchronous, unbudgeted load of every tensor whose manifest name starts
            /// with <paramref name="prefix"/> (null = all). EDITOR/VALIDATION ONLY: parity probes
            /// run inside a single editor invoke and cannot pump frames — games use BeginLoad().
            /// Idempotent per tensor; a later BeginLoad() skips already-resident slots.</summary>
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

            /// <summary>Safe unload at any point (incl. mid-load). slowBytesPerFrame &lt;= 0 →
            /// instant release; &gt; 0 → buffers released across frames under that budget
            /// (the anti-frame-drop mirror of the loader).</summary>
            public void Defetch(long slowBytesPerFrame = 0)
            {
                if (_disposed || Residency == ModelResidency.Unloaded
                              || Residency == ModelResidency.Defetching) return;
                _epoch++;                      // in-flight reads become stale
                IsReady = false;
                Residency = ModelResidency.Defetching;
                ResidencyLog.Defetching(ResidencyLog.Label(_root), slowBytesPerFrame);
                DeepUnityDispatcher.Run(DefetchPump(slowBytesPerFrame));
            }

            IEnumerator DefetchPump(long slowBytesPerFrame)
            {
                // 1. drain: wait out in-flight readers, discarding their (stale) jobs as they land
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

                // 2. release resident buffers — instantly or budgeted per frame
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

                // a prefetch was requested while we were draining ("player came right back"):
                if (_reloadAfterDefetch && !_disposed)
                {
                    _reloadAfterDefetch = false;
                    BeginLoad();
                }
            }

            /// <summary>GPU buffer of an exported tensor (manifest name, e.g.
            /// "llm/layers.0.self_attn.q_proj.weight"). Null until its upload completed.</summary>
            public ComputeBuffer Get(string name)
            {
                if (!_entries.TryGetValue(name, out Entry e))
                    throw new KeyNotFoundException($"CosyVoice manifest has no tensor '{name}'.");
                return e.slot[0];
            }

            public int[] Shape(string name) => _entries[name].shape;
            public bool Has(string name) => _entries.ContainsKey(name);

            /// <summary>Synchronous CPU read of a small tensor's file (voice conds: prompt tokens,
            /// campplus embedding). i32 -> int[]; f16 -> float[] (widened). Tiny tensors only.</summary>
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

            async Task LoadAllAsync(int epoch)
            {
                try
                {
                    var tasks = new List<Task>(_entries.Count);
                    foreach (var kv in _entries)
                        tasks.Add(ReadOneAsync(kv.Value, epoch));
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

            async Task ReadOneAsync(Entry en, int epoch)
            {
                await _ioGate.WaitAsync().ConfigureAwait(false);
                try
                {
                    if (epoch != _epoch) { _ioGate.Release(); return; }   // defetched meanwhile
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
                // fp16 packed 2-per-uint; odd numel -> tail word zero-extended in UploadPump
                return new ComputeBuffer((numel + 1) / 2, 4, ComputeBufferType.Structured);
            }

            IEnumerator UploadPump(int epoch)
            {
                long budget = BudgetBytesPerFrame;

                while (true)
                {
                    if (_disposed || epoch != _epoch) yield break;   // defetched: DefetchPump owns cleanup

                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        if (job.epoch != _epoch)   // stale job from a defetched load
                        {
                            ReturnToPool(job.data);
                            _ioGate.Release();
                            continue;
                        }
                        // From here the job's byte[] and IO slot are OWED cleanup on EVERY exit
                        // path — otherwise a mid-job Defetch would strand the gate slot and the
                        // DefetchPump's drain would wait forever.
                        bool abort = false;

                        if (job.slot[0] == null)
                        {
                            while (budget <= 0)   // hard pause: budget 0 uploads NOTHING
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
                        // fp16/q8 with a non-word-multiple numel: upload all full 4-byte words,
                        // then patch the tail word.
                        int even = byteLen & ~3;
                        int src = 0;
                        while (!abort && src < even)
                        {
                            if (_disposed || epoch != _epoch) { abort = true; break; }
                            while (budget <= 0)                  // budget re-sampled each frame:
                            {                                    // SlowPrefetch/BoostFetch apply next
                                yield return null;               // frame; 0 = HARD pause (no bytes)
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
                        if (!abort && even < byteLen)   // 1-3 tail bytes: pad into a zero-extended word
                        {
                            byte[] tail = new byte[4];
                            for (int i = 0; even + i < byteLen; i++) tail[i] = job.data[even + i];
                            job.slot[0].SetData(tail, 0, even, 4);
                            BytesUploaded += byteLen - even;
                        }

                        ReturnToPool(job.data);   // cleanup happens on success AND abort paths
                        _ioGate.Release();        // (partially-filled buffers are released by the
                        if (abort) yield break;   //  DefetchPump / Dispose, which own the slots)
                        _jobsUploaded++;
                    }
                    else if (_allReadsEnqueued && _uploads.IsEmpty)
                    {
                        break;
                    }
                    else
                    {
                        yield return null;
                        budget = BudgetBytesPerFrame;
                    }
                }

                _pool.Clear();

                if (_jobsUploaded != _jobsTotal)
                    ConsoleMessage.Warning($"CosyVoice weights: only {_jobsUploaded}/{_jobsTotal} tensors uploaded " +
                                           "(missing or failed reads — see earlier exceptions). Output will be invalid.");
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
