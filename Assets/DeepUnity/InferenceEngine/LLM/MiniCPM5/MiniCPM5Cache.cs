using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace DeepUnity
{
    namespace MiniCPM5Modeling
    {
        // KV cache for MiniCPM5 — every layer is full attention (no sliding window, no recurrent
        // state), so this is a straight clone of Gemma3Cache with its own on-disk magic.
        public class MiniCPM5Cache : IDisposable
        {
            public ComputeBuffer[] kCaches;
            public ComputeBuffer[] vCaches;
            // INT8 KV only: per-(token, kv-head) fp16 scale + fp16 zero-point for K and V (asymmetric).
            // null unless KV == INT8. Laid out [capacity, headsKV].
            public ComputeBuffer[] kScaleZp;
            public ComputeBuffer[] vScaleZp;
            public int CachedTokenCount { get; set; }

            public readonly KVQuant KV;

            readonly int numLayers;
            readonly int headsKV;
            readonly int headDim;
            readonly int capacity;

            const int FILE_MAGIC = 0x4D354B56; // "M5KV"
            const int FILE_VERSION = 1;

            public MiniCPM5Cache(int numLayers, int capacity, int headsKV, int headDim, KVQuant kv = KVQuant.FP32)
            {
                this.numLayers = numLayers;
                this.headsKV = headsKV;
                this.headDim = headDim;
                this.capacity = capacity;
                this.KV = kv;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];

                // Element count of one K (or V) cache across `capacity` tokens. Storage width depends
                // on KV precision (all buffers are stride-4 uint; the count is what shrinks):
                //   FP32 -> 4 B/elem            -> count = elems
                //   FP16 -> 2 B/elem (2 / uint) -> count = elems / 2   (head_dim even, so exact)
                //   INT8 -> 1 B/elem (4 / uint) -> count = elems / 4   + per-(token,head) scale+zp
                int elems = capacity * headsKV * headDim;
                int uintCount = KVQuantUtil.UIntCount(elems, kv);
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i] = new ComputeBuffer(uintCount, 4, ComputeBufferType.Structured);
                    vCaches[i] = new ComputeBuffer(uintCount, 4, ComputeBufferType.Structured);
                }

                if (kv == KVQuant.INT8)
                {
                    // one fp16 scale + one fp16 zero-point per (token, kv-head) → 2 halves = 1 uint each
                    kScaleZp = new ComputeBuffer[numLayers];
                    vScaleZp = new ComputeBuffer[numLayers];
                    int szCount = capacity * headsKV;   // uints (scale|zp packed 2 halves per uint)
                    for (int i = 0; i < numLayers; i++)
                    {
                        kScaleZp[i] = new ComputeBuffer(szCount, 4, ComputeBufferType.Structured);
                        vScaleZp[i] = new ComputeBuffer(szCount, 4, ComputeBufferType.Structured);
                    }
                }

                CachedTokenCount = 0;
            }

            public void Reset() => CachedTokenCount = 0;

            // Persist the populated slice of K/V caches to a folder (same design as Gemma3Cache —
            // frame-budgeted async readbacks, then FP32→FP16 packing + writes on a worker thread).
            public IEnumerator SaveAsync(string folder)
            {
                // Disk prompt-cache currently supports FP32 KV only (readback + on-disk format
                // assume 4-byte floats). For quantized KV it's skipped — the prompt is recomputed.
                if (KV != KVQuant.FP32) yield break;
                int tokens = CachedTokenCount;
                if (tokens <= 0) yield break;

                int perLayerFloats = tokens * headsKV * headDim;
                int perLayerBytes = perLayerFloats * 4;

                int total = numLayers * 2;
                var blobs = new byte[total][];   // raw FP32 bytes per buffer (even = K, odd = V)
                var reqs = new AsyncGPUReadbackRequest[total];
                int nextToIssue = 0, doneCount = 0, inFlight = 0;
                while (doneCount < total)
                {
                    while (inFlight < LLM.SaveReadbacksInFlight && nextToIssue < total)
                    {
                        var buf = (nextToIssue & 1) == 0 ? kCaches[nextToIssue / 2] : vCaches[nextToIssue / 2];
                        reqs[nextToIssue] = AsyncGPUReadback.Request(buf, perLayerBytes, 0);
                        nextToIssue++; inFlight++;
                    }
                    for (int r = 0; r < nextToIssue; r++)
                    {
                        if (blobs[r] != null) continue;
                        if (reqs[r].hasError)
                        {
                            ConsoleMessage.Info($"MiniCPM5 cache save: readback error on layer {r / 2}.");
                            yield break;
                        }
                        if (reqs[r].done)
                        {
                            blobs[r] = reqs[r].GetData<byte>().ToArray();
                            doneCount++; inFlight--;
                        }
                    }
                    if (doneCount < total) yield return null;
                }

                int n = numLayers;
                var task = Task.Run(() =>
                {
                    Directory.CreateDirectory(folder);
                    using (var bw = new BinaryWriter(File.Create(Path.Combine(folder, "meta.bin"))))
                    {
                        bw.Write(FILE_MAGIC);
                        bw.Write(FILE_VERSION);
                        bw.Write(n);
                        bw.Write(headsKV);
                        bw.Write(headDim);
                        bw.Write(tokens);
                    }
                    var f = new float[perLayerFloats];          // scratch, reused across files
                    var half = new byte[perLayerFloats * 2];
                    for (int r = 0; r < total; r++)
                    {
                        Buffer.BlockCopy(blobs[r], 0, f, 0, perLayerBytes);
                        for (int j = 0; j < perLayerFloats; j++)
                        {
                            ushort h = FloatToHalfBits(f[j]);
                            half[j * 2] = (byte)h;
                            half[j * 2 + 1] = (byte)(h >> 8);
                        }
                        string name = ((r & 1) == 0 ? "k" : "v") + $"_cache_layer_{r / 2}.bin";
                        File.WriteAllBytes(Path.Combine(folder, name), half);
                    }
                });
                while (!task.IsCompleted) yield return null;
                if (task.IsFaulted)
                    ConsoleMessage.Warning("MiniCPM5 cache save failed: " + task.Exception?.GetBaseException().Message);
            }

            // Attempt to load a previously persisted KV cache from a folder (frame-budgeted; see
            // Gemma3Cache.TryLoadAsync for the design notes). onComplete(true) = loaded.
            public IEnumerator TryLoadAsync(string folder, Action<bool> onComplete)
            {
                if (KV != KVQuant.FP32) { onComplete?.Invoke(false); yield break; }   // FP32-only for now
                string metaPath = Path.Combine(folder, "meta.bin");
                if (!File.Exists(metaPath)) { onComplete?.Invoke(false); yield break; }

                int n = numLayers;
                var kData = new float[n][];
                var vData = new float[n][];
                int fTokens = 0;
                bool ok = false;

                var task = Task.Run(() =>
                {
                    using (var br = new BinaryReader(File.OpenRead(metaPath)))
                    {
                        if (br.BaseStream.Length < 24) return;
                        if (br.ReadInt32() != FILE_MAGIC || br.ReadInt32() != FILE_VERSION) return;
                        if (br.ReadInt32() != n || br.ReadInt32() != headsKV || br.ReadInt32() != headDim) return;
                        fTokens = br.ReadInt32();
                    }
                    if (fTokens <= 0 || fTokens > capacity) return;

                    int perLayerFloats = fTokens * headsKV * headDim;
                    int expectedBytes = perLayerFloats * 2;
                    var scratchH = new ushort[perLayerFloats];   // reused across files
                    for (int i = 0; i < n; i++)
                    {
                        for (int kv = 0; kv < 2; kv++)
                        {
                            string p = Path.Combine(folder, (kv == 0 ? "k" : "v") + $"_cache_layer_{i}.bin");
                            if (!File.Exists(p)) return;
                            byte[] raw = File.ReadAllBytes(p);
                            if (raw.Length != expectedBytes) return;
                            Buffer.BlockCopy(raw, 0, scratchH, 0, expectedBytes);
                            var f = new float[perLayerFloats];
                            for (int j = 0; j < perLayerFloats; j++) f[j] = HalfBitsToFloat(scratchH[j]);
                            if (kv == 0) kData[i] = f; else vData[i] = f;
                        }
                    }
                    ok = true;
                });
                while (!task.IsCompleted) yield return null;
                if (task.IsFaulted || !ok) { onComplete?.Invoke(false); yield break; }

                var budget = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < n; i++)
                {
                    var up = UploadChunked(kCaches[i], kData[i], budget);
                    while (up.MoveNext()) yield return up.Current;
                    up = UploadChunked(vCaches[i], vData[i], budget);
                    while (up.MoveNext()) yield return up.Current;
                }

                CachedTokenCount = fTokens;
                onComplete?.Invoke(true);
            }

            // Uploads `data` into `buf` in LLM.UploadChunkFloats-sized SetData calls, yielding a
            // frame whenever the shared budget stopwatch crosses LLM.UploadFrameBudgetMs.
            IEnumerator UploadChunked(ComputeBuffer buf, float[] data, System.Diagnostics.Stopwatch budget)
            {
                int offset = 0;
                while (offset < data.Length)
                {
                    if (budget.Elapsed.TotalMilliseconds >= LLM.UploadFrameBudgetMs)
                    {
                        yield return null;
                        budget.Restart();
                    }
                    int count = Math.Min(LLM.UploadChunkFloats, data.Length - offset);
                    buf.SetData(data, offset, offset, count);
                    offset += count;
                }
            }

            // FP16 converters usable off the main thread (see Gemma3Cache for the rationale).
            static ushort FloatToHalfBits(float value)
            {
                int f = BitConverter.SingleToInt32Bits(value);
                int sign = (f >> 16) & 0x8000;
                int exp = ((f >> 23) & 0xff) - 127 + 15;
                if (exp <= 0) return (ushort)sign;                      // underflow → ±0
                if (exp >= 31) return (ushort)(sign | 0x7c00);          // overflow/inf/nan → ±inf
                return (ushort)(sign | (exp << 10) | ((f & 0x7fffff) >> 13));
            }

            static float HalfBitsToFloat(ushort h)
            {
                int sign = (h & 0x8000) << 16;
                int exp = (h >> 10) & 0x1f;
                int mant = h & 0x3ff;
                if (exp == 0) return BitConverter.Int32BitsToSingle(sign);   // ±0 / subnormal → ±0
                if (exp == 31) return BitConverter.Int32BitsToSingle(sign | 0x7f800000 | (mant << 13));
                return BitConverter.Int32BitsToSingle(sign | ((exp + 112) << 23) | (mant << 13));
            }

            public void Dispose()
            {
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i]?.Release();
                    vCaches[i]?.Release();
                    kScaleZp?[i]?.Release();
                    vScaleZp?[i]?.Release();
                }
            }
        }
    }
}
