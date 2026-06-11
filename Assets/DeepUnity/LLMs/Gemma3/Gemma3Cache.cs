using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace DeepUnity
{
    namespace Gemma3Modeling
    {
        public class Gemma3Cache : IDisposable
        {
            public ComputeBuffer[] kCaches;
            public ComputeBuffer[] vCaches;
            public int CachedTokenCount { get; set; }

            readonly int numLayers;
            readonly int headsKV;
            readonly int headDim;
            readonly int capacity;

            const int FILE_MAGIC = 0x47334B56; // "G3KV"
            const int FILE_VERSION = 1;

            public Gemma3Cache(int numLayers, int capacity, int headsKV, int headDim)
            {
                this.numLayers = numLayers;
                this.headsKV = headsKV;
                this.headDim = headDim;
                this.capacity = capacity;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];

                // FP32 activations — stride 4, full count
                int bufSize = capacity * headsKV * headDim;
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i] = new ComputeBuffer(bufSize, 4, ComputeBufferType.Structured);
                    vCaches[i] = new ComputeBuffer(bufSize, 4, ComputeBufferType.Structured);
                }

                CachedTokenCount = 0;
            }

            public void Reset() => CachedTokenCount = 0;

            // Persist the populated slice of K/V caches to a folder.
            // Layout:
            //   {folder}/meta.bin                  -- header (magic, version, dims, tokenCount)
            //   {folder}/k_cache_layer_{i}.bin     -- packed FP16, perLayerFloats halves
            //   {folder}/v_cache_layer_{i}.bin     -- packed FP16, perLayerFloats halves
            //
            // Coroutine, frame-budgeted like Qwen3_5Cache (shared knobs in Base/LLM.cs): a
            // sliding window of LLM.SaveReadbacksInFlight async readbacks (results must be
            // copied the frame they complete, so the window bounds per-frame copy work), then
            // the FP32→FP16 packing + all file writes happen on a worker thread.
            public IEnumerator SaveAsync(string folder)
            {
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
                            ConsoleMessage.Info($"Gemma3 cache save: readback error on layer {r / 2}.");
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
                    ConsoleMessage.Warning("Gemma3 cache save failed: " + task.Exception?.GetBaseException().Message);
            }

            // Attempt to load a previously persisted KV cache from a folder.
            // Coroutine, frame-budgeted like Qwen3_5Cache (shared knobs in Base/LLM.cs): file
            // IO + FP16→FP32 conversion run on a worker thread, then SetData uploads are
            // chunked under LLM.UploadFrameBudgetMs / LLM.UploadChunkFloats so no frame
            // hitches. Uploads only start after the whole file set validates. Result delivered
            // via onComplete (true = loaded, false = missing/mismatch/IO error).
            public IEnumerator TryLoadAsync(string folder, Action<bool> onComplete)
            {
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

            // FP16 converters usable off the main thread (Mathf.FloatToHalf/HalfToFloat are engine
            // calls, kept off the worker). Save truncates the mantissa and flushes denormals to 0
            // (< 6e-5 — noise for KV states); load treats half-subnormals as 0 for the same reason.
            // On-disk format is unchanged, so caches written by the old Mathf path still load.
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
                }
            }
        }
    }
}
