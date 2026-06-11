using System;
using System.Collections;
using System.IO;
using System.Threading.Tasks;
using UnityEngine;
using UnityEngine.Rendering;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // Hybrid cache:
        //   - Full-attention layers store standard K/V buffers (per token, per kv-head, per head_dim).
        //   - Linear-attention (Gated DeltaNet) layers store SSM state:
        //         conv_state      [conv_dim * (kernel_size - 1)]   FP32
        //         recurrent_state [num_v_heads * head_k_dim * head_v_dim] FP32
        //
        // v2: SaveYielding/LoadYielding persist the current prefix state (the system prompt) to disk,
        // so re-initializing with the same prompt restores the cache instead of recomputing prefill.
        // K/V layout is token-major ((pos * heads_kv + h) * head_dim + d), so the first
        // CachedTokenCount * floatsPerToken floats are exactly the prefix — partial save is valid.
        public class Qwen3_5Cache : IDisposable
        {
            public ComputeBuffer[] kCaches; // length numLayers; null on linear layers
            public ComputeBuffer[] vCaches; // length numLayers; null on linear layers

            public ComputeBuffer[] convStates;      // length numLayers; null on full layers
            public ComputeBuffer[] recurrentStates; // length numLayers; null on full layers

            public int CachedTokenCount { get; set; }
            public int Capacity => capacity;

            readonly int numLayers;
            readonly int capacity;

            const uint FILE_MAGIC = 0x51354B56;   // "Q5KV"
            const int FILE_VERSION = 1;

            // Hitch tuning (frame budget, chunk size, readbacks in flight) is shared across all
            // models — the knobs live in Base/LLM.cs: LLM.UploadFrameBudgetMs,
            // LLM.UploadChunkFloats, LLM.SaveReadbacksInFlight.

            public Qwen3_5Cache(
                int capacity,
                Qwen3_5LayerType[] layerTypes,
                int headsKV, int headDim,
                int convDim, int convKernelSize,
                int numVHeads, int headKDim, int headVDim)
            {
                this.numLayers = layerTypes.Length;
                this.capacity = capacity;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];
                convStates = new ComputeBuffer[numLayers];
                recurrentStates = new ComputeBuffer[numLayers];

                int kvFloats = capacity * headsKV * headDim;
                int convFloats = convDim * (convKernelSize - 1);
                int recFloats = numVHeads * headKDim * headVDim;

                for (int i = 0; i < numLayers; i++)
                {
                    if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                    {
                        kCaches[i] = new ComputeBuffer(kvFloats, 4, ComputeBufferType.Structured);
                        vCaches[i] = new ComputeBuffer(kvFloats, 4, ComputeBufferType.Structured);
                    }
                    else // LinearAttention
                    {
                        convStates[i]      = new ComputeBuffer(convFloats, 4, ComputeBufferType.Structured);
                        recurrentStates[i] = new ComputeBuffer(recFloats, 4, ComputeBufferType.Structured);
                    }
                }

                CachedTokenCount = 0;
            }

            // Resets the logical token count only. The SSM state zero-fill is done GPU-side by
            // Qwen3_5Model.ResetCache (ZeroBuffer kernel) — the old CPU path allocated ~19 MB of
            // managed zero arrays and SetData'd them on the main thread on every reset.
            public void Reset()
            {
                CachedTokenCount = 0;
            }

            public void Dispose()
            {
                for (int i = 0; i < numLayers; i++)
                {
                    kCaches[i]?.Release();
                    vCaches[i]?.Release();
                    convStates[i]?.Release();
                    recurrentStates[i]?.Release();
                }
            }

            // ------------------------------------------------------------------ disk persistence

            /// <summary>
            /// Writes the current prefix state (CachedTokenCount tokens of K/V + the full SSM states)
            /// to disk. GPU reads go through AsyncGPUReadback (no pipeline stall, at most
            /// LLM.SaveReadbacksInFlight pending) and the file write happens on a worker thread —
            /// safe to run during gameplay.
            /// </summary>
            public IEnumerator SaveYielding(string path)
            {
                int tokens = CachedTokenCount;
                if (tokens <= 0) yield break;

                // sliding window of readbacks: at most LLM.SaveReadbacksInFlight pending at once
                // (K/V only need the first `tokens` rows). Each result is copied to managed
                // memory the same frame it completes — readback data doesn't survive past its
                // frame — so the window size also bounds per-frame copy work.
                int total = numLayers * 2;
                var blobs = new byte[total][];
                var reqs = new AsyncGPUReadbackRequest[total];
                int nextToIssue = 0, doneCount = 0, inFlight = 0;
                while (doneCount < total)
                {
                    while (inFlight < LLM.SaveReadbacksInFlight && nextToIssue < total)
                    {
                        int i = nextToIssue / 2;
                        bool firstHalf = (nextToIssue & 1) == 0;
                        if (kCaches[i] != null)
                        {
                            int bytes = tokens * (kCaches[i].count / capacity) * 4;
                            reqs[nextToIssue] = AsyncGPUReadback.Request(firstHalf ? kCaches[i] : vCaches[i], bytes, 0);
                        }
                        else
                            reqs[nextToIssue] = AsyncGPUReadback.Request(firstHalf ? convStates[i] : recurrentStates[i]);
                        nextToIssue++; inFlight++;
                    }
                    for (int r = 0; r < nextToIssue; r++)
                    {
                        if (blobs[r] != null) continue;
                        if (reqs[r].hasError)
                        {
                            ConsoleMessage.Warning("Qwen3.5 prompt-cache save aborted: GPU readback error");
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
                var kinds = new byte[n];
                for (int i = 0; i < n; i++) kinds[i] = (byte)(kCaches[i] != null ? 0 : 1);

                var task = Task.Run(() =>
                {
                    using var bw = new BinaryWriter(File.Create(path));
                    bw.Write(FILE_MAGIC);
                    bw.Write(FILE_VERSION);
                    bw.Write(tokens);
                    bw.Write(n);
                    for (int i = 0; i < n; i++)
                    {
                        bw.Write(kinds[i]);
                        bw.Write(blobs[i * 2].Length); bw.Write(blobs[i * 2]);
                        bw.Write(blobs[i * 2 + 1].Length); bw.Write(blobs[i * 2 + 1]);
                    }
                });
                while (!task.IsCompleted) yield return null;
                if (task.IsFaulted)
                    ConsoleMessage.Warning("Qwen3.5 prompt-cache save failed: " + task.Exception?.GetBaseException().Message);
            }

            /// <summary>
            /// Restores a prefix state written by SaveYielding. File IO + parsing run on a worker
            /// thread; GPU uploads are chunked under a per-frame time budget (LLM.UploadFrameBudgetMs /
            /// LLM.UploadChunkFloats) so no single frame hitches. On success
            /// CachedTokenCount is set and onLoaded(true) fires; any mismatch reports false (caller
            /// falls back to recomputing the prompt).
            /// </summary>
            public IEnumerator LoadYielding(string path, Action<bool> onLoaded)
            {
                int n = numLayers;
                var first = new float[n][];    // K or conv
                var second = new float[n][];   // V or recurrent
                int tokens = 0;
                string error = null;

                var task = Task.Run(() =>
                {
                    using var br = new BinaryReader(File.OpenRead(path));
                    if (br.ReadUInt32() != FILE_MAGIC || br.ReadInt32() != FILE_VERSION) { error = "bad header"; return; }
                    tokens = br.ReadInt32();
                    if (br.ReadInt32() != n) { error = "layer count mismatch"; return; }
                    byte[] scratch = null;   // reused across reads — halves the transient garbage
                    for (int i = 0; i < n; i++)
                    {
                        br.ReadByte();   // kind — re-validated against live buffers below
                        for (int half = 0; half < 2; half++)
                        {
                            int len = br.ReadInt32();
                            if (len < 0) { error = "truncated file"; return; }
                            if (scratch == null || scratch.Length < len) scratch = new byte[len];
                            int read = 0;
                            while (read < len)
                            {
                                int got = br.Read(scratch, read, len - read);
                                if (got <= 0) { error = "truncated file"; return; }
                                read += got;
                            }
                            var f = new float[len / 4];
                            Buffer.BlockCopy(scratch, 0, f, 0, len);
                            if (half == 0) first[i] = f; else second[i] = f;
                        }
                    }
                });
                while (!task.IsCompleted) yield return null;
                if (task.IsFaulted) error = task.Exception?.GetBaseException().Message;

                // validate shapes against the live cache before touching the GPU
                if (error == null && (tokens <= 0 || tokens > capacity)) error = "token count out of range";
                if (error == null)
                    for (int i = 0; i < n && error == null; i++)
                    {
                        if (kCaches[i] != null)
                        {
                            int expected = tokens * (kCaches[i].count / capacity);
                            if (first[i].Length != expected || second[i].Length != expected) error = "k/v size mismatch";
                        }
                        else if (first[i].Length != convStates[i].count || second[i].Length != recurrentStates[i].count)
                            error = "ssm state size mismatch";
                    }
                if (error != null)
                {
                    ConsoleMessage.Warning($"Qwen3.5 prompt-cache load failed ({error}) — recomputing the prompt");
                    onLoaded?.Invoke(false);
                    yield break;
                }

                // budgeted upload: SetData in LLM.UploadChunkFloats-sized pieces, yielding once
                // LLM.UploadFrameBudgetMs of main-thread copy time is spent in a frame. (The old
                // one-LAYER-per-frame upload pushed K+V of a full layer — several MB — in a
                // single frame and dropped play mode to ~48 fps.)
                var budget = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < n; i++)
                {
                    var a = kCaches[i] != null ? kCaches[i] : convStates[i];
                    var b = kCaches[i] != null ? vCaches[i] : recurrentStates[i];
                    var up = UploadChunked(a, first[i], budget);
                    while (up.MoveNext()) yield return up.Current;
                    up = UploadChunked(b, second[i], budget);
                    while (up.MoveNext()) yield return up.Current;
                }

                CachedTokenCount = tokens;
                onLoaded?.Invoke(true);
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
        }
    }
}
