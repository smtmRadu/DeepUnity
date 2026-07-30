using System;
using System.Collections;
using System.Collections.Generic;
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
        // SaveYielding/LoadYielding persist the current prefix state to disk (system prompt OR a
        // whole conversation — they are prefix-agnostic), so re-initializing with the same context
        // restores the cache instead of recomputing prefill. K/V layout is token-major
        // ((pos * heads_kv + h) * head_dim + d) and head_dim is a multiple of 4, so at ANY packing
        // (FP32 1 elem/uint, FP16 2, INT8 4) the first CachedTokenCount * uintsPerToken uints are
        // exactly the prefix — partial save is valid. File format v2, see the disk-persistence
        // section below (v1 was FP32-KV-only and is treated as a cache miss).
        public class Qwen3_5Cache : IDisposable
        {
            public ComputeBuffer[] kCaches; // length numLayers; null on linear layers
            public ComputeBuffer[] vCaches; // length numLayers; null on linear layers

            // INT8 KV only: per-(token, kv-head) fp16 scale + fp16 zero-point for K and V (asymmetric),
            // packed 2 halves per uint. null unless KV == INT8. Laid out [capacity, headsKV] per full layer.
            public ComputeBuffer[] kScaleZp;
            public ComputeBuffer[] vScaleZp;

            public ComputeBuffer[] convStates;      // length numLayers; null on full layers
            public ComputeBuffer[] recurrentStates; // length numLayers; null on full layers

            /// <summary>How many tokens of this cache are live. The setter exists for the restore
            /// paths (<see cref="LoadYielding"/>), which set it AFTER uploading a matching state.
            /// <para><b>It is not a rewind.</b> Assigning a SMALLER value to drop the tail of a
            /// conversation truncates the full-attention layers correctly — the K/V layout is
            /// token-major, so the first N rows really are the first N tokens — and leaves the 18
            /// Gated DeltaNet layers holding <c>conv_state</c>/<c>recurrent_state</c> for the WHOLE
            /// conversation, because that state is running, not indexed by position. The result is a
            /// model that forgot in a quarter of its layers and remembers in the rest: no error, no
            /// exception, nothing a smoke test sees. Anything that needs to go back to an earlier
            /// prefix must re-establish the whole state instead (re-initialize, or restore a snapshot
            /// taken AT that prefix) — see NPCChatBase.ResetConversationRoutine and its probe.</para></summary>
            public int CachedTokenCount { get; set; }
            public int Capacity => capacity;

            // KV-cache precision for the full-attention layers' K/V (independent of weight quant).
            // FP16 packs 2 halves/uint (half the buffer + read bandwidth). DeltaNet conv/recurrent
            // states are always FP32 regardless of this.
            public readonly KVQuant KV;

            readonly int numLayers;
            readonly int capacity;

            const uint FILE_MAGIC = 0x51354B56;   // "Q5KV"
            const int FILE_VERSION = 2;           // v1 (FP32-KV-only payload) loads as a miss

            // Hitch tuning (frame budget, chunk size, readbacks in flight) is shared across all
            // models — the knobs live in Base/LLM.cs: LLM.UploadFrameBudgetMs,
            // LLM.UploadChunkFloats, LLM.SaveReadbacksInFlight.

            public Qwen3_5Cache(
                int capacity,
                Qwen3_5LayerType[] layerTypes,
                int headsKV, int headDim,
                int convDim, int convKernelSize,
                int numVHeads, int headKDim, int headVDim,
                KVQuant kv = KVQuant.FP32)
            {
                this.numLayers = layerTypes.Length;
                this.capacity = capacity;
                this.KV = kv;

                kCaches = new ComputeBuffer[numLayers];
                vCaches = new ComputeBuffer[numLayers];
                convStates = new ComputeBuffer[numLayers];
                recurrentStates = new ComputeBuffer[numLayers];

                // K/V storage width depends on KV precision (all buffers are stride-4 uint; the count
                // is what shrinks): FP32 -> count = elems; FP16 -> elems/2; INT8 -> elems/4 + scale/zp.
                // (head_dim is even and a multiple of 4, so the division is always exact.) DeltaNet
                // conv/recurrent states stay FP32.
                int kvElems = capacity * headsKV * headDim;
                int kvUints = KVQuantUtil.UIntCount(kvElems, kv);
                int convFloats = convDim * (convKernelSize - 1);
                int recFloats = numVHeads * headKDim * headVDim;

                for (int i = 0; i < numLayers; i++)
                {
                    if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                    {
                        kCaches[i] = new ComputeBuffer(kvUints, 4, ComputeBufferType.Structured);
                        vCaches[i] = new ComputeBuffer(kvUints, 4, ComputeBufferType.Structured);
                    }
                    else // LinearAttention
                    {
                        convStates[i]      = new ComputeBuffer(convFloats, 4, ComputeBufferType.Structured);
                        recurrentStates[i] = new ComputeBuffer(recFloats, 4, ComputeBufferType.Structured);
                    }
                }

                if (kv == KVQuant.INT8)
                {
                    // one fp16 scale + one fp16 zero-point per (token, kv-head) → 2 halves = 1 uint each,
                    // on the full-attention layers only.
                    kScaleZp = new ComputeBuffer[numLayers];
                    vScaleZp = new ComputeBuffer[numLayers];
                    int szCount = capacity * headsKV;   // uints (scale|zp packed 2 halves per uint)
                    for (int i = 0; i < numLayers; i++)
                    {
                        if (layerTypes[i] != Qwen3_5LayerType.FullAttention) continue;
                        kScaleZp[i] = new ComputeBuffer(szCount, 4, ComputeBufferType.Structured);
                        vScaleZp[i] = new ComputeBuffer(szCount, 4, ComputeBufferType.Structured);
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
                    kScaleZp?[i]?.Release();
                    vScaleZp?[i]?.Release();
                    convStates[i]?.Release();
                    recurrentStates[i]?.Release();
                }
            }

            // ------------------------------------------------------------------ disk persistence
            //
            // On-disk layout, FILE_VERSION 2 (little-endian; v1 files were FP32-KV-only and load
            // as a miss so they get regenerated):
            //   uint32 magic 'Q5KV'
            //   int32  version              (2)
            //   uint64 contextHash          caller-provided identity — the Qwen3.5 callers fold in
            //                               the weights path (→ model size + weight quant), cache
            //                               capacity, KV quant and the prompt/conversation tokens;
            //                               any mismatch on load = miss
            //   uint8  kvQuant              payload packing ((byte)KVQuant) — must equal the live KV
            //   int32  tokens               CachedTokenCount at save time
            //   int32  layerCount
            //   int32  extraLen + bytes     opaque caller state (token-seen counts, chat flags,
            //                               transcript, ... — 0 for the plain system-prompt cache)
            //   per layer: uint8 kind (0 = full attention, 1 = DeltaNet), then length-prefixed blobs:
            //     kind 0: K rows, V rows            tokens x rowUints x 4 bytes each — token-major
            //                                       layout means the first `tokens` rows ARE the
            //                                       prefix at any packing (FP32/FP16/INT8)
            //             [INT8 KV only] kScaleZp, vScaleZp prefixes (tokens x headsKV uints — the
            //                                       per-(token,head) packed fp16 scale|zp pairs)
            //     kind 1: conv_state, recurrent_state (full FP32 buffers)

            /// <summary>
            /// Writes the current prefix state (CachedTokenCount tokens of K/V — and their
            /// scale/zero-points under INT8 KV — plus the full SSM states) to disk, tagged with
            /// <paramref name="contextHash"/> and an optional opaque <paramref name="extraState"/>
            /// blob. GPU reads go through AsyncGPUReadback (no pipeline stall, at most
            /// LLM.SaveReadbacksInFlight pending) and the file write happens on a worker thread
            /// (to a .tmp swapped in atomically, so a torn write can't shadow a good cache) —
            /// safe to run during gameplay.
            /// </summary>
            public IEnumerator SaveYielding(string path, ulong contextHash, byte[] extraState = null)
            {
                int tokens = CachedTokenCount;
                if (tokens <= 0) yield break;
                bool int8 = KV == KVQuant.INT8;

                // Readback manifest, in file order. Per-token row size in BYTES is
                // (buf.count / capacity) * 4 — exact at any packing, because buf.count is already
                // the packed uint count and capacity divides it evenly.
                var bufs = new List<ComputeBuffer>();
                var sizes = new List<int>();
                for (int i = 0; i < numLayers; i++)
                {
                    if (kCaches[i] != null)
                    {
                        int rowBytes = (kCaches[i].count / capacity) * 4;
                        bufs.Add(kCaches[i]); sizes.Add(tokens * rowBytes);
                        bufs.Add(vCaches[i]); sizes.Add(tokens * rowBytes);
                        if (int8)
                        {
                            int szBytes = (kScaleZp[i].count / capacity) * 4;   // headsKV uints/token
                            bufs.Add(kScaleZp[i]); sizes.Add(tokens * szBytes);
                            bufs.Add(vScaleZp[i]); sizes.Add(tokens * szBytes);
                        }
                    }
                    else
                    {
                        bufs.Add(convStates[i]);      sizes.Add(convStates[i].count * 4);
                        bufs.Add(recurrentStates[i]); sizes.Add(recurrentStates[i].count * 4);
                    }
                }

                // sliding window of readbacks: at most LLM.SaveReadbacksInFlight pending at once.
                // Each result is copied to managed memory the same frame it completes — readback
                // data doesn't survive past its frame — so the window size also bounds per-frame
                // copy work.
                int total = bufs.Count;
                var blobs = new byte[total][];
                var reqs = new AsyncGPUReadbackRequest[total];
                int nextToIssue = 0, doneCount = 0, inFlight = 0;
                while (doneCount < total)
                {
                    while (inFlight < LLM.SaveReadbacksInFlight && nextToIssue < total)
                    {
                        reqs[nextToIssue] = AsyncGPUReadback.Request(bufs[nextToIssue], sizes[nextToIssue], 0);
                        nextToIssue++; inFlight++;
                    }
                    for (int r = 0; r < nextToIssue; r++)
                    {
                        if (blobs[r] != null) continue;
                        if (reqs[r].hasError)
                        {
                            ConsoleMessage.Warning("Qwen3.5 KV-cache save aborted: GPU readback error");
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
                byte kvByte = (byte)KV;
                int blobsPerFull = int8 ? 4 : 2;

                var task = Task.Run(() =>
                {
                    string tmp = path + ".tmp";
                    using (var bw = new BinaryWriter(File.Create(tmp)))
                    {
                        bw.Write(FILE_MAGIC);
                        bw.Write(FILE_VERSION);
                        bw.Write(contextHash);
                        bw.Write(kvByte);
                        bw.Write(tokens);
                        bw.Write(n);
                        int extraLen = extraState?.Length ?? 0;
                        bw.Write(extraLen);
                        if (extraLen > 0) bw.Write(extraState);
                        int b = 0;
                        for (int i = 0; i < n; i++)
                        {
                            bw.Write(kinds[i]);
                            int cnt = kinds[i] == 0 ? blobsPerFull : 2;
                            for (int j = 0; j < cnt; j++, b++)
                            {
                                bw.Write(blobs[b].Length);
                                bw.Write(blobs[b]);
                            }
                        }
                    }
                    if (File.Exists(path)) File.Delete(path);
                    File.Move(tmp, path);
                });
                while (!task.IsCompleted) yield return null;
                if (task.IsFaulted)
                    ConsoleMessage.Warning("Qwen3.5 KV-cache save failed: " + task.Exception?.GetBaseException().Message);
            }

            /// <summary>
            /// Restores a prefix state written by SaveYielding. File IO + parsing run on a worker
            /// thread; GPU uploads are chunked under a per-frame time budget (LLM.UploadFrameBudgetMs /
            /// LLM.UploadChunkFloats) so no single frame hitches. The header must match this cache
            /// exactly (version, <paramref name="expectedContextHash"/>, KV quant, layer layout,
            /// per-blob sizes) or onLoaded(false) fires and nothing is uploaded (caller falls back
            /// to recomputing). <paramref name="acceptExtra"/> (optional) sees the file's extra-state
            /// blob BEFORE any GPU upload and can veto the restore by returning false.
            /// On success CachedTokenCount is set and onLoaded(true) fires.
            /// </summary>
            public IEnumerator LoadYielding(string path, ulong expectedContextHash, Action<bool> onLoaded,
                                            Func<byte[], bool> acceptExtra = null)
            {
                int n = numLayers;
                bool int8 = KV == KVQuant.INT8;
                byte kvByte = (byte)KV;

                // Main-thread snapshot of the live layout so the worker validates the file against
                // it without touching Unity objects.
                var kinds = new byte[n];
                var rowBytes = new int[n];   // per-token K/V bytes (full layers)
                var szBytes = new int[n];    // per-token scale/zp bytes (INT8 full layers)
                var convBytes = new int[n];  // full conv_state bytes (linear layers)
                var recBytes = new int[n];   // full recurrent_state bytes (linear layers)
                for (int i = 0; i < n; i++)
                {
                    if (kCaches[i] != null)
                    {
                        kinds[i] = 0;
                        rowBytes[i] = (kCaches[i].count / capacity) * 4;
                        if (int8) szBytes[i] = (kScaleZp[i].count / capacity) * 4;
                    }
                    else
                    {
                        kinds[i] = 1;
                        convBytes[i] = convStates[i].count * 4;
                        recBytes[i] = recurrentStates[i].count * 4;
                    }
                }

                // Up to 4 payloads per layer (K, V, kScaleZp, vScaleZp), parsed to uint[] —
                // bit-exact for every packing; the cache buffers are stride-4 uint anyway.
                var slots = new uint[n * 4][];
                byte[] extra = null;
                int tokens = 0;
                string error = null;

                var task = Task.Run(() =>
                {
                    using var br = new BinaryReader(File.OpenRead(path));
                    if (br.ReadUInt32() != FILE_MAGIC) { error = "bad magic"; return; }
                    int ver = br.ReadInt32();
                    if (ver != FILE_VERSION) { error = $"stale format v{ver}"; return; }   // v1 = miss
                    if (br.ReadUInt64() != expectedContextHash) { error = "context hash mismatch"; return; }
                    if (br.ReadByte() != kvByte) { error = "kv quant mismatch"; return; }
                    tokens = br.ReadInt32();
                    if (tokens <= 0 || tokens > capacity) { error = "token count out of range"; return; }
                    if (br.ReadInt32() != n) { error = "layer count mismatch"; return; }

                    int extraLen = br.ReadInt32();
                    if (extraLen < 0 || extraLen > (64 << 20)) { error = "bad extra-state size"; return; }
                    byte[] scratch = null;   // reused across reads — halves the transient garbage
                    bool ReadExact(byte[] dst, int len)
                    {
                        int read = 0;
                        while (read < len)
                        {
                            int got = br.Read(dst, read, len - read);
                            if (got <= 0) return false;
                            read += got;
                        }
                        return true;
                    }
                    extra = new byte[extraLen];
                    if (!ReadExact(extra, extraLen)) { error = "truncated file"; return; }

                    for (int i = 0; i < n; i++)
                    {
                        if (br.ReadByte() != kinds[i]) { error = "layer kind mismatch"; return; }
                        int cnt = kinds[i] == 0 ? (int8 ? 4 : 2) : 2;
                        for (int j = 0; j < cnt; j++)
                        {
                            int want = kinds[i] == 0
                                ? tokens * (j < 2 ? rowBytes[i] : szBytes[i])
                                : (j == 0 ? convBytes[i] : recBytes[i]);
                            int len = br.ReadInt32();
                            if (len != want) { error = "payload size mismatch"; return; }
                            if (scratch == null || scratch.Length < len) scratch = new byte[len];
                            if (!ReadExact(scratch, len)) { error = "truncated file"; return; }
                            var u = new uint[len / 4];
                            Buffer.BlockCopy(scratch, 0, u, 0, len);
                            slots[i * 4 + j] = u;
                        }
                    }
                });
                while (!task.IsCompleted) yield return null;
                if (task.IsFaulted) error = task.Exception?.GetBaseException().Message;

                if (error == null && acceptExtra != null && !acceptExtra(extra))
                    error = "extra state rejected by caller";
                if (error != null)
                {
                    ConsoleMessage.Warning($"Qwen3.5 KV-cache load failed ({error}) — recomputing");
                    onLoaded?.Invoke(false);
                    yield break;
                }

                // budgeted upload: SetData in LLM.UploadChunkFloats-sized pieces (uints — same
                // 4 bytes each), yielding once LLM.UploadFrameBudgetMs of main-thread copy time
                // is spent in a frame. (The old one-LAYER-per-frame upload pushed K+V of a full
                // layer — several MB — in a single frame and dropped play mode to ~48 fps.)
                var budget = System.Diagnostics.Stopwatch.StartNew();
                for (int i = 0; i < n; i++)
                {
                    int cnt = kinds[i] == 0 ? (int8 ? 4 : 2) : 2;
                    for (int j = 0; j < cnt; j++)
                    {
                        ComputeBuffer dst = kinds[i] == 0
                            ? (j == 0 ? kCaches[i] : j == 1 ? vCaches[i] : j == 2 ? kScaleZp[i] : vScaleZp[i])
                            : (j == 0 ? convStates[i] : recurrentStates[i]);
                        var up = UploadChunked(dst, slots[i * 4 + j], budget);
                        while (up.MoveNext()) yield return up.Current;
                    }
                }

                CachedTokenCount = tokens;
                onLoaded?.Invoke(true);
            }

            // Uploads `data` into `buf` in LLM.UploadChunkFloats-sized SetData calls, yielding a
            // frame whenever the shared budget stopwatch crosses LLM.UploadFrameBudgetMs.
            IEnumerator UploadChunked(ComputeBuffer buf, uint[] data, System.Diagnostics.Stopwatch budget)
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
