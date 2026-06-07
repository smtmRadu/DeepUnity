using System;
using System.Collections;
using System.Threading.Tasks;
using UnityEngine;

namespace DeepUnity
{
    namespace Qwen3_5Modeling
    {
        // FP16 weights packed two halves per uint, stored on the GPU as ComputeBuffer<uint>.
        // Per-layer arrays are sized to NUM_LAYERS; entries are null for the wrong layer type.
        public class Qwen3_5Weights : IDisposable
        {
            // Tied embedding (also serves as lm_head) + final RMSNorm
            public ComputeBuffer embedLmHead;
            public ComputeBuffer finalNormGamma;

            // Per-layer common
            public ComputeBuffer[] inputLnGamma;
            public ComputeBuffer[] postAttnLnGamma;
            public ComputeBuffer[] mlpGate, mlpUp, mlpDown;

            // Full-attention layers (null on linear layers)
            public ComputeBuffer[] W_Q;        // [hidden, heads_q * head_dim * 2]   (Q + gate)
            public ComputeBuffer[] W_K;        // [hidden, heads_kv * head_dim]
            public ComputeBuffer[] W_V;        // [hidden, heads_kv * head_dim]
            public ComputeBuffer[] W_O;        // [heads_q * head_dim, hidden]
            public ComputeBuffer[] qNormGamma; // [head_dim]
            public ComputeBuffer[] kNormGamma; // [head_dim]

            // Linear (Gated DeltaNet) layers (null on full layers)
            public ComputeBuffer[] W_inProjQKV;     // [hidden, key_dim*2 + value_dim]
            public ComputeBuffer[] W_inProjZ;       // [hidden, value_dim]
            public ComputeBuffer[] W_inProjA;       // [hidden, num_v_heads]
            public ComputeBuffer[] W_inProjB;       // [hidden, num_v_heads]
            public ComputeBuffer[] convWeight;      // [conv_dim, kernel_size]   depthwise
            public ComputeBuffer[] dtBias;          // [num_v_heads]
            public ComputeBuffer[] ALog;            // [num_v_heads]
            public ComputeBuffer[] linearNormGamma; // [head_v_dim]
            public ComputeBuffer[] W_outProj;       // [value_dim, hidden]

            public bool IsReady { get; private set; }

            // Boot timings (ms), filled across construction + upload, read once by the consolidated
            // "model booted up" log emitted from Qwen3_5ForCausalLM.InitializeChat.
            public double allocMs, bootTokenizerMs, bootKernelsMs, bootCacheMs, bootRopeMs, bootScratchMs;
            public double uploadMs, worstUploadMs, ropeAsyncMs, warmupMs;
            public int uploadFrames;

            readonly int numLayers, hidden, headDim, headsQ, headsKV, intermediate, vocab;
            readonly int keyDim, valueDim, qkvLinDim, convDim, kernelSize, numVHeads, headVDim;
            readonly Qwen3_5LayerType[] layerTypes;

            const int EMBED_NUM_CHUNKS = 16;
            // Per-frame GPU upload budget (packed uints; 1 uint = 4 bytes => ~24 MB/frame). Every weight
            // buffer is streamed to the GPU through this budget so no single frame stalls on SetData.
            const int UPLOAD_BUDGET_UINTS = 24 * 1024 * 1024 / 4;

            // Background readers fill this queue; the main-thread UploadPump coroutine drains it under the
            // per-frame budget. ConcurrentQueue so readers can enqueue from threadpool threads safely.
            readonly System.Collections.Concurrent.ConcurrentQueue<UploadJob> _uploads
                = new System.Collections.Concurrent.ConcurrentQueue<UploadJob>();
            volatile bool _allReadsEnqueued;

            struct UploadJob
            {
                public ComputeBuffer target;
                public uint[] data;
                public int dstOffset;   // packed-uint offset within target (for sharded buffers like the embedding)
            }

            void Enqueue(ComputeBuffer target, uint[] data, int dstOffset = 0)
                => _uploads.Enqueue(new UploadJob { target = target, data = data, dstOffset = dstOffset });

            public Qwen3_5Weights(string paramsPath)
            {
                hidden = Qwen3_5Config.HIDDEN_SIZE;
                headDim = Qwen3_5Config.HEAD_DIM;
                headsQ = Qwen3_5Config.HEADS_Q;
                headsKV = Qwen3_5Config.HEADS_KV;
                intermediate = Qwen3_5Config.MLP_INTERMEDIATE_SIZE;
                vocab = Qwen3_5Config.VOCAB_SIZE;
                numLayers = Qwen3_5Config.NUM_LAYERS;
                layerTypes = Qwen3_5Config.layer_types;

                keyDim    = Qwen3_5Config.LINEAR_KEY_DIM;
                valueDim  = Qwen3_5Config.LINEAR_VALUE_DIM;
                qkvLinDim = keyDim * 2 + valueDim;
                convDim   = Qwen3_5Config.LINEAR_CONV_DIM;
                kernelSize = Qwen3_5Config.LINEAR_CONV_KERNEL_DIM;
                numVHeads = Qwen3_5Config.LINEAR_NUM_VALUE_HEADS;
                headVDim  = Qwen3_5Config.LINEAR_VALUE_HEAD_DIM;

                // Allocate (GPU buffer creation is main-thread only; timed to confirm it isn't the boot hitch).
                var swAlloc = System.Diagnostics.Stopwatch.StartNew();
                embedLmHead    = HalfBuf(vocab * hidden);
                finalNormGamma = HalfBuf(hidden);

                inputLnGamma    = new ComputeBuffer[numLayers];
                postAttnLnGamma = new ComputeBuffer[numLayers];
                mlpGate = new ComputeBuffer[numLayers];
                mlpUp   = new ComputeBuffer[numLayers];
                mlpDown = new ComputeBuffer[numLayers];

                W_Q = new ComputeBuffer[numLayers];
                W_K = new ComputeBuffer[numLayers];
                W_V = new ComputeBuffer[numLayers];
                W_O = new ComputeBuffer[numLayers];
                qNormGamma = new ComputeBuffer[numLayers];
                kNormGamma = new ComputeBuffer[numLayers];

                W_inProjQKV = new ComputeBuffer[numLayers];
                W_inProjZ = new ComputeBuffer[numLayers];
                W_inProjA = new ComputeBuffer[numLayers];
                W_inProjB = new ComputeBuffer[numLayers];
                convWeight = new ComputeBuffer[numLayers];
                dtBias = new ComputeBuffer[numLayers];
                ALog = new ComputeBuffer[numLayers];
                linearNormGamma = new ComputeBuffer[numLayers];
                W_outProj = new ComputeBuffer[numLayers];

                int qProjOut = headsQ * headDim * 2; // 4096 (Q + gate)
                int kvProjOut = headsKV * headDim;   // 512

                for (int i = 0; i < numLayers; i++)
                {
                    inputLnGamma[i]    = HalfBuf(hidden);
                    postAttnLnGamma[i] = HalfBuf(hidden);
                    mlpGate[i] = HalfBuf(hidden * intermediate);
                    mlpUp[i]   = HalfBuf(hidden * intermediate);
                    mlpDown[i] = HalfBuf(intermediate * hidden);

                    if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                    {
                        W_Q[i] = HalfBuf(hidden * qProjOut);
                        W_K[i] = HalfBuf(hidden * kvProjOut);
                        W_V[i] = HalfBuf(hidden * kvProjOut);
                        W_O[i] = HalfBuf((headsQ * headDim) * hidden);
                        qNormGamma[i] = HalfBuf(headDim);
                        kNormGamma[i] = HalfBuf(headDim);
                    }
                    else
                    {
                        W_inProjQKV[i] = HalfBuf(hidden * qkvLinDim);
                        W_inProjZ[i]   = HalfBuf(hidden * valueDim);
                        W_inProjA[i]   = HalfBuf(hidden * numVHeads);
                        W_inProjB[i]   = HalfBuf(hidden * numVHeads);
                        convWeight[i]  = HalfBuf(convDim * kernelSize);
                        dtBias[i]      = HalfBuf(numVHeads);
                        ALog[i]        = HalfBuf(numVHeads);
                        linearNormGamma[i] = HalfBuf(headVDim);
                        W_outProj[i]   = HalfBuf(valueDim * hidden);
                    }
                }

                allocMs = swAlloc.Elapsed.TotalMilliseconds;

                // Start the drain coroutine before the reads so it idles until the first shard lands,
                // then kick off the background file reads (which enqueue upload jobs as they complete).
                DeepUnityDispatcher.Run(UploadPump());
                _ = LoadAllAsync(paramsPath);
            }

            // FP16 packed: 2 halves per 4-byte uint.
            static ComputeBuffer HalfBuf(int halfCount)
            {
                if ((halfCount & 1) != 0)
                    throw new ArgumentException($"HalfBuf needs even count, got {halfCount}");
                return new ComputeBuffer(halfCount / 2, 4, ComputeBufferType.Structured);
            }

            static uint[] ReadFP16Packed(string path, int numHalves)
            {
                byte[] bytes = System.IO.File.ReadAllBytes(path);
                if (bytes.Length != numHalves * 2)
                    throw new System.IO.IOException($"Bad size {bytes.Length}, expected {numHalves * 2} for {path}");
                uint[] packed = new uint[numHalves / 2];
                Buffer.BlockCopy(bytes, 0, packed, 0, bytes.Length);
                return packed;
            }

            async Task LoadAllAsync(string paramsPath)
            {
                var tasks = new System.Collections.Generic.List<Task>(numLayers + 2);

                // All reads run on background threads and enqueue (buffer, data) jobs as they finish.
                // The GPU upload itself is done exclusively by UploadPump on the main thread, budgeted.
                tasks.Add(ReadEmbeddingAsync(paramsPath));
                tasks.Add(Task.Run(() => Enqueue(finalNormGamma, ReadFP16Packed(paramsPath + "/norm.bin", hidden))));
                for (int i = 0; i < numLayers; i++)
                {
                    int idx = i;
                    tasks.Add(LoadLayerAsync(paramsPath, idx));
                }

                await Task.WhenAll(tasks);
                _allReadsEnqueued = true; // once the queue drains, UploadPump knows nothing more is coming
            }

            // Reads the 16 embedding shards on background threads and enqueues each at its slot offset.
            async Task ReadEmbeddingAsync(string paramsPath)
            {
                int totalHalves = vocab * hidden;
                int perChunk = totalHalves / EMBED_NUM_CHUNKS; // exactly divisible for 0.8B
                int perChunkPacked = perChunk / 2;

                Task[] tasks = new Task[EMBED_NUM_CHUNKS];
                for (int i = 0; i < EMBED_NUM_CHUNKS; i++)
                {
                    int idx = i;
                    tasks[i] = Task.Run(() =>
                        Enqueue(embedLmHead, ReadFP16Packed($"{paramsPath}/embed_tokens/part_{idx}.bin", perChunk), idx * perChunkPacked));
                }
                await Task.WhenAll(tasks);
            }

            // Single main-thread consumer: drains the upload queue, capping GPU SetData at ~24 MB/frame.
            // Large buffers (embedding shards, MLP/QKV) are sliced with the partial SetData overload so a
            // single buffer can never blow the per-frame budget. Flips IsReady once everything is uploaded.
            IEnumerator UploadPump()
            {
                var sw = System.Diagnostics.Stopwatch.StartNew();
                var slice = new System.Diagnostics.Stopwatch();
                int frames = 0;
                double worstSliceMs = 0;
                int budget = UPLOAD_BUDGET_UINTS;

                while (true)
                {
                    if (_uploads.TryDequeue(out UploadJob job))
                    {
                        int src = 0, len = job.data.Length;
                        while (src < len)
                        {
                            if (budget <= 0)
                            {
                                yield return null;               // hand the frame back to rendering
                                frames++;
                                budget = UPLOAD_BUDGET_UINTS;
                            }
                            int count = Math.Min(budget, len - src);
                            slice.Restart();
                            job.target.SetData(job.data, src, job.dstOffset + src, count);
                            double ms = slice.Elapsed.TotalMilliseconds; // one SetData; spikes here = first-touch GPU commit
                            if (ms > worstSliceMs) worstSliceMs = ms;
                            src += count;
                            budget -= count;
                        }
                    }
                    else if (_allReadsEnqueued)
                    {
                        break;                                   // queue empty and no more reads in flight => done
                    }
                    else
                    {
                        yield return null;                       // reads still in flight; check again next frame
                        frames++;
                        budget = UPLOAD_BUDGET_UINTS;
                    }
                }

                uploadFrames = frames;
                uploadMs = sw.Elapsed.TotalMilliseconds;
                worstUploadMs = worstSliceMs;
                IsReady = true;
            }

            async Task LoadLayerAsync(string paramsPath, int i)
            {
                string lp = $"{paramsPath}/layer_{i}";

                int qProjOut = headsQ * headDim * 2;
                int kvProjOut = headsKV * headDim;
                int oIn = headsQ * headDim;

                var tInLn   = Task.Run(() => ReadFP16Packed(lp + "/input_layernorm.bin", hidden));
                var tPALn   = Task.Run(() => ReadFP16Packed(lp + "/post_attention_layernorm.bin", hidden));
                var tGate   = Task.Run(() => ReadFP16Packed(lp + "/mlp_gate_proj.bin", hidden * intermediate));
                var tUp     = Task.Run(() => ReadFP16Packed(lp + "/mlp_up_proj.bin",   hidden * intermediate));
                var tDown   = Task.Run(() => ReadFP16Packed(lp + "/mlp_down_proj.bin", intermediate * hidden));

                if (layerTypes[i] == Qwen3_5LayerType.FullAttention)
                {
                    var tQ  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_proj.bin", hidden * qProjOut));
                    var tK  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_proj.bin", hidden * kvProjOut));
                    var tV  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_v_proj.bin", hidden * kvProjOut));
                    var tO  = Task.Run(() => ReadFP16Packed(lp + "/self_attn_o_proj.bin", oIn * hidden));
                    var tQN = Task.Run(() => ReadFP16Packed(lp + "/self_attn_q_norm.bin", headDim));
                    var tKN = Task.Run(() => ReadFP16Packed(lp + "/self_attn_k_norm.bin", headDim));

                    await Task.WhenAll(tInLn, tPALn, tGate, tUp, tDown, tQ, tK, tV, tO, tQN, tKN);

                    Enqueue(inputLnGamma[i], tInLn.Result);
                    Enqueue(postAttnLnGamma[i], tPALn.Result);
                    Enqueue(mlpGate[i], tGate.Result);
                    Enqueue(mlpUp[i], tUp.Result);
                    Enqueue(mlpDown[i], tDown.Result);
                    Enqueue(W_Q[i], tQ.Result);
                    Enqueue(W_K[i], tK.Result);
                    Enqueue(W_V[i], tV.Result);
                    Enqueue(W_O[i], tO.Result);
                    Enqueue(qNormGamma[i], tQN.Result);
                    Enqueue(kNormGamma[i], tKN.Result);
                }
                else
                {
                    var tQKV  = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_qkv.bin", hidden * qkvLinDim));
                    var tZ    = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_z.bin",   hidden * valueDim));
                    var tA    = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_a.bin",   hidden * numVHeads));
                    var tB    = Task.Run(() => ReadFP16Packed(lp + "/linear_in_proj_b.bin",   hidden * numVHeads));
                    var tCv   = Task.Run(() => ReadFP16Packed(lp + "/linear_conv1d.bin",      convDim * kernelSize));
                    var tDt   = Task.Run(() => ReadFP16Packed(lp + "/linear_dt_bias.bin",     numVHeads));
                    var tAlog = Task.Run(() => ReadFP16Packed(lp + "/linear_A_log.bin",       numVHeads));
                    var tNm   = Task.Run(() => ReadFP16Packed(lp + "/linear_norm.bin",        headVDim));
                    var tOut  = Task.Run(() => ReadFP16Packed(lp + "/linear_out_proj.bin",    valueDim * hidden));

                    await Task.WhenAll(tInLn, tPALn, tGate, tUp, tDown, tQKV, tZ, tA, tB, tCv, tDt, tAlog, tNm, tOut);

                    Enqueue(inputLnGamma[i], tInLn.Result);
                    Enqueue(postAttnLnGamma[i], tPALn.Result);
                    Enqueue(mlpGate[i], tGate.Result);
                    Enqueue(mlpUp[i], tUp.Result);
                    Enqueue(mlpDown[i], tDown.Result);
                    Enqueue(W_inProjQKV[i], tQKV.Result);
                    Enqueue(W_inProjZ[i], tZ.Result);
                    Enqueue(W_inProjA[i], tA.Result);
                    Enqueue(W_inProjB[i], tB.Result);
                    Enqueue(convWeight[i], tCv.Result);
                    Enqueue(dtBias[i], tDt.Result);
                    Enqueue(ALog[i], tAlog.Result);
                    Enqueue(linearNormGamma[i], tNm.Result);
                    Enqueue(W_outProj[i], tOut.Result);
                }
            }

            public void Dispose()
            {
                embedLmHead?.Release();
                finalNormGamma?.Release();
                for (int i = 0; i < numLayers; i++)
                {
                    inputLnGamma[i]?.Release();
                    postAttnLnGamma[i]?.Release();
                    mlpGate[i]?.Release();
                    mlpUp[i]?.Release();
                    mlpDown[i]?.Release();
                    W_Q[i]?.Release();
                    W_K[i]?.Release();
                    W_V[i]?.Release();
                    W_O[i]?.Release();
                    qNormGamma[i]?.Release();
                    kNormGamma[i]?.Release();
                    W_inProjQKV[i]?.Release();
                    W_inProjZ[i]?.Release();
                    W_inProjA[i]?.Release();
                    W_inProjB[i]?.Release();
                    convWeight[i]?.Release();
                    dtBias[i]?.Release();
                    ALog[i]?.Release();
                    linearNormGamma[i]?.Release();
                    W_outProj[i]?.Release();
                }
            }
        }
    }
}
