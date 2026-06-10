# LLM Loading & Inference Smoothness — what actually matters

Findings from making Qwen3.5 (and then Gemma3) load and run **without frame freezes** in a live
scene (June 2026, RTX 4060 laptop, Unity 2022.3, D3D11). Every item below was found by measuring
(per-frame ms + GC counts via `FrameSpikeLogger`, phase markers in the logs, Editor.log analysis),
not by guessing — the freezes had *five* independent causes, and fixing any one of them alone was
not enough.

## The five freeze sources, in the order we found them

### 1. GC stop-the-world from weight-loading garbage (the big one)
Reading every weight file with `File.ReadAllBytes` + a same-size `uint[]` copy allocated ~2x the
model size (≈3 GB for the 1.5 GB Qwen) in temporary arrays. Unity's Boehm GC pauses **all**
threads — background-thread allocations freeze the main thread just the same.
**Fix** (`Qwen3_5Weights` / `Gemma3Weights`): pooled exact-size `byte[]` reused across files +
`SetData(byte[], srcByte, dstByte, countBytes)` directly (offsets/counts are in *array-element*
units, so bytes). A `SemaphoreSlim(4)` gates reads and is released only **after** a file's GPU
upload completes, bounding boot RAM to ~120 MB. Result: a 1.5 GB stream with **2** gen-0
collections total and mean load frame ~1 ms.

### 2. GPU work bursts on single frames
Three different bursts, one budget: (a) creating all ComputeBuffers (1.5 GB, incl. one 486 MB
buffer) in the constructor frame, (b) whole-buffer `SetData` calls (Gemma3 uploaded its entire
335 MB embedding in ONE call), (c) the system-prompt prefill running each transformer layer over
the *whole* prompt (~70 ms of GPU per layer-frame in-editor → ~14 fps for 1.5 s).
**Fix**: an `UploadPump` coroutine charges *both* lazy buffer creation and sliced `SetData`
against a 24 MB/frame byte budget; the constructor only builds a file manifest. Prompt prefill is
**chunked** (8 tokens per forward — the KV cache / SSM states carry context between chunks), so a
layer-frame's GPU bite is ~8x smaller.

### 3. Driver kernel compilation on first dispatch
Each compute kernel's ISA is compiled by the driver on its FIRST dispatch — up to ~800 ms for the
biggest kernel (DeltaNet). A normal forward compiles a whole layer's kernels in one frame.
**Fix**: `PrewarmKernels()` (static, per model) dispatches every kernel once with zero-size
uniforms and dummy buffers — degenerate (all threads early-out), needs no weights, **one kernel
compile per frame**. Run it at scene start via `Model.Prewarm()`; `Warmup()`/`InitializeChat`
also run it (idempotent). Distinct dummy buffer per property name — D3D11 forbids the same UAV
in two slots of one dispatch.

### 4. Single-threaded GPU sampler (also the tok/s ceiling)
`SampleToken` / `ArgMax` were `numthreads(1,1,1)` — ONE GPU thread doing ~45 full passes over a
248–262k-token vocab (~11M serial iterations ≈ 200 ms of GPU **per generated token**). Looked
like a readback stall; was actually kernel runtime.
**Fix** (in `Qwen3_5CS.compute` / `Gemma3CS.compute`): 256-thread group — parallel max/sum
reductions, top-k as k parallel argmax passes (winners knocked out of a scratch buffer), final
≤64-candidate min-p/top-p/draw serial on thread 0. Math identical to the reference serial sampler
(kept as fallback for `top_k <= 0 || > 64`): probs normalized over the FULL vocab, nucleus cut
includes the crossing token. On top of that, token readback uses `AsyncGPUReadback`
(`SampleYielding`) instead of a blocking `GetData`, so the main thread never waits on the GPU
queue.

### 5. Per-construction managed churn
(a) The tokenizer (10+ MB JSON → ~250k dict entries) was re-parsed on every model construction —
~1 s of background work whose garbage triggered a ~300 ms full GC mid-load. Now cached per path
in the `LLM` base (`GetOrCreateTokenizer`); `Prewarm()` starts the parse at scene start.
(b) `DeepUnityMeta` eagerly loaded all 13 compute shaders on first touch (~100+ ms inside the
model constructor); now lazy per shader.
(c) The `LLM` base subscribed each instance to a static editor event and never unsubscribed —
dead models stayed GC-rooted and their finalizers re-ran `Release()` off the main thread.
`OnReleased()` (call it at the end of every concrete `Release()`) unhooks + suppresses the
finalizer.
(d) A blocking `GC.Collect()` after releasing a model costs ~400 ms; with incremental GC enabled,
spread it: `while (GarbageCollector.CollectIncremental(2_000_000UL)) yield return null;`
(e) Gemma3's RoPE tables were computed synchronously in the constructor with `Mathf.FloatToHalf`
(slow per-call native overhead) — now computed on a background thread with a managed converter
and uploaded by a coroutine (same as Qwen).

## The user-facing API (all of the above is internal)

```csharp
// scene start (optional but recommended — hides one-time costs while the player does other things):
StartCoroutine(Qwen3_5ForCausalLM.Prewarm());      // kernels (1/frame) + tokenizer parse

// when the model is needed:
var llm = new Qwen3_5ForCausalLM();                 // cheap ctor; weights stream in background
yield return llm.InitializeChat(systemPrompt);      // waits stream + warms kernels + caches prompt
yield return llm.Chat(prompt, onToken);             // chunked prefill + async sampling internally
llm.Release();                                      // deterministic GPU free
```

Same surface on `Gemma3ForCausalLM`. Generalized pieces live in the abstract `LLM` base:
`Warmup()` contract, `GetOrCreateTokenizer`, `OnReleased`.

## How to measure (the pipeline that found all of this)

1. Drop a `FrameSpikeLogger` in the scene (`DeepUnity.FrameSpikeLogger.Ensure()`, opt-in, in
   `Main/FrameSpikeLogger.cs`) — logs every frame > 20 ms with whether a GC collection ran on it.
2. Add one `Debug.Log` phase marker per load step.
3. Reproduce, then read Editor.log: match spike frames to phases; `GC: YES` ⇒ allocation problem,
   no GC + around a sampler/prefill ⇒ GPU-bound frame.
4. Batch-mode numbers: `LMBootProbeRunner.RunQwenBootProbe` (-batchmode, NO -nographics) writes
   per-phase frame stats + a greedy-reply correctness check to `ProbeLogs/`.

Measured end state (Qwen3.5 0.8B, 1.5 GB weights, in-editor): load mean ~1 ms/frame, ctor frame
~50–150 ms (one-time JIT), no sampler stalls, prefill within 60 fps budget, GC hits only at
scene start / incremental on close.
