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

## FlashAttention-1 fused attention (Gemma3, June 2026)

The legacy attention was 4 dispatches per layer (`ComputeAttentionScores` -> `ApplyMask` ->
`SoftmaxRows` -> `AttendValues`) with the full `[heads, seq_q, seq_k]` score matrix written,
masked, softmaxed and re-read from global memory — and `SoftmaxRows` ran **one thread per row**
making 3 serial passes over the whole KV length. On sliding-window layers (15 of Gemma3's 18)
scores were computed for the ENTIRE cache and then masked off.

**Fix** (`FlashAttention` kernel in `Gemma3CS.compute`, toggled by
`Gemma3Model.UseFlashAttention`, on by default, requires `head_dim <= 256`): all four steps fused
into one dispatch per layer using the FlashAttention-1 online-softmax recurrence — running max
`m`, denominator `l`, accumulator rescaled by `exp(m_old - m_new)` per 256-wide KV tile, score
matrix never materialized. One threadgroup per (query, head); within a tile each thread scores
one KV position (q row in groupshared, scale pre-folded), then threads switch roles to output
dims so V reads coalesce. Sliding-window layers clamp the KV walk to `[abs_q - 512, abs_q]`, so
out-of-window tiles are never read at all.

Measured (Gemma3-270m, RTX 4060 laptop, D3D11; `FlashAttnProbeRunner.Run` via bridge):
- equivalence: worst |Δlogit| 1.8e-4 over 262k logits x 8 steps, all greedy argmax matches
- sync decode (Forward+Sample): cache 120: 62 -> 64 tok/s (~par); cache 1900: 42 -> **53 tok/s (1.26x)**
- production `Generate()` (per-layer-yielding): short prompt 23.7 -> 25.3 tok/s; 1800-token
  prompt 20.7 -> **24.2 tok/s (1.17x)**
- the gap between sync 53 tok/s and Generate() 24 tok/s is frame pacing (~19 yields/token in
  `ForwardYielding`), not GPU time — the next lever for generation speed is yield granularity,
  not attention.

The win grows with context (legacy attention work is O(kv_len) on every layer; flash is O(512)
on SW layers + O(kv_len) on the 3 full layers).

**Qwen3.5 port** (same kernel minus the sliding-window logic in `Qwen3_5CS.compute`, toggled by
`Qwen3_5Model.UseFlashAttention`): equivalent (worst |Δlogit| 6.3e-5, all greedy tokens match)
but speed-NEUTRAL — 0.94x sync decode @ 120 ctx, 1.03x @ 1900 (`FlashAttnProbeRunner.RunQwen`).
Reason: only 6 of 24 layers are full attention; the 18 DeltaNet linear-attention layers already
cost O(1) per token regardless of context, so attention is a small slice of Qwen's GPU time and
the fused kernel's lower occupancy at short KV (one threadgroup per query x head) eats the
dispatch savings. Kept ON by default anyway: it can only pull ahead as contexts approach Qwen's
8k max, and one kernel maintained for both models beats two paths diverging.

## Weight-only quantization (Qwen3.5, June 2026)

Weights dequantize IN-REGISTER inside the matmul kernels (`readQ8`/`readQ4` next to `readH`;
`INT8_WEIGHTS`/`INT4_WEIGHTS` multi_compile variants — the FP16 variant compiles to the exact
pre-quant code). Activations/KV stay fp32. `new Qwen3_5ForCausalLM(size, LLMQuant.X)` picks the
params folder and keyword; one quant mode per session. Schemes (the convention for all models):
- **INT8** — symmetric, ONE fp16 scale per output row (factors out of the dot product). 721 MB
  vs 1.5 GB. Measured ~lossless: greedy story identical to fp16, logit noise ~0.07 mean,
  speed 0.99x (0.8B decode is DISPATCH-bound, ~700+ dispatches/token through the DeltaNet
  chain — quantization buys VRAM, not tok/s, at this size).
- **INT4** — Q4_0-style, scale per 32-weight group, index = flat idx >> 5. 406 MB, but on 0.8B:
  logit noise 0.53 mean, greedy text diverges immediately into repetitive prose, AND 0.76x
  (per-element scale read in the inner loop). NOT shippable as-is on small models; salvage
  levers if 2B size ever demands it: hoist the group scale via a group-loop restructure +
  keep embed/lm_head at int8 (Q4_K_M-style mixed precision).
- Verdict: **int8 is the sweet spot**; 2B@int8 ~2.2 GB fits the 4060. Probes:
  `FlashAttnProbeRunner.RunQwenInt8` / `RunQwenInt4` (QuantProbe — full-vocab logit diff vs
  fp16, sync tok/s, side-by-side greedy stories).
