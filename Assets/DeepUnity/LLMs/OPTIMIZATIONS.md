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
   `LLMs/benchmarking/FrameSpikeLogger.cs`) — logs every frame > 20 ms with whether a GC collection ran on it.
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

### What stays higher-precision, and why (standard practice + references)

The universal rule in production PTQ: **quantize only the transformer-block linear weight
matrices** (attention `q/k/v/o`, MLP `gate/up/down`) and keep **normalization, token
embeddings, and the LM head** at higher precision. Papers/tools that explicitly do this:

- **GPTQ** — Frantar et al., [arXiv 2210.17323](https://arxiv.org/abs/2210.17323). Layer-wise PTQ
  applied to the `nn.Linear` weights *inside* the blocks; norms, embeddings and the head are not in
  the quantized set.
- **AWQ** — Lin et al., [arXiv 2306.00978](https://arxiv.org/abs/2306.00978). Protects salient
  weight channels; standard AWQ pipelines put **`lm_head` in the ignore list** (e.g. LLM Compressor
  `ignore: ["lm_head"]`).
- **QLoRA / NF4** — Dettmers et al., [arXiv 2305.14314](https://arxiv.org/abs/2305.14314). NF4 applied
  to the base linear weights only; bitsandbytes keeps `modules_to_not_convert` (defaults include
  **`lm_head`**) and LayerNorm params in fp32.
- **LLM.int8()** — Dettmers et al., [arXiv 2208.07339](https://arxiv.org/abs/2208.07339). Mixed
  precision: emergent outlier feature dims stay fp16.
- **SmoothQuant** — Xiao et al., [arXiv 2211.10438](https://arxiv.org/abs/2211.10438). Quantizes the
  block linear layers (after migrating activation outliers into the weights); norms/embeddings/head
  untouched.
- **llama.cpp GGUF** — `--leave-output-tensor` / `--output-tensor-type` / `--token-embedding-type`;
  the `_M` k-quant mixes deliberately raise the bit-width on `output.weight` (and embeddings).
  ([quantize README](https://github.com/ggml-org/llama.cpp/blob/master/tools/quantize/README.md))

**Why norms especially must never be quantized:** a norm gamma *multiplies the entire hidden
vector element-wise*, so its quantization error is **not** averaged away inside a dot-product sum
— it rescales a whole activation channel and compounds layer-over-layer. Norms are also ~0.1% of
params, so quantizing them saves essentially nothing. Pure downside.

**DeepUnity status vs the above (`import_params.py`):**
- ✅ **Norms already correct** — every `*layernorm` / `q_norm` / `k_norm` / final `norm` (and the
  DeltaNet `A_log`/`dt_bias`/`conv1d`/`in_proj_a/b`) go through `Exporter.fp16` in *every* quant
  mode. We do **not** quantize norms.
- ✅ **RESOLVED (2026-06-26): embed_tokens/lm_head is now fp16 in EVERY mode.** Previously it was
  int8/int4-quantized — and since it doubles as the lm_head mapping to 248k–262k logits, int4 error
  there poisoned every logit (prime suspect for the int4 decode collapse, esp. Gemma3-270M where the
  tied embedding is 63% of the model). Now only the transformer-block linear weights are quantized.
  Implemented across: `import_params.py` (`Exporter.embedding` → always fp16), both `*Weights.cs`
  loaders (embed manifest fp16, no scales), both `*CS.compute` (embed-lookup + `LmHeadPredict[1Vec]`
  read via `readH` in all INT8/INT4 variants, `embed_scales` dropped), both `*Model.cs`
  (`BindScales` null-guard). New on-disk sizes vs fp16 (Qwen 1436 / Gemma 512 MB):
  **Qwen int8 963 (-33%) / int4 755 (-47%); Gemma int8 417 (-19%) / int4 375 (-27%).**
  Note: with the embedding fp16, **int4 barely beats int8 on Gemma** (375 vs 417 MB) because the
  321 MB fp16 embedding dominates both — int4 only pays off on models where the block linears, not
  the embedding, hold most of the weight. **Next: re-measure decode quality with `QuantProbe`** —
  does int4 (and int8) now produce coherent text on Gemma, vs the old collapse?

## KV-cache quantization (2026-06-26)

KV-cache precision is a SEPARATE axis from weight quant, chosen via `KVQuant {FP32, FP16, INT8}` in
the model ctor (independent of `LLMQuant`). Shared, DRY: the precision→layout math + keyword wiring
live in `KVQuantUtil` (Base/LLM.cs); the pack/unpack lives once in the `KVCache.hlsl` compute include
(`KV_READ_K`/`KV_READ_V`/`KV_WRITE2`, selected by the `KV_FP16`/`KV_INT8` multi_compile). Applies only
to the attention layers' K/V (Qwen3.5's 6 full-attn layers + Gemma3's attention). **DeltaNet
conv/recurrent state stays FP32 always** — it's fixed-size (doesn't grow with context, so no
bandwidth/VRAM win) and recurrently accumulated (error compounds), and every engine keeps it fp32
(fla keeps the scan state fp32 even in bf16 mode; Mamba/Quamba keep the selective-scan state high
precision). Disk prompt-cache is FP32-only for now (quantized KV → recompute prompt).

**FP16 KV (the default).** Packed 2 halves/uint, in-kernel f32tof16/f16tof32. Measured (RTX 4060):
- **~Lossless**: Qwen greedy reply BIT-IDENTICAL to FP32 KV (24 tokens); Gemma matches the opening
  then diverges to a coherent continuation (small fp16 rounding, more attention layers).
- **VRAM**: halves the KV cache — Qwen 192→**96 MB** (@8k cap), Gemma 72→**36 MB** (@2k cap).
- **Speed: NEUTRAL** — decode 31.0/27.3 vs FP32's 30.9/27.2 (Qwen), 57.8/51.3 vs 57.4/51.1 (Gemma);
  decay ~12% both. This **corrects the earlier "decode is KV-bandwidth-bound" assumption**: at these
  model sizes/contexts decode is **dispatch-bound** (hundreds of small kernel launches per token,
  esp. Qwen's DeltaNet chain), so halving KV *bytes* doesn't change tok/s — same reason int8 *weights*
  were speed-neutral. ⇒ **FP16 KV's payoff is VRAM, not speed.** Default since it's lossless + free.

**INT8 KV (optional, `KVQuant.INT8`).** Asymmetric uint8 + per-(token,head) scale/zp (K/V are
activation-like, not zero-centered → asymmetric, unlike symmetric weight-int8). Buys a further KV
VRAM cut (Qwen 96→48, Gemma 36→18 MB) but — being dispatch-bound — likely **also speed-neutral**, and
it's lossy (error accumulates over context). So it's a marginal VRAM option for tight-VRAM cases, NOT
the default. (Status: implemented — the quantizing write is `WriteCacheFull` under the `KV_INT8`
keyword: one threadgroup per (token, kv-head) reduces head_dim min/max, packs uint8 4/uint into
`kv_cache` and the fp16 scale|zp into `kv_scale_zp_w` = `cache.kScaleZp/vScaleZp`; attention reads
dequantize via `kv_unpack8`. Pending: human in-editor validation of decode vs FP32; the FP32-only KV
disk cache is not yet extended to INT8 — an INT8-cache prompt is recomputed, not restored.)
