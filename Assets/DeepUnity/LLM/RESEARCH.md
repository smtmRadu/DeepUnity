# RESEARCH — Making Qwen3.5-0.8B prefill (and warmup) non-blocking in the DeepUnity DX11 compute-shader engine

Scope: a hand-written HLSL-compute LLM engine running **Qwen3.5-0.8B** inside Unity (Windows / DX11), hybrid **18 Gated DeltaNet (linear) + 6 full-attention** layers, hidden=1024, 24 layers, vocab=248320. Forward is a C# coroutine that `cs.Dispatch(...)`s ~30 kernels and `yield return null` once per layer. Verified engine constants used below come from this repo:

- `Qwen3_5Config.cs`: `HIDDEN_SIZE=1024`, `NUM_LAYERS=24`, full-attn every 4th (`L L L F …`), `LINEAR_NUM_VALUE_HEADS=16`, `LINEAR_NUM_KEY_HEADS=16`, `LINEAR_KEY_HEAD_DIM=128`, `LINEAR_VALUE_HEAD_DIM=128`, `LINEAR_CONV_KERNEL_DIM=4`, `LINEAR_CONV_DIM=6144`.
- `Qwen3_5Model.cs:545`: the recurrent scan is dispatched as **`cs.Dispatch(kDeltaNet, 1, numVHeads, 1)`** → `(1, 16, 1)` thread groups. This is Problem B's bottleneck, confirmed in code.
- `Qwen3_5Cache.cs`: per linear layer the engine stores `conv_state [conv_dim*(kernel-1)]` and `recurrent_state [num_v_heads*head_k_dim*head_v_dim] = 16*128*128 = 262144 floats = 1 MiB FP32`.

---

## 1. TL;DR / recommended plan

**Single highest-leverage fix:** Re-implement the Gated DeltaNet prefill as a **chunkwise-parallel** scan (chunk size **C=64**), turning the `(1,16,1)`-thread sequential time-loop into a sequence of parallel matmul dispatches with only `ceil(L/C)` sequential boundary-state steps. For a ~60-token system prompt this is **one or two chunks** — i.e. the sequential axis collapses from ~60 steps to ~1–2, and the per-token work moves onto thread-saturating matmuls. This is the only change that attacks the ~24 ms/token term at its root (Problem B). See [§Chunkwise sketch](#chunkwise-gated-deltanet--implementation-sketch).

Ranked by (impact × low-effort):

| # | Action | Effort | Impact | Notes |
|---|--------|--------|--------|-------|
| **R1** | **KV/state caching of the fixed system prompt** (Q6): prefill the system prompt **once ever**, snapshot every layer's `recurrent_state` + `conv_state` (linear) and K/V cache slice (full), and `CopyBuffer` them back on each `InitializeChat`. | **Low** | **Very high** | Eliminates the *recurring* ~1447 ms entirely for the common case (system prompt unchanged). Pure C# + a `CopyBuffer` kernel you already have (`kCopy`). **Do this first** — it removes the cost rather than speeding it up. |
| **R2** | **Parallelize the existing sequential scan across state entries** (Q4): keep the per-timestep time loop but dispatch `(numVHeads × headKDim)`-ish thread groups instead of `(1,16,1)`. Each of the 262 144 `recurrent_state` entries per layer can update independently within a timestep. | **Low–Med** | **High** | Cheap interim win before the full chunkwise rewrite; expected order-of-magnitude better occupancy on the per-timestep matmul. Keeps O(L) sequential steps but each is far faster. |
| **R3** | **Chunkwise-parallel Gated DeltaNet prefill, C=64** (Q2). | **High** | **Highest (root cause)** | The production approach (fla / Qwen / DeltaNet paper). Collapses sequential steps to `L/C`. Best long-term fix; supersedes R2. |
| **R4** | **Keep the compute-kernel warmup you already have** (Q1). | Done | Med | Unity's `ShaderVariantCollection`/`GraphicsStateCollection`/`ShaderWarmup` are for **graphics** PSOs, **not compute kernels** — the dummy-dispatch warmup is the correct and essentially only tool for DX11 compute. Minor refinements below. |
| **R5** | **Finer-grained frame yielding / fences** (Q5) — but *measure first*. | Low | Low–Med | Per-layer `yield` is already reasonable. Avoid any `GetData()`/synchronous readback on the hot path; use `AsyncGPUReadback` for the final logits. Finer yields reduce per-frame hitch but add latency; only useful if a single layer's dispatch batch still spikes a frame. |

**The combination that wins:** R1 removes the recurring cost for the fixed prompt; R3 (or R2 as a stopgap) makes the *unavoidable* prefills (dynamic prompts, prompt changes) fast. R1+R3 together is the target state.

---

## 2. Q1 — First-dispatch / pipeline compilation hitches (Problem A)

### What actually happens on DX11
DX11 has **no PSO object**: shaders are created with `ID3D11Device::CreateComputeShader` at load, but the **driver defers final ISA/microcode generation until first use** (first `Dispatch`). The runtime/driver compiles the GPU-specific machine code lazily, which is the ~525 ms you measured spread across ~30 kernels' first dispatches. DX12/Vulkan make this explicit via **PSOs** (`CreateComputePipeline`), so the compile happens when you create the PSO rather than at first dispatch — but the *total* cost is similar; it's just schedulable. ([D3D11 functional spec](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm), [DX background processing / PSO recompiles](https://microsoft.github.io/DirectX-Specs/d3d/BackgroundProcessing.html))

### Is there a Unity API to pre-create compute pipeline state?
**No first-class one for compute.** The Unity warmup machinery is graphics-only:
- `ShaderVariantCollection.WarmUp()` / `Shader.WarmupAllShaders()` prewarm **graphics shader variants**; the docs never list compute. Fully effective only on DX11/GL for graphics; on DX12/Vulkan/Metal the driver may still re-do work if vertex layout/render-target state differs. ([WarmUp](https://docs.unity3d.com/ScriptReference/ShaderVariantCollection.WarmUp.html), [WarmupAllShaders](https://docs.unity3d.com/ScriptReference/Shader.WarmupAllShaders.html), [Prewarm shaders manual](https://docs.unity3d.com/6000.2/Documentation/Manual/shader-prewarm.html))
- `Experimental.Rendering.GraphicsStateCollection` is the modern replacement that records runtime GPU **states** and warms PSOs on DX12/Metal/Vulkan, with `WarmUp` / `WarmUpProgressively` returning a `JobHandle` for **async** warmup — again **graphics PSOs**, tied to render-target/vertex state. ([GraphicsStateCollection](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Experimental.Rendering.GraphicsStateCollection.html), [Warm up PSOs manual](https://docs.unity3d.com/6000.3/Documentation/Manual/shader-prewarm.html))
- `Experimental.Rendering.ShaderWarmup.WarmupShader` / `WarmupShaderFromCollection` warm a graphics `Shader` for a given rendering config across all APIs, optionally on background threads (`-max-async-pso-job-count`). Still graphics. ([ShaderWarmup](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Experimental.Rendering.ShaderWarmup.html))

**Conclusion:** For **compute kernels** there is no Unity prewarm API. The correct mechanism is exactly what the engine already does (`Qwen3_5.cs:77`): **dispatch every kernel once with dummy data behind the loading screen** so the driver compiles each on its first real dispatch. That is the canonical DX11 technique.

### Refinements to the existing warmup
1. **Touch every code path, not just every kernel handle.** Driver compilation can specialize on the dynamic-branch reality. Run the dummy warmup at both `seq_len=1` (decode) and `seq_len>1` (prefill), and for **both** layer types (linear and full), since divergent branches inside a kernel can trigger separate codegen. Make sure the warmup exercises the same `SetInt`/keyword permutations used at runtime.
2. **Avoid Unity multi_compile keyword explosion.** Each `#pragma multi_compile` / `shader_feature` combination in a compute shader is a *separate kernel variant* the driver compiles independently. If kernels use keywords, prefer `cs.EnableKeyword` runtime branching or `uniform` ints over compile-time variants so there are fewer variants to warm. (General Unity shader-variant guidance: [shader variants manual](https://docs.unity3d.com/Manual/shader-variant-collections.html).)
3. **Consider DX12 for *schedulable* warmup.** Switching the project to DX12 doesn't reduce total compile time, but PSO creation can be issued from worker threads, so warmup can overlap asset loading rather than blocking the first dispatch. On DX11 you cannot move the compile off the immediate context cheaply. Only worth it if 525 ms behind the loading screen ever becomes user-visible. ([BackgroundProcessing.html](https://microsoft.github.io/DirectX-Specs/d3d/BackgroundProcessing.html))
4. **Persist nothing — you can't.** DX11 has no app-visible compute PSO cache to serialize; the driver keeps its own shader cache keyed by bytecode hash, so the *second app run* on the same machine is usually much faster than the first (cold) run. Don't rely on it across driver updates.
5. **Warmup is already "solved-ish"; don't over-invest.** The measured 525 ms is one-time and behind a loading screen. R1/R3 (prefill) are where the recurring user-visible cost is.

---

## 3. Q2 — Chunkwise PARALLEL Gated DeltaNet (the big one)

This is the core of Problem B. The recurrent scan is mathematically a matrix-valued linear recurrence; production linear-attention stacks (fla, Qwen3-Next/Qwen3.5, Mamba2-SSD) all prefill via a **chunkwise** form that does most work as dense matmuls and carries only the chunk-**boundary** state sequentially.

### 3.1 The (gated) delta rule recurrence
DeltaNet's per-token state update (state `S_t ∈ ℝ^{d_k×d_v}`, here `128×128` per head):

```
S_t = S_{t-1}(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ
```
where `β_t∈(0,1)` is the write strength. Equivalently `S_t = S_{t-1} − v_t^old k_tᵀ + v_t^new k_tᵀ`, `v_t^old = S_{t-1}k_t`, `v_t^new = β_t v_t + (1−β_t)v_t^old`. ([DeltaNet paper, arXiv 2406.06484](https://arxiv.org/html/2406.06484v4))

**Gated** DeltaNet adds Mamba2-style scalar decay `α_t∈(0,1)` (the engine's `linear_a`, derived from `A_log` + `dt_bias`):

```
S_t = S_{t-1}( α_t (I − β_t k_t k_tᵀ) ) + β_t v_t k_tᵀ        (GatedDeltaNet Eq. 8)
```
`α_t` is the data-dependent forget gate; `β_t` the delta write gate. Keys/queries get **L2 normalization** (the engine's `kL2NormHead` step) so `I − k_tk_tᵀ` is a projection when `β=1`, keeping eigenvalues in `[1−β‖k‖, 1]` for stability. ([Gated DeltaNet, arXiv 2412.06464 / ICLR'25](https://ar5iv.labs.arxiv.org/html/2412.06464))

### 3.2 WY representation / UT transform (why it becomes matmuls)
Instead of materializing `S_t` (d_k×d_v) every step, write `S_t = Σ_{i≤t} u_i k_iᵀ` with **pseudo-values** `u_i`:

```
u_t = β_t ( v_t − Σ_{i<t} u_i (k_iᵀ k_t) )
```
Within a chunk of size `C`, stack `Q,K,V ∈ ℝ^{C×d}`, `β,α` as length-C vectors. Define the strictly-lower-triangular matrix and its inverse (the **UT transform**):

```
A   = tril( Diag(β) K Kᵀ , −1)                # C×C, strictly lower-tri
T   = (I − A)⁻¹ Diag(β)                        # via forward substitution, NOT general inverse
W   = T K                                      # C×d
U   = T V                                      # C×d   (the pseudo-values)
```
`(I − A)⁻¹` is cheap because `A` is strictly lower-triangular → solve by **forward substitution** (each row depends only on earlier rows). ([DeltaNet Part II blog, Songlin Yang](https://sustcsonglin.github.io/blog/2024/deltanet-2/), [DeltaNet paper](https://arxiv.org/html/2406.06484v4))

With gating, fold the cumulative decay `γ^r = Π_{i=1}^r α_i` (a within-chunk prefix product) into the triangular solves and masks (`Diag(γ)` factors), which **does not change the matmul structure**:

```
A^W = (I − tril( Diag(β) K Kᵀ ))⁻¹ ;  W = A^W Diag(β) K
A^U = (I − tril( Diag(β) (Γ ⊙ K Kᵀ) ))⁻¹ ;  U = A^U Diag(β) V
```
(`Γ` is the matrix of decay ratios `γ^r/γ^i`.) ([Gated DeltaNet Eq. 9–10](https://ar5iv.labs.arxiv.org/html/2412.06464))

### 3.3 The chunkwise parallel equations
Let chunk index `t`, state `S_{[t]}` entering chunk `t`. Per chunk:

**Inter-chunk state recurrence (the only sequential part — `L/C` steps):**
```
S_{[t+1]} = γ_{[t]}^C · S_{[t]} + ( U_{[t]} − Diag(γ_{[t]}) W_{[t]} S_{[t]}ᵀ )ᵀ · Diag(γ_{[t]}^C / γ_{[t]}) K_{[t]}
```
(ungated special case: `S_{[t+1]} = S_{[t]} + (U_{[t]} − W_{[t]}S_{[t]}ᵀ)ᵀ K_{[t]}`). ([Gated DeltaNet Eq. 11](https://ar5iv.labs.arxiv.org/html/2412.06464), [DeltaNet Eq. 8](https://arxiv.org/html/2406.06484v4))

**Intra-chunk output (fully parallel across all chunks):**
```
O_{[t]} = Diag(γ_{[t]}) Q_{[t]} S_{[t]}ᵀ + ( Q_{[t]}K_{[t]}ᵀ ⊙ M_causal ⊙ Γ_{[t]} ) ( U_{[t]} − Diag(γ_{[t]}) W_{[t]} S_{[t]}ᵀ )
```
First term = contribution of state carried into the chunk; second term = causal intra-chunk attention-like term. `M_causal` is the C×C lower-triangular mask. ([Gated DeltaNet Eq. 12](https://ar5iv.labs.arxiv.org/html/2412.06464), [DeltaNet Eq. 9](https://arxiv.org/html/2406.06484v4))

This is structurally identical to **Mamba2's SSD chunk scan**: split into chunks of 64–256; intra-chunk matmul + chunk-state + output combine run in parallel on tensor cores, only inter-chunk state propagation is sequential — "reducing sequential operations by ~100× for typical sequences." ([Mamba2 SSD / PyTorch kernel-fusion blog](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/))

### 3.4 Chunk size, complexity, stability
- **Chunk size C = 64** is the production default: DeltaNet paper says "C is a small constant (usually 64 or 128)"; **fla's `chunk_gated_delta_rule` default `chunk_size=64`/`BT=64`**; Qwen3.5 gated-deltanet config uses `chunk_size=64`. Use **64** (multiple of 16 for matmul alignment). ([DeltaNet paper](https://arxiv.org/html/2406.06484v4), [fla chunk.py](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule), [Qwen3.5 gated-deltanet analysis](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15))
- **Complexity:** chunkwise = `O(L·C·d + L·d²)` work with `O(L/C)` sequential steps. Recurrent = `O(L·d²)` work but `O(L)` sequential steps (current engine). Fully-parallel/quadratic = `O(L²d + Ld²)`. Chunkwise is the sweet spot: subquadratic FLOPs **and** sequence-level parallelism. ([DeltaNet paper complexity table](https://arxiv.org/html/2406.06484v4))
- **Why it's faster in wall-clock even at higher FLOPs:** it saturates the GPU with dense matmuls instead of a thread-starved scalar loop. "Raw FLOP counts don't translate to wall-clock time; high GPU utilization matters more." ([DeltaNet Part II blog](https://sustcsonglin.github.io/blog/2024/deltanet-2/))
- **Numerical stability:** L2-normalize q,k (engine already does `kL2NormHead`); compute the per-chunk decay `γ` as a **prefix product in log-space then exp** (fla does `chunk_local_cumsum` on `log α`) to avoid under/overflow; keep `(I−A)⁻¹` via forward substitution (never a general inverse). ([Gated DeltaNet](https://ar5iv.labs.arxiv.org/html/2412.06464), [fla](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule))

### 3.5 fla reference pipeline (what to mirror in HLSL)
fla's `chunk_gated_delta_rule_fwd` (chunk.py, `chunk_size=64`) runs these stages — the first/last are **parallel over chunks**, only the 4th is **sequential over chunks**:
1. `chunk_local_cumsum` / gate cumsum — within-chunk prefix sum of `log α` (parallel).
2. `chunk_gated_delta_rule_fwd_intra` — build the UT transform / **WY representation** `W, U` and the A matrix (parallel; uses forward substitution on C×C).
3. *(state pre-process — context-parallel only; skip for single-GPU.)*
4. `chunk_gated_delta_rule_fwd_h` — **inter-chunk hidden-state recurrence** producing per-chunk `S_{[t]}` and `v_new` (**sequential over the L/C chunks**).
5. `chunk_fwd_o` — combine `Q, K, S, U` into the output `O` (parallel over chunks).

([fla gated_delta_rule chunk.py](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule), [fla cp/README](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md))

---

## 4. Q3 — Parallel scan / prefix-sum in HLSL, and casting the recurrence as an associative scan

### 4.1 Standard GPU scans
- **Hillis–Steele** (inclusive scan): `log₂(T)` passes, each thread adds the element `2^d` positions back. Span `O(log T)` but **work `O(T log T)`** (not work-efficient). Simplest in HLSL groupshared memory with a `GroupMemoryBarrierWithGroupSync()` between passes. Good for small per-chunk prefix products (e.g. the within-chunk `γ` cumprod over C=64). ([GPU Gems 3 Ch.39](https://www.oreilly.com/library/view/gpu-gems-3/9780321545428/ch39.html), [WebGPU prefix-sum comparison](https://yayo1.com/en/blog/webgpu-prefix-sum/))
- **Blelloch (work-efficient)**: up-sweep (reduce) + down-sweep, **work `O(T)`**, span `O(log T)`. Preferred when T is large; for C=64 within a single thread group, Hillis–Steele is fine and simpler. Process 2–4 elements/thread to cut bank conflicts. ([GPU Gems 3 Ch.39 PDF](https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf))

### 4.2 The gated-delta recurrence IS an associative (matrix) scan
`S_t = A_t S_{t-1} + B_t` with `A_t = α_t(I − β_t k_t k_tᵀ)` (d_k×d_k) and `B_t = β_t v_t k_tᵀ`. Linear recurrences compose associatively:
```
(A_j, B_j) • (A_i, B_i) = (A_j A_i ,  A_j B_i + B_j)
```
so a Blelloch scan over this operator computes all `S_t` in `Θ(log L)` sequential steps. ([Parallelizing linear recurrences / prefix sum, Linxi](https://linxic.com/blog/prefix-sum/), [Parallelizing Linear Recurrent Nets over Sequence Length, arXiv 1709.04057](https://arxiv.org/pdf/1709.04057), [Mamba/S5 associative scan, S5 paper arXiv 2208.04933](https://arxiv.org/pdf/2208.04933))

**But don't scan the dense matrices directly.** `A_t` is `128×128`; the associative-scan operator cost is `Θ(T_⊙·L)` with matrix-matrix multiply `T_⊙ = O(d³)` per combine — "for dense `A_t ∈ ℝ^{N×N}`, the associative scan quickly becomes prohibitively expensive." ([Structured transition matrices, arXiv 2509.22284](https://arxiv.org/pdf/2509.22284)) The **chunkwise** form (§3) is exactly the practical answer: it's a *blocked* scan that exploits `A_t`'s rank-1-update structure (Householder/WY) so each "combine" is cheap matmuls, not full `O(d³)` per token. **Recommendation:** use Hillis–Steele only for the scalar within-chunk `γ` cumprod; use the chunkwise state recurrence (§3.3) for `S` — do **not** implement a generic matrix associative scan over the full sequence.

### 4.3 HLSL pattern
```hlsl
groupshared float sdata[64];          // C=64 per group
// inclusive Hillis-Steele scan of log-alpha within a chunk
uint t = GTid.x; sdata[t] = logAlpha[t];
GroupMemoryBarrierWithGroupSync();
[unroll] for (uint ofs = 1; ofs < 64; ofs <<= 1) {
    float v = (t >= ofs) ? sdata[t - ofs] : 0.0;
    GroupMemoryBarrierWithGroupSync();
    sdata[t] += v;
    GroupMemoryBarrierWithGroupSync();
}
gamma[t] = exp(sdata[t]);             // cumulative decay product
```

---

## 5. Q4 — Cheap interim win: parallelize the *existing* sequential scan across state entries

**Current code (`Qwen3_5Model.cs:545`):** `cs.Dispatch(kDeltaNet, 1, numVHeads, 1)` → **16 thread groups total**, one per value head, each looping the whole sequence internally. With one head's state being `128×128 = 16384` entries updated by (at most) a handful of threads per group, the GPU is massively under-occupied. This is why prefill is ~24 ms/token: the scan is essentially serial per head.

**Key observation:** within a single timestep `t`, the recurrence `S_t = α_t(S_{t-1} − (S_{t-1}k_t)β_t k_tᵀ) + β_t v_t k_tᵀ` updates **all `num_v_heads × head_k_dim × head_v_dim = 16×128×128 = 262 144` state entries independently** once the per-head scalar `(S_{t-1}k_t)` reduction is done. **Only the time axis (t) is sequential.** So you can keep the sequential time loop but pour far more threads at each timestep.

**Concrete change (low effort, no algorithm change):**
- Split `kDeltaNet` into a per-timestep kernel called in a C# loop (or keep the internal loop but restructure threading): dispatch `(head_v_dim_blocks, num_v_heads, head_k_blocks)` so that **thousands of threads** cooperate per timestep instead of ~16 groups for the whole sequence.
- The per-timestep dependency is just the rank-1 update; the `S k_t` and `Σ` reductions are per-head over 128 elements (use a groupshared reduction). Everything else (the outer-product write across 16384 entries/head) is embarrassingly parallel.
- Trade-off: you reintroduce a C# dispatch per timestep (≈60 dispatches for the prompt) **or** keep one kernel with an internal `for t` loop and a `GroupMemoryBarrierWithGroupSync()` per step using a 2D/3D thread block that covers the state. The single-kernel internal-loop version avoids per-step dispatch overhead and is preferred.

**Expected speedup:** moving from ~16 active groups to GPU-saturating thread counts is the same class of win Mamba2/SSD report when going from a scalar scan to parallel work — the bottleneck is occupancy, not FLOPs. Realistically expect a **multiple-× reduction** in prefill wall-time (the 24 ms/token is dominated by under-utilization, not arithmetic). This does **not** reduce the `O(L)` sequential-step count — for that you need R3 (chunkwise). R4/R2 is the stopgap; R3 is the real fix. ([Mamba2 SSD occupancy argument](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/), [DeltaNet Part II — utilization > FLOPs](https://sustcsonglin.github.io/blog/2024/deltanet-2/))

> Note: decode (`seq_len=1`) is already optimal — there's a single timestep, so the `(1,16,1)` dispatch is fine there. Only **prefill** needs this. Branch on `seq_len>1`.

---

## 6. Q5 — Time-slicing GPU work across frames in Unity

### What blocks vs queues
- `ComputeShader.Dispatch` **records** the command into the immediate context and returns quickly on the CPU; it does **not** wait for GPU completion. ([Dispatch](https://docs.unity3d.com/ScriptReference/ComputeShader.Dispatch.html)) The frame stall in Problem B is **GPU-side execution time** of the thread-starved scan kernel landing within one frame's GPU budget — not CPU dispatch cost. So the cure is making the *kernel* fast (R2/R3), not just spreading dispatches.
- **Never call `ComputeBuffer.GetData()` / `Texture2D.GetPixels()` on the hot path** — these force a full CPU↔GPU **sync** (flush + stall until GPU drains). That single call can cost more than the whole forward. For the final logits readback use **`AsyncGPUReadback.Request(buffer, callback)`**, which copies GPU→CPU with **no stall**, at the cost of a few frames latency. ([AsyncGPUReadback](https://docs.unity3d.com/ScriptReference/Rendering.AsyncGPUReadback.html))
- `GraphicsFence` + `CommandBuffer.WaitOnAsyncGraphicsFence` let the **GPU** wait on a fence while the **CPU returns immediately** — useful if you split work across the async-compute queue, but on a single DX11 device it mostly serializes anyway. ([WaitOnAsyncGraphicsFence](https://docs.unity3d.com/2020.1/Documentation/ScriptReference/Rendering.CommandBuffer.WaitOnAsyncGraphicsFence.html))

### Per-layer yield: keep it, tune it
- The current "one `yield return null` per layer" is sound: it lets the GPU process that layer's dispatch batch and the render thread proceed, spreading 24 layers over up to 24 frames. **Finer yielding (per-kernel) mostly adds latency** (more frames to finish a forward) without reducing any single kernel's GPU cost — only worth it if a *single* layer's batch still over-budgets a frame. For Problem B, the DeltaNet layers are the heavy ones; once R2/R3 make them cheap, per-layer yield is plenty.
- If a single prefill layer is still too heavy after R2, yield **inside** it at chunk boundaries (after each `chunk_fwd_h` sequential step). That's a natural, low-count split (≈`L/64` points), not per-kernel.
- **Don't insert `GraphicsFence`/readback between layers just to "sync"** — that would reintroduce stalls. The implicit ordering of dispatches on the immediate context already guarantees correctness.

**Measure with the right tool:** wrap suspected sync points and confirm no hidden `GetData`. The 1447 ms is GPU execution, so a GPU profiler (RenderDoc / PIX / Unity GPU profiler) will show the DeltaNet kernel dominating — fix the kernel, not the scheduling.

---

## 7. Q6 — Stateful prompt/KV caching to avoid re-prefill (do this first)

**Problem:** `InitializeChat` re-prefills the fixed system prompt every conversation → the full ~1447 ms recurs each time. If the system prompt is constant, this is pure waste.

**Technique — snapshot & restore the post-system-prompt state:**
1. **Once, at boot** (behind the loading screen, after warmup): run the system-prompt prefill exactly once.
2. **Snapshot** every layer's state into a parallel set of "golden" `ComputeBuffer`s:
   - **Linear (Gated DeltaNet) layers:** copy `recurrent_state` (`16×128×128 = 262 144` floats = **1 MiB FP32/layer**) and `conv_state` (`conv_dim×(kernel−1) = 6144×3 = 18 432` floats ≈ **72 KiB/layer**). 18 linear layers → ~18 MiB + ~1.3 MiB.
   - **Full-attention layers (6):** copy the K and V cache **slices** for the system-prompt tokens only (`seq_sys × heads_kv × head_dim` each). With `heads_kv=2`, `head_dim=256`, ~60 tokens → `60×2×256 = 30 720` floats per K and per V per layer ≈ 120 KiB/layer FP32 (less if FP16). 6 layers → <1 MiB.
   - Also snapshot the **current position/length** counters so generation resumes at `seq_sys`.
   - **Total snapshot cost ≈ ~20 MiB** — trivial for a desktop GPU.
3. **On each new `InitializeChat`:** instead of re-prefilling, `CopyBuffer` (you already have `kCopy` / `kCopySlice`) the golden buffers back into the live cache, set length = `seq_sys`, and start decoding the user turn immediately. **Cost: ~20 MiB of GPU→GPU copies ≈ sub-millisecond**, vs ~1447 ms.

**Correctness:**
- Valid **iff the system prompt token IDs are byte-identical** every time (same template, same special tokens). If any per-conversation variable is injected into the system prompt (date, player name…), either (a) keep it out of the cached prefix and prefill only the small variable tail, or (b) re-snapshot when it changes.
- Gated DeltaNet state is **position-independent in the sense that the recurrence is causal and Markovian** — the `recurrent_state` after token `n` fully summarizes tokens `1..n`, so restoring it is exactly equivalent to having processed them. The `conv_state` holds the last `kernel−1=3` pre-conv activations, also exactly restorable. No RoPE issue for linear layers (they don't use RoPE).
- Full-attention layers: K/V are cached per absolute position with **partial RoPE already applied** (engine applies RoPE before `WriteCacheFull`). Restoring the same slice at the same positions is exact. Continue generation at position `seq_sys`.

**This is standard "prompt prefix caching"** (vLLM/SGLang call it prefix KV cache; for recurrent/SSM layers it generalizes to saving the recurrent + conv state). For a *fixed* system prompt it's the single biggest recurring-cost win and is far simpler than the chunkwise rewrite. Combine: **R1 kills the recurring case; R3 makes the unavoidable (changed-prompt) prefills fast.**

---

## Chunkwise Gated DeltaNet — implementation sketch

Target: replace the `(1,16,1)` sequential `kDeltaNet` **when `seq_len>1`** (prefill) with chunked dispatches. Keep the existing single-step recurrent kernel for `seq_len==1` (decode). Chunk size **C=64**.

### Engine constants (from this repo)
`num_v_heads = num_k_heads = 16`, `head_k_dim = head_v_dim = 128`, `conv_kernel=4`, `recurrent_state` per layer = `[16,128,128]` FP32 (1 MiB), inputs already produced by the existing pipeline: `linearQNormBuf`,`linearKNormBuf` `[seq,16,128]`, `linearVBuf` `[seq,16,128]`, `linearABuf`/`linearBBuf` `[seq,16]` (the per-token `α`,`β` after `LinearInProjA/B`). Output goes to `linearYBuf` `[seq,16,128]` as today.

### Buffer layout (new scratch, per linear layer, reused across layers)
```
gamma        [num_chunks, C, num_v_heads]            // within-chunk cumulative decay (prefix product of alpha)
A_tri        [num_chunks, num_v_heads, C, C]         // strictly-lower-tri (I - tril(diag(beta) K K^T))
T_inv        [num_chunks, num_v_heads, C, C]         // (I - A)^-1  (forward substitution)
W            [num_chunks, num_v_heads, C, head_k_dim]
U            [num_chunks, num_v_heads, C, head_v_dim]   // pseudo-values
S_chunk      [num_chunks+1, num_v_heads, head_k_dim, head_v_dim]  // boundary states; S_chunk[0]=incoming recurrent_state
```
`num_chunks = ceil(seq_len / 64)`. For a 60-token prompt that's **1 chunk** (one boundary step). Memory for 1–2 chunks is tiny; allocate for max prompt length.

### Dispatch sequence (per linear layer, prefill path)
Steps 1,2,3,5 are **parallel over (chunk × head)**; only step 4 is sequential over chunks.

```
# 1. within-chunk decay cumprod  (Hillis-Steele over C in groupshared)
Dispatch kGammaCumsum  groups=(num_chunks, num_v_heads, 1), C threads/group
   gamma[c,r,h] = exp(prefix_sum_{i<=r} log(alpha[c*C+i, h]))

# 2. build A_tri then T_inv = (I - A)^-1 by forward substitution
Dispatch kBuildA       groups=(num_chunks, num_v_heads, 1), (C threads or C*C)
   A[c,h,r,j] = (r>j) ? beta[..,h]*dot(K[r],K[j])*gamma_ratio(r,j) : 0
Dispatch kForwardSub   groups=(num_chunks*num_v_heads, 1, 1), C threads/group
   solve T_inv row-by-row (row r depends on rows <r; sequential over C inside group, parallel across chunk*head)

# 3. WY representation
Dispatch kWU           groups=(num_chunks, num_v_heads, 1)
   W[c,h] = T_inv[c,h] * (diag(beta) K[c,h])        # C x 128
   U[c,h] = T_inv[c,h] * (diag(beta) V[c,h])        # C x 128

# 4. inter-chunk state recurrence  (SEQUENTIAL over chunks — only num_chunks steps)
S_chunk[0] = recurrent_state (incoming)
for c in 0..num_chunks-1:                            # C# loop OR single kernel internal loop
   Dispatch kChunkState  groups=(num_v_heads, head_k_dim/blk, head_v_dim/blk)
      Scratch = U[c] - diag(gamma[c]) * (W[c] * S_chunk[c]^T)
      S_chunk[c+1] = gammaC[c]*S_chunk[c]
                   + ( diag(gammaC/gamma) K[c] )^T * Scratch     # rank-C update, all 16*128*128 entries parallel
recurrent_state = S_chunk[num_chunks]                # final state for subsequent decode

# 5. intra-chunk output  (parallel over all chunks)
Dispatch kChunkOut     groups=(num_chunks, num_v_heads, C/blk)
   Y[c] = diag(gamma[c]) * Q[c] * S_chunk[c]^T
        + ( (Q[c]K[c]^T) ⊙ M_causal ⊙ Gamma[c] ) * (U[c] - diag(gamma[c]) W[c] S_chunk[c]^T)
```
Then the existing `kRMSGated` + `kLinearOut` consume `linearYBuf` unchanged.

### Key implementation notes
- **Forward substitution (step 2)** is the only intra-chunk sequential bit, but it's over **C=64 inside a thread group** with `head*chunk` groups running in parallel — cheap. Don't invert a general matrix.
- **Step 4 is the new "scan"** but now over `num_chunks ≈ 1–2` for a 60-token prompt instead of 60 timesteps. Each step is dense matmuls saturating threads. This is the whole win.
- **Decode (seq_len=1)** keeps the current single-step recurrent kernel — chunking 1 token is pointless; just branch.
- **Numerical:** compute `gamma` via log-space cumsum then `exp` (avoids cumprod underflow); q,k already L2-normalized by `kL2NormHead`.
- **Validate** against the existing recurrent kernel: run both on the same prompt, compare `recurrent_state` and `Y` — they must match to FP tolerance (the chunkwise form is mathematically exact, not an approximation). ([fla naive vs chunk equivalence](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule))
- **Reference kernels to port from:** fla `chunk_gated_delta_rule` stages map 1:1 — `chunk_local_cumsum`→step1, `chunk_gated_delta_rule_fwd_intra`/`wy_fast`→steps2-3, `chunk_gated_delta_rule_fwd_h`→step4, `chunk_fwd_o`→step5. ([fla chunk.py](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule))

### Recommended rollout
1. **R1 (prompt-state caching)** — eliminates the recurring 1447 ms for the fixed system prompt. ~1 day.
2. **R2 (re-thread the existing scan across state entries)** — immediate multiple-× prefill win for dynamic prompts, no new math. ~1–2 days.
3. **R3 (full chunkwise C=64)** — the asymptotic fix; supersedes R2. Validate against the recurrent kernel. ~1 week.

---

## Sources

**Gated DeltaNet / DeltaNet chunkwise (Q2, Q3, sketch)**
- [Parallelizing Linear Transformers with the Delta Rule over Sequence Length — Yang et al., arXiv 2406.06484 (HTML)](https://arxiv.org/html/2406.06484v4) · [abs](https://arxiv.org/abs/2406.06484) · [NeurIPS 2024 poster](https://neurips.cc/virtual/2024/poster/93040)
- [Gated Delta Networks: Improving Mamba2 with Delta Rule — Yang et al., arXiv 2412.06464 (ar5iv HTML)](https://ar5iv.labs.arxiv.org/html/2412.06464) · [ICLR'25 PDF (jankautz.com)](https://jankautz.com/publications/GatedDeltaNet_ICLR25.pdf)
- [DeltaNet Explained Part II — Songlin Yang blog (chunkwise/UT-transform engineering)](https://sustcsonglin.github.io/blog/2024/deltanet-2/)
- [flash-linear-attention `gated_delta_rule` ops (chunk.py / wy_fast / chunk_fwd)](https://github.com/fla-org/flash-linear-attention/tree/main/fla/ops/gated_delta_rule) · [repo root](https://github.com/fla-org/flash-linear-attention) · [cp/README (state-passing math)](https://github.com/fla-org/flash-linear-attention/blob/main/fla/ops/cp/README.md)
- [Qwen3.5 gated-deltanet config analysis gist (chunk_size=64, head dims, conv kernel)](https://gist.github.com/justinchuby/0213aa253664fb72e9adb0089816de15)
- [Qwen3-Next architecture analysis (3:1 linear:full ratio)](https://dev.to/czmilo/qwen3-next-complete-technical-analysis-major-breakthrough-in-ai-model-architecture-for-2025-3kml)
- [Accelerating Mamba2 with Kernel Fusion — PyTorch blog (SSD chunk scan, sequential-op reduction)](https://pytorch.org/blog/accelerating-mamba2-with-kernel-fusion/)

**Parallel scan / associative linear recurrence (Q3)**
- [GPU Gems 3 Ch.39 — Parallel Prefix Sum (Scan) with CUDA, Harris](https://www.oreilly.com/library/view/gpu-gems-3/9780321545428/ch39.html) · [PDF](https://developer.download.nvidia.com/compute/cuda/2_2/sdk/website/projects/scan/doc/scan.pdf)
- [Prefix Sum on WebGPU: Hillis–Steele, Blelloch, Subgroups — Yamasaki](https://yayo1.com/en/blog/webgpu-prefix-sum/)
- [Parallel Simulation of Linear Recurrence Relations via Prefix Sum — Linxi](https://linxic.com/blog/prefix-sum/)
- [Parallelizing Linear Recurrent Nets over Sequence Length — arXiv 1709.04057](https://arxiv.org/pdf/1709.04057)
- [Simplified State Space Layers (S5) — arXiv 2208.04933 (associative scan SSM)](https://arxiv.org/pdf/2208.04933)
- [Structured Sparse Transition Matrices for State Tracking — arXiv 2509.22284 (dense-scan cost)](https://arxiv.org/pdf/2509.22284)

**Unity compute warmup & frame-slicing (Q1, Q5)**
- [Unity Manual — Prewarm shaders / Warm up PSOs](https://docs.unity3d.com/6000.3/Documentation/Manual/shader-prewarm.html) · [6000.2](https://docs.unity3d.com/6000.2/Documentation/Manual/shader-prewarm.html)
- [ShaderVariantCollection.WarmUp](https://docs.unity3d.com/ScriptReference/ShaderVariantCollection.WarmUp.html) · [Shader.WarmupAllShaders](https://docs.unity3d.com/ScriptReference/Shader.WarmupAllShaders.html) · [shader variant collections manual](https://docs.unity3d.com/Manual/shader-variant-collections.html)
- [Experimental GraphicsStateCollection](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Experimental.Rendering.GraphicsStateCollection.html) · [Experimental ShaderWarmup](https://docs.unity3d.com/6000.3/Documentation/ScriptReference/Experimental.Rendering.ShaderWarmup.html)
- [ComputeShader.Dispatch](https://docs.unity3d.com/ScriptReference/ComputeShader.Dispatch.html) · [AsyncGPUReadback](https://docs.unity3d.com/ScriptReference/Rendering.AsyncGPUReadback.html) · [CommandBuffer.WaitOnAsyncGraphicsFence](https://docs.unity3d.com/2020.1/Documentation/ScriptReference/Rendering.CommandBuffer.WaitOnAsyncGraphicsFence.html)
- [D3D11.3 Functional Spec](https://microsoft.github.io/DirectX-Specs/d3d/archive/D3D11_3_FunctionalSpec.htm) · [DirectX Background Processing / PSO recompiles](https://microsoft.github.io/DirectX-Specs/d3d/BackgroundProcessing.html) · [Compute Shader Overview (MS Learn)](https://learn.microsoft.com/en-us/windows/win32/direct3d11/direct3d-11-advanced-stages-compute-shader)
