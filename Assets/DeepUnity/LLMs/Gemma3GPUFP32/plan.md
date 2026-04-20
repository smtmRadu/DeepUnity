# Gemma3GPU Plan

## Goal

Build a new text-only Gemma 3 inference stack under `Assets/DeepUnity/LLMs/Gemma3GPU` that preserves the external behavior of the current Gemma 3 text model while keeping the decoder path on GPU end to end.

The existing `Gemma3` implementation stays untouched.

## Hard Constraints

- Do not modify files under `Assets/DeepUnity/LLMs/Gemma3`.
- Reuse existing tokenizer assets and parameter folders instead of duplicating large data.
- Ignore `EmbeddingGemma`.
- Keep tied embeddings: one GPU buffer is used for both token embedding lookup and `lm_head`.
- Remove CPU round-trips for:
  - hidden-state residual adds
  - RMSNorm
  - Q/K norm
  - RoPE application
  - attention masking
  - softmax
  - KV cache writes/reads
  - MLP input/output transitions
- CPU readback is allowed only where the public API truly requires it.
- Prefer GPU sampling too, so `Generate` and `Chat` can stay fully GPU after tokenization.

## Non-Goals

- No training path.
- No embedding model.
- No multimodal Gemma 3 vision path.
- No batching-first redesign. First-class path is single-sequence inference and cached autoregressive decode.

## Public API Target

Create a new class with nearly the same outside behavior:

- `Gemma3GPUForCausalLM`
- constructor compatible with:
  - `string params_path = "Assets/DeepUnity/LLMs/Gemma3/params_it"`
  - `string tokenizer_path = "Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.json"`
- expose:
  - `bool IsReady`
  - `float TokensPerSecond`
  - `Tensor Predict(Tensor input_ids, Tensor attn_mask = null)`
  - `IEnumerator Generate(...)`
  - `IEnumerator InitializeChat(...)`
  - `IEnumerator Chat(...)`

Reuse the current `Gemma3TokenizerFast` rather than cloning it.

## Folder Layout

Planned files under `Assets/DeepUnity/LLMs/Gemma3GPU`:

- `Gemma3GPU.cs`
- `Gemma3GPUModel.cs`
- `Gemma3GPUDecoderLayer.cs`
- `Gemma3GPUAttention.cs`
- `Gemma3GPUMlp.cs`
- `Gemma3GPUWeights.cs`
- `Gemma3GPUCache.cs`
- `Gemma3GPUBufferPool.cs`
- `Gemma3GPUSampling.cs`
- `plan.md`

Planned shader:

- `Assets/Resources/ComputeShaders/Gemma3NextCS.compute`

## Note About DeepUnityMeta

Earlier direction was to reference the new shader from `DeepUnityMeta`.

The latest constraint says not to modify current files. To honor that, the new Gemma3GPU code should load its shader directly with:

`Resources.Load<ComputeShader>("ComputeShaders/Gemma3NextCS")`

If you later want framework-wide integration, `DeepUnityMeta` can be updated in a follow-up without changing the runtime design.

## High-Level Architecture

### 1. Weight Ownership

Create a dedicated GPU weight container that owns:

- `embed_lm_head`
- per-layer QKV weights
- per-layer O projection
- per-layer MLP packed weights
- per-layer RMSNorm gamma buffers
- per-layer Q norm gamma
- per-layer K norm gamma
- final norm gamma

Important:

- `embed_lm_head` is a single `ComputeBuffer`.
- embedding lookup and lm-head projection both read from the same buffer.

### 2. Hidden-State Residency

Decoder hidden states never become `Tensor` during internal execution.

Use persistent `ComputeBuffer`s for:

- current hidden states
- residual scratch
- norm output scratch
- QKV packed output
- Q buffer
- K buffer
- V buffer
- attention scores
- attention probabilities
- attended values
- MLP intermediate
- final logits
- sampled token output

### 3. KV Cache Residency

Each decoder layer owns GPU KV cache buffers:

- full-attention layers: capacity = `MAX_POSITION_EMBEDDINGS`
- sliding-window layers: capacity = `SLIDING_WINDOW`

Cache format:

- K cache: `(cache_capacity, num_heads_kv, head_dim)`
- V cache: `(cache_capacity, num_heads_kv, head_dim)`

Metadata per layer:

- `cachedTokenCount`
- `cacheWriteCursor`
- `isSlidingWindow`

For sliding-window layers, writes become ring-buffer writes.

## Execution Modes

### Prefill

Input is a full prompt sequence.

Steps:

1. Tokenize on CPU.
2. Upload token ids to a small input buffer.
3. Run embedding lookup on GPU.
4. Run all decoder layers on GPU.
5. Populate KV cache on GPU during the pass.
6. Run final norm.
7. Run `lm_head`.
8. Sample next token on GPU if generating.

### Decode

Input is one token at a time.

Steps:

1. Upload one token id.
2. Run embedding lookup on GPU.
3. For each layer:
   - compute QKV
   - normalize Q/K
   - apply RoPE at absolute position
   - append K/V into cache
   - attend against cache-resident K/V
   - output projection
   - residual + norms
   - MLP
4. Run final norm.
5. Run tied `lm_head`.
6. Sample on GPU.

## Shader Kernel Plan

All kernels live in one file:

- `Gemma3NextCS.compute`

### Embedding / Output

- `EmbeddingLookup`
  - input: token ids
  - output: hidden states
  - multiply by `sqrt(hidden_size)` here
- `LmHead`
  - input: final hidden state
  - weight source: shared embedding buffer
  - output: logits

### Utility

- `CopyBuffer`
- `CopySlice`
- `AddResidual`
- `AddBiasIfNeeded`
- `ZeroBuffer`
- `FillNegInf`

### RMSNorm

- `RmsNormHidden`
  - hidden size = 640
  - applies Gemma form: `x * rsqrt(mean(x^2)+eps) * (1 + gamma)`
- `RmsNormHead`
  - head size = 256
  - used for Q and K normalization

### QKV / Attention Prep

- `QkvProj`
- `SplitQkv`
- `ApplyRopeSplitHalf`
- `WriteCacheFull`
- `WriteCacheSliding`
- `BuildAttentionScores`
- `ApplyCausalMask`
- `ApplySlidingWindowMask`
- `SoftcapScores`
- `SoftmaxRows`
- `AttendValues`
- `OProj`

### MLP

These can be adapted from current GLU kernels:

- `MlpGateUp`
- `MlpDown`

### Sampling

Minimum viable GPU sampling:

- `ArgMaxLastLogits`

Stretch kernels:

- `TemperatureScale`
- `TopKMask`
- `SoftmaxSample`

The first working version can use GPU argmax only.
Sampling beyond argmax can remain CPU temporarily if needed, but the target is GPU sampling too.

## Shader Math Notes

### RoPE

Do not precompute RoPE on CPU.

Options:

- compute `sin/cos` on the fly in shader from position and frequency
- or upload compact sin/cos lookup buffers once

Preferred first pass:

- upload sin/cos lookup buffers once from C#
- shader only applies rotation

Reason:

- less shader complexity
- still no hidden-state CPU round-trip

### Softmax

Need numerically stable row-wise softmax:

1. row max
2. exponentiate shifted values
3. row sum
4. divide

For the first pass, prioritize correctness over maximal kernel fusion.

## Class Responsibilities

### `Gemma3GPUForCausalLM`

- public API
- tokenizer ownership
- coroutine generation/chat loop
- timing
- lifecycle cleanup

### `Gemma3GPUModel`

- owns layers
- owns shared embedding/lm-head buffer
- owns final norm
- exposes GPU forward for prefill/decode

### `Gemma3GPUDecoderLayer`

- owns per-layer attention + MLP + norms
- runs one full decoder block on buffer inputs

### `Gemma3GPUAttention`

- QKV projection
- Q/K norm
- RoPE
- masking
- softmax
- value aggregation
- cache write/read behavior

### `Gemma3GPUMlp`

- wraps packed gate/up/down weights
- dispatches GLU-style kernels

### `Gemma3GPUWeights`

- async loading of binary files
- buffer creation
- one-time upload
- tied embedding reuse

### `Gemma3GPUCache`

- per-layer cache buffers
- reset logic
- optional serialization format later

### `Gemma3GPUBufferPool`

- shared scratch buffers by size
- avoids per-token allocation churn

### `Gemma3GPUSampling`

- GPU argmax and later top-k/top-p sampling

## Implementation Order

### Phase 1. Skeleton

Create new folder and file structure.

Deliverables:

- empty runtime classes
- constructor wiring
- direct shader loading
- config reuse from existing `Gemma3Config`
- tokenizer reuse

### Phase 2. Weight Loader

Load all required model weights into GPU buffers.

Deliverables:

- shared embedding/lm-head buffer
- layer weight buffers
- norm gamma buffers
- ready-state tracking

Validation:

- parameter counts match expectations
- no duplicate embedding/lm-head upload

### Phase 3. GPU RMSNorm + Residual

Implement utility and RMSNorm kernels first.

Deliverables:

- `CopyBuffer`
- `AddResidual`
- `RmsNormHidden`
- `RmsNormHead`

Validation:

- compare against existing CPU Gemma3 for one short prompt
- tolerance target around `1e-4` to `1e-3`

### Phase 4. GPU Attention Without Cache

Implement full prompt forward without decode cache first.

Deliverables:

- QKV projection
- split Q/K/V
- q/k norm
- RoPE
- score computation
- masking
- softmax
- attend values
- O projection

Validation:

- compare layer outputs against current Gemma3 on short sequences
- test both full-attention and sliding-window layers

### Phase 5. GPU MLP

Adapt or reimplement current GLU kernels for the new buffer flow.

Deliverables:

- MLP forward on GPU
- no `Tensor` conversion in the block

Validation:

- compare MLP outputs against existing implementation

### Phase 6. Full Decoder Block

Wire one whole layer:

1. input RMSNorm
2. attention
3. post-attn RMSNorm
4. residual add
5. pre-ffn RMSNorm
6. MLP
7. post-ffn RMSNorm
8. residual add

Validation:

- per-layer output comparison with current Gemma3

### Phase 7. Whole-Model Forward

Run:

- embedding lookup
- all layers
- final norm
- tied lm-head

Validation:

- logits comparison against current Gemma3 for fixed prompts
- exact token match under greedy decoding on a few canned prompts

### Phase 8. KV Cache

Implement cache-resident decode.

Deliverables:

- prefill writes cache
- decode reads cache
- sliding-window ring buffer
- full-attention append-only cache

Validation:

- prompt prefill + iterative decode must match full-sequence greedy output

### Phase 9. Generation / Chat

Port:

- `Generate`
- `InitializeChat`
- `Chat`

Behavior target:

- same prompt formatting
- same chat-template token layout
- same end-of-turn handling

### Phase 10. Cache Persistence

Optional first pass:

- no disk persistence

Second pass:

- serialize GPU cache to a compact binary format
- reload without rebuilding from prompt

This is lower priority than a correct live GPU decoder.

## Validation Plan

### Numerical Checks

For each module:

- RMSNorm output
- attention output
- MLP output
- decoder layer output
- final logits

Compare new GPU path against the existing Gemma3 path on:

- one-token decode
- short prompt prefill
- prompt crossing sliding-window size boundary

### Behavioral Checks

- greedy generation exact token match
- chat loop exact token match
- same stop behavior at `END_OF_TURN`

### Performance Checks

Measure:

- prefill latency
- decode tokens/sec
- GPU allocation churn

Target:

- no per-token buffer recreation in steady-state decode

## Risks

### 1. Sliding-Window Cache Semantics

This is the highest correctness risk.

Need to ensure:

- absolute RoPE positions keep increasing
- sliding-window layers only attend to the valid local history window
- ring-buffer indexing does not scramble logical token order

### 2. Softmax Stability

Attention logits can blow up if masking and softcap are not applied in the right order.

### 3. Unity Compute Limits

Some kernels may need tuning for thread-group sizes because:

- vocab is large
- seq lengths vary
- head dim is 256

### 4. Predict API

`Predict` currently returns `Tensor`.

That means some readback is unavoidable when the caller explicitly wants logits as a `Tensor`.

This is acceptable as long as internal decoder execution stays GPU-native.

## First Deliverable Cut

The smallest useful first version is:

- new `Gemma3GPU` folder
- new runtime classes
- one compute shader
- greedy-only generation
- no cache persistence to disk
- `Predict`, `Generate`, `Chat` working
- old Gemma3 untouched

## Follow-Up Improvements

- GPU top-k/top-p sampling
- binary KV cache serialization
- buffer pooling for long-lived sessions
- optional `DeepUnityMeta` integration later
- optional logits softcapping if future checkpoints need it

## Acceptance Criteria

The plan is complete when the implementation satisfies all of:

- old `Gemma3` code remains unchanged
- new code lives under `Gemma3GPU`
- decoder hidden states do not bounce CPU<->GPU inside the layer stack
- tied embeddings are stored once on GPU
- greedy generation matches current Gemma3 on reference prompts
- `Generate` and `Chat` run with GPU-resident KV cache
- one shader file contains the required kernels

## Claude Notes

This section is the practical handoff. It lists the implementation details and outliers that are easy to miss if you only read the high-level plan.

### First Principles

- Do not try to retrofit the existing `Gemma3` classes.
- Do not reuse `Tensor` operations in the hot path.
- Do not use `Tensor.Constant(ComputeBuffer, ...)` except at the very edge where the public API explicitly needs a `Tensor`.
- Keep the rewrite clean even if that means duplicating some logic from the old implementation.

The old code is useful as a math reference, not as an architecture template.

## Existing Files Worth Reading

Useful references only:

- `Assets/DeepUnity/LLMs/Gemma3/Gemma3.cs`
- `Assets/DeepUnity/LLMs/Gemma3/Gemma3Config.cs`
- `Assets/DeepUnity/LLMs/Gemma3/Gemma3DecoderLayer.cs`
- `Assets/DeepUnity/LLMs/Gemma3/Gemma3GQA.cs`
- `Assets/DeepUnity/LLMs/Gemma3/Gemma3MLP.cs`
- `Assets/DeepUnity/LLMs/Gemma3/Gemma3RMSNorm.cs`
- `Assets/DeepUnity/LLMs/Gemma3/Gemma3TokenizerFast.cs`
- `Assets/DeepUnity/Modules/Other/RotaryPositionalEmbeddings.cs`
- `Assets/Resources/ComputeShaders/GQAInferenceCS.compute`
- `Assets/Resources/ComputeShaders/GLUInferenceCS.compute`
- `Assets/Resources/ComputeShaders/LmHeadInferenceCS.compute`

Do not edit them. Use them only to preserve behavior.

## Exact Model Facts To Preserve

### Config

- vocab size: `262144`
- hidden size: `640`
- MLP intermediate size: `2048`
- number of layers: `18`
- query heads: `4`
- kv heads: `1`
- head dim: `256`
- query pre-attention scalar: `256`
- RMS eps: `1e-6`
- tie embedding: `true`
- max positions: `32768`
- sliding window size: `512`
- local rope base frequency: `10000`
- full rope theta: `1000000`

### Layer Pattern

Layer types by index:

- sliding: `0,1,2,3,4`
- full: `5`
- sliding: `6,7,8,9,10`
- full: `11`
- sliding: `12,13,14,15,16`
- full: `17`

This pattern matters because sliding layers and full-attention layers should not share the same cache semantics or rope base.

### Decoder Block Order

The order is:

1. save residual
2. input RMSNorm
3. attention
4. post-attention RMSNorm
5. add residual
6. save residual
7. pre-feedforward RMSNorm
8. MLP
9. post-feedforward RMSNorm
10. add residual

Do not simplify this to a more standard transformer block order. Preserve the current behavior exactly.

### Gemma RMSNorm Convention

This model uses:

`y = x * rsqrt(mean(x^2) + eps) * (1 + gamma)`

not:

`y = x * rsqrt(mean(x^2) + eps) * gamma`

This applies to:

- hidden RMSNorms
- Q norm
- K norm

### Embedding Scaling

Immediately after lookup:

`hidden = embedding(token) * sqrt(hidden_size)`

Do not forget this. It is present in the old model and affects output equivalence.

### RoPE Variant

Use split-half rotation, not interleaved.

For head vector split into `[x1 | x2]`:

- `y1 = x1 * cos - x2 * sin`
- `y2 = x2 * cos + x1 * sin`

This matches the current `RotaryPositionalEmbeddings.ApplyRotaryEmbeddings(..., SplitHalf)`.

## Weight Layouts

### Shared Embedding / LM Head

Parameter directory:

- `Assets/DeepUnity/LLMs/Gemma3/params_it`
- `Assets/DeepUnity/LLMs/Gemma3/params_pt`
- `Assets/DeepUnity/LLMs/Gemma3/params_ft`

Embedding / lm-head data is stored as:

- `lm_head/part_0.bin`
- ...
- `lm_head/part_13.bin`

The old code concatenates them into one flat array of shape:

- `[vocab_size, hidden_size]`

Flattening assumption is row-major:

- row = vocab index
- col = hidden index

Use one GPU buffer only.

### Attention Weights

Per layer files:

- `self_attn_q_proj.bin`
- `self_attn_k_proj.bin`
- `self_attn_v_proj.bin`
- `self_attn_o_proj.bin`
- `self_attn_q_norm.bin`
- `self_attn_k_norm.bin`

Packed QKV order should stay:

- Q first
- then K
- then V

Q output width:

- `num_heads_q * head_dim = 4 * 256 = 1024`

K output width:

- `num_heads_kv * head_dim = 1 * 256 = 256`

V output width:

- `256`

So packed QKV width is:

- `1536`

### MLP Weights

Per layer files:

- `mlp_gate_proj.bin`
- `mlp_up_proj.bin`
- `mlp_down_proj.bin`

Pack in this order:

- gate
- up
- down

Use the existing GLU kernels as reference, but move them into `Gemma3NextCS.compute`.

### Hidden Norm Weights

Per layer files:

- `input_layernorm.bin`
- `post_attention_layernorm.bin`
- `pre_feedforward_layernorm.bin`
- `post_feedforward_layernorm.bin`

Final model norm:

- `norm.bin`

## Exact Shapes To Use Internally

Keep internal shapes explicit and simple.

### Preferred Unbatched Shapes

- tokens: `[L]`
- hidden: `[L, H]`
- Q: `[Lq, Hq, D]`
- K: `[Lk, Hkv, D]`
- V: `[Lk, Hkv, D]`
- scores: `[Hq, Lq, Lk]`
- attention probs: `[Hq, Lq, Lk]`
- attended values: `[Lq, Hq, D]`
- logits last-token only: `[V]`

Do not over-generalize to full batch support in the first pass.

### Decode-Step Shapes

When decoding one token:

- input token count = `1`
- hidden shape = `[1, H]`
- Q shape = `[1, Hq, D]`
- K/V append one new row into the layer cache
- scores shape = `[Hq, 1, effective_cache_len]`

## Important Performance Guidance

### 1. Do Not Materialize Full Prompt Logits During Generation

This is a major trap.

The old implementation computes logits for all prompt positions, but generation only needs the last row.

For prompt length `2048`:

- full logits = `2048 * 262144` floats
- that is over `536 million` floats
- over `2 GB` just for the logits buffer

So:

- `Generate` and `Chat` should use a last-token lm-head path
- `Predict` may still support full logits because that is its public contract
- but internal generation should never allocate `[L, vocab]` if only the last token is needed

Add a dedicated kernel such as:

- `LmHeadLast`

### 2. Keep Buffers Persistent

Do not create and release large `ComputeBuffer`s every token.

Steady-state decode should reuse:

- hidden buffers
- QKV buffers
- attention scratch
- logits buffer
- sample buffer

Only tiny CPU-side arrays for one token id or one sampled id are acceptable per step.

### 3. Tiny CPU Readback Is Fine

There is one unavoidable boundary:

- the sampled next token id must come back to CPU so the tokenizer can decode it to text and so the coroutine can emit the token string

That is acceptable.

What is not acceptable:

- reading back full hidden states
- reading back Q/K/V
- reading back attention matrices
- reading back logits rows unless the public `Predict` API explicitly asks for them

## Known Traps In The Current Codebase

These are useful warnings so they do not get copied into the new implementation.

### 1. `Tensor.Constant(ComputeBuffer, ...)` Is A Readback

This is the main reason the old path is not fully GPU-resident.

Treat any conversion from `ComputeBuffer` to `Tensor` as a sync point and a readback.

### 2. Current `Gemma3GQA` Is Math Reference Only

It currently does:

- GPU QKV projection
- CPU unpack
- CPU RMSNorm
- CPU RoPE
- CPU mask build
- CPU softmax
- GPU value projection
- GPU output projection

Do not preserve that structure.

### 3. Current Chat Cache Serialization Is Editor-Oriented

The old `InitializeChat` stores cache files under:

- `Assets/Resources/Cache/<hash>`

and uses `AssetDatabase.Refresh`.

That is not a good runtime design for builds.

For the first Gemma3GPU pass:

- it is completely acceptable to keep cache only in memory

If persistence is added later, prefer:

- binary cache files
- or `Application.persistentDataPath`

### 4. Do Not Copy The Old Constructor Path Check

The old code checks the wrong variable with `File.Exists(path)`.

The new implementation should validate:

- parameter directory exists
- tokenizer file exists

### 5. Do Not Copy Incidental Buffer Bugs

Some old helper methods mix buffer size checks incorrectly.

The new implementation should have one clear ownership model:

- one class owns one buffer
- one place resizes it
- no hidden side effects

## Suggested Buffer Strategy

Use a ping-pong layout for hidden states:

- `hiddenA`
- `hiddenB`
- `residual`

Typical block flow:

1. copy `hiddenA -> residual`
2. norm into `hiddenB`
3. attention reads `hiddenB`, writes `hiddenB` or `hiddenA`
4. norm result
5. add residual
6. repeat same pattern for MLP

Do not allocate fresh buffers per layer.

## Suggested KV Cache Strategy

### Full Attention Layers

Use append-only layout:

- K cache buffer capacity `32768 * Hkv * D`
- V cache buffer capacity `32768 * Hkv * D`

Metadata:

- `cachedTokenCount`

Absolute token position for RoPE is always the global decoded position, not the local cache index.

### Sliding Layers

Use ring-buffer layout:

- capacity `512 * Hkv * D`
- write cursor modulo `512`
- track logical length separately

When computing attention:

- map logical key positions to physical ring-buffer positions
- keep absolute positions for masking and RoPE logic separate from physical storage location

This is one of the hardest parts of the implementation.

Do not conflate:

- absolute token position
- logical cache order
- physical ring-buffer slot

## Recommended Kernel Boundaries

Do not over-fuse the first version.

Prefer correct small kernels over one giant unreadable kernel.

Reasonable first split:

- embedding
- hidden RMSNorm
- QKV projection
- split QKV
- Q/K head RMSNorm
- RoPE
- cache write
- score build
- mask
- softcap
- softmax
- attend values
- O projection
- residual add
- MLP gate/up
- MLP down
- final norm
- lm-head last
- argmax

After correctness is proven, fuse if needed.

## Suggested HLSL Conventions

Keep flat index helpers at the top of the shader.

Suggested helpers:

- hidden index
- q index
- k index
- v index
- cache index
- score index
- attn prob index
- attended value index
- logits index

Be explicit about whether a dimension is:

- current query length
- total key length
- cache capacity
- effective cache length

Ambiguous names are how cache bugs get introduced.

## Numerical Guidance

### RMSNorm

Use float accumulation even if the stored weights stay fp32.

### Softmax

Always subtract row max first.

Suggested row sequence:

1. compute max
2. compute exp sum
3. normalize

For masked entries:

- use a large negative number before softmax

### Attention Softcap

Current config has:

- `ATTN_LOGIT_SOFTCAPPING = null`

Still implement the hook so future checkpoints do not require a redesign.

If enabled, the old code does:

1. divide scores by softcap
2. apply `tanh`
3. multiply by softcap

## Sampling Guidance

### First Working Version

Use greedy decode only:

- `temperature = 0`
- GPU argmax kernel

This gets the main path working faster.

### Second Version

Add:

- temperature scaling
- top-k
- top-p
- min-p

If this takes too much time, it is acceptable to:

- keep logits on GPU
- read back only the final logits row for sampling on CPU

But only for non-greedy mode.

Greedy mode should stay fully GPU except for the final sampled token id readback.

## API Guidance

### `Predict`

This is the one method where returning a `Tensor` may force a readback.

That is fine.

Recommended split:

- internal GPU method for full forward returning a logits buffer
- internal GPU method returning last-token logits only
- public `Predict` wraps the full forward and converts to `Tensor`
- `Generate` and `Chat` call the last-token path

### `Generate` / `Chat`

Preserve the coroutine style.

Yield between major steps so Unity stays responsive:

- after prefill
- maybe every few layers during long prefill
- after token sampling

Do not yield between tiny utility kernels during decode unless profiling shows a hitch.

## Validation Suggestions

Use deterministic greedy prompts first.

### Prompt 1

Simple arithmetic:

- user: `3*7?`

Expected use:

- compare last logits and sampled token

### Prompt 2

Short chat continuation:

- user: `Who are you?`

Expected use:

- compare first 16 generated tokens greedily against old Gemma3

### Prompt 3

Long prompt past the sliding window threshold:

- synthetic prompt near 600 to 700 tokens

Expected use:

- verify sliding-window layers still match
- verify full-attention layers still see the whole prefix

### Prompt 4

Very short decode loop:

- prefill a short prompt
- decode 8 tokens one-by-one
- compare against one-shot full-sequence greedy continuation

This catches most cache bugs.

## Memory Reality Check

Rough numbers for fp32:

- tied embedding / lm-head: `262144 * 640 * 4` bytes = about `671 MB`
- total model weights: about `268M params` = about `1.0 GB`

This means:

- unnecessary duplicate buffers are expensive
- full logits for long prompts are unacceptable
- caches and scratch must be designed consciously

If memory gets tight, the first place to optimize is:

- not materializing prompt-wide logits during generation

## Final Recommendation

If there is a choice between:

- matching the existing external behavior exactly
- and preserving a questionable internal implementation detail

choose:

- exact external behavior
- mathematically equivalent internals
- clean GPU-native execution

The point of this rewrite is not to imitate the current plumbing. It is to reproduce the same model behavior without the CPU bottlenecks.
