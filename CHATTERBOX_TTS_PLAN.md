# Chatterbox-Turbo TTS on DeepUnity — Full-GPU Implementation Plan

> Handoff doc. Goal: implement Resemble AI **Chatterbox-Turbo** (streaming TTS) fully on GPU inside
> DeepUnity, reusing the existing transformer/quant/dispatch stack and building the missing audio-DSP
> shaders. Target: real-time streaming (LLM text stream → audio), baked-voice first, cloning later.

---

## 1. Model architecture (chatterbox-turbo, confirmed from source)

Repo: `resemble-ai/chatterbox`, module tree under `src/chatterbox/models/`. License **MIT**. English-only turbo.

Pipeline:

```
text ─▶ EnTokenizer ─▶ T3 (Llama decoder) ─▶ speech tokens @25Hz
                          ▲ speaker_emb (+emotion)
speech tokens ─▶ S3Gen: upsample-encoder(25→50Hz) ─▶ 1-step flow-matching(mel) ─▶ HiFTGenerator vocoder ─▶ 24kHz wav
                                                          ▲ x-vector (CAMPPlus)   ▲ prompt mel/tokens
cloning (one-shot): ref audio(16kHz) ─▶ S3Tokenizer(→tokens) + VoiceEncoder(LSTM→speaker_emb) + CAMPPlus(→x-vector)
```

**Turbo specifics that make real-time easy:** 350M streamlined build; the token→mel decoder is
**distilled from 10 flow steps to 1** (`s3gen/utils/intmeanflow.py`) → a *single* estimator forward per
chunk, no ODE loop. Speech tokens are only **25Hz**, so T3 autoregression is cheap. Output 24kHz.
Watermarker (Perth) present but **optional — skip for internal use**.

Sample/token rates: VoiceEncoder + S3Tokenizer @ **16kHz**; T3 tokens **25Hz** → upsampled **50Hz** in
S3Gen (mel frame rate); final audio **24kHz** (HiFTGenerator).

Component source files:
- **T3**: `t3/t3.py`, `t3/llama_configs.py`, `t3/modules/{cond_enc,perceiver,learned_pos_emb,t3_config}.py`
- **S3Gen**: `s3gen/{s3gen,flow,flow_matching,decoder,hifigan,f0_predictor,xvector}.py`,
  `s3gen/matcha/{decoder,flow_matching,transformer}.py`, `s3gen/transformer/*` (conformer encoder),
  `s3gen/utils/{mel,intmeanflow,mask}.py`
- **S3Tokenizer**: `s3tokenizer/s3tokenizer.py`
- **VoiceEncoder**: `voice_encoder/{voice_encoder,melspec}.py` (resemblyzer LSTM)
- **Text tokenizer**: `tokenizers/tokenizer.py` (EnTokenizer)

---

## 2. DeepUnity — what already exists and is directly reusable

DeepUnity does full-GPU transformer inference via Unity `ComputeShader` dispatches. The transformer math is
in HLSL (`Assets/Resources/ComputeShaders/`), C# just orchestrates (cache kernel indices, set uniforms,
bind persistent `ComputeBuffer`s, `Dispatch` with `ceil(N/threads)` groups, **no CPU readback on the hot path**).

**Reuse verbatim (Gemma3 stack is the base — it's a dense decoder; Qwen3.5 is hybrid DeltaNet, ignore it):**

| Need | Reuse from |
|---|---|
| Embedding lookup | `Gemma3CS:EmbeddingLookup` |
| RMSNorm | `Gemma3CS:RmsNormHidden/Head`, `RMSNormGated` |
| QKV proj + attention | `QKVProj`, `SplitQKV`, `FlashAttention` (+ `ComputeAttentionScores`/`ApplyMask`/`SoftmaxRows`/`AttendValues`) |
| RoPE (full NeoX rotary = Llama) | `ApplyRopeSplitHalf` |
| KV cache (GPU, FP16/INT8, full+sliding) | `WriteCacheFull/Sliding` + `KVCache.hlsl` |
| SwiGLU MLP (silu) | `GateUp`/`Down` with `activation_type=0` |
| LM head + sampling | `LmHeadPredict`, `ArgMax`, `SampleToken`, `ApplyRepetitionPresencePenalty` |
| Activations (inline) | `gelu`, `silu`, `safe_tanh`, sigmoid-as-`1/(1+exp)` |
| Matmul (generic) | `TensorCS:MatMul/BatchedMatMul` |
| Conv2d + ConvTranspose2d (fwd) | `Conv2DCS`, `ConvTranspose2DCS` (can drive with H=1 as stopgap 1D) |
| Weight quant FP16/INT8/INT4 + packer | `import_params.py` + `readW/wScale` macros |
| Buffer-persistent dispatch idiom | `Gemma3Model.cs` (copy this pattern) |
| Streaming per-token generation + tap | see §4 |
| Final GPU→CPU readback | `AsyncGPUReadback` (per `LLM/RESEARCH.md`, never `GetData` on hot path) |

**Quant scheme (reuse for T3 + all S3Gen weights):** weights packed into `StructuredBuffer<uint>`, keywords
`INT8_WEIGHTS`/`INT4_WEIGHTS`. FP16 = 2 halves/uint; INT8 = symmetric 4/uint + FP16 scale per output row;
INT4 = Q4_0 (8 nibbles/uint + FP16 scale per 32-group). Embedding/LM-head always FP16.

---

## 3. DeepUnity — what is MISSING (must be built, all GPU)

| Op | Status | Needed by |
|---|---|---|
| **FFT / IFFT (complex, `float2`, twiddles)** | MISSING | STFT/iSTFT foundation |
| **STFT / iSTFT (Hann window, framing, overlap-add)** | MISSING | vocoder (n_fft=16/hop=4), mel analysis |
| **Mel filterbank + log-mel** | MISSING | S3Gen mel prompt, S3Tokenizer, VoiceEncoder |
| **General Conv1d (in/out ch, stride, dilation, pad, groups)** | MISSING (only depthwise-causal k=4 in Qwen3_5CS) | vocoder, flow estimator, conformer conv, TDNN |
| **ConvTranspose1d (upsample)** | MISSING (only 2D) | HiFTGenerator upsampling |
| **Snake activation** `x + sin²(αx)/α` (trainable α) | MISSING | HiFTGenerator ResBlocks |
| **LeakyReLU (named GPU kernel)** | trivial/MISSING | vocoder |
| **NSF source module** (SineGen: F0 upsample + cumsum-phase + sin + noise) | MISSING | HiFTGenerator excitation |
| **Flow-matching step (1-step meanflow)** | MISSING | S3Gen (turbo = single forward, no ODE loop) |
| **LSTM cell** | MISSING (only vanilla RNNCell) | VoiceEncoder (cloning) — or run on CPU |
| **FSQ/VQ quantizer + stats-pooling** | MISSING | S3Tokenizer / CAMPPlus (cloning) — or CPU |
| **Audio I/O (AudioClip / PCM / WAV, ring buffer)** | MISSING (greenfield) | output bridge |

---

## 4. The real-time tap point (already streaming)

Generation is a Unity coroutine that `yield`s **one token per frame** and already frame-budgets everything.
In `Gemma3.cs`/`Qwen3_5.cs` `Generate`/`Chat`, each step computes the raw sampled id into local `sampled[0]`
**right before** `tokenizer.Decode`:

```csharp
tokenId = sampled[0];                       // ← RAW next-token id  (TTS TAP)
if (tokenId == EOS) break;
onTokenGenerated?.Invoke(tokenizer.Decode(...)); // text callback (skip for speech tokens)
yield return null;                          // one token per frame
```

For T3: add `Action<int> onSpeechToken`, invoke it with `sampled[0]`, skip `tokenizer.Decode` (speech tokens
aren't text). The existing per-token `yield` lets a streaming vocoder be pumped in real time on the same
frame loop — no threading needed, single-GPU interleaving is natural.

---

## 5. Component build plan (Chatterbox module → DeepUnity work)

**Hot path (per utterance, must be GPU/fast):** T3, S3Gen encoder + estimator + vocoder.
**One-shot per voice (can be CPU, or precomputed offline):** VoiceEncoder, S3Tokenizer, CAMPPlus, ref-mel.

### 5.1 Text tokenizer (EnTokenizer)
Port to C# alongside `Base/BPETokenizer.cs`. CPU, cheap. English-only → simple.

### 5.2 T3 (Llama-style decoder) — **mostly reuse**
Clone `Gemma3/` → `T3/`: `T3Config.cs` (Llama dims, single global RoPE theta, **2 norms/layer**, silu,
**untied lm_head**, speech-token vocab), `T3Model.cs` (strip Gemma extras: drop q/k-norm, embed `sqrt(h)`
scale, dual-RoPE, sliding-window, the 4-norm sandwich), `T3Weights.cs` (separate `lm_head` buffer — embedding
is untied), `T3Cache.cs` (uniform K/V, reuse as-is). Reuse `Gemma3CS.compute` kernels (or fork `T3CS.compute`).
Add conditioning: `cond_enc` (perceiver resampler = attention, reusable) + `learned_pos_emb`; speaker_emb +
emotion injected as prefix conditioning. Add `Action<int> onSpeechToken` tap (§4).

### 5.3 S3Gen
- **Upsample-encoder (25→50Hz), conformer** (`s3gen/transformer/*`): attention (reuse) + **Conv1d** module + FF.
- **Flow-matching estimator** (`matcha/decoder.py`, 1D U-Net): **Conv1d** resnet blocks + down/up + transformer
  mid + step-embedding. Turbo = **single forward** (`intmeanflow`), no Euler loop. Needs Conv1d, norm, silu/snake, attention.
- **CAMPPlus x-vector** (`xvector.py`): TDNN = dilated **Conv1d** stack + stats-pooling + linear. One-shot → CPU ok.
- **Ref mel** (`utils/mel.py`): **STFT + mel + log**. One-shot (cloning) → CPU/precompute ok.
- **HiFTGenerator vocoder** (`hifigan.py`) — **biggest single build**: input `Conv1d(80→512,k7)`; upsample
  `ConvTranspose1d` rates [8,8] kernels [16,16]; ResBlocks kernels [3,7,11] dilations [1,3,5]; **Snake** + LeakyReLU;
  **SourceModuleHnNSF** (F0 predictor → SineGen harmonic sines w/ cumsum phase + noise, nb_harmonics=8);
  **STFT + iSTFT** n_fft=16 hop=4 (magnitude/phase). Output 24kHz. Fold `weight_norm` at export.

### 5.4 S3Tokenizer (cloning, one-shot)
Conv+transformer encoder + FSQ/VQ, mel input, 16kHz→25Hz tokens. Run **on CPU** (one-shot) or GPU later.

### 5.5 VoiceEncoder (cloning, one-shot)
Resemblyzer **LSTM** over mel → speaker_emb. Run **on CPU** to avoid building a GPU LSTM in v1.

### 5.6 Watermarker (Perth) — skip.

> **Scope collapse for v1 (baked voice):** precompute speaker_emb + x-vector + prompt tokens/mel **offline in
> Python**, ship as an asset. Runtime then needs only: text tokenizer + T3 + S3Gen(encoder+estimator+vocoder).
> This removes S3Tokenizer, VoiceEncoder, CAMPPlus, and the analysis-side STFT/mel from the runtime path.
> The only runtime FFT is the vocoder's tiny internal STFT/iSTFT (n_fft=16).

---

## 6. New compute shaders to author (prioritized)

1. **`ConvCS.compute`** — `Conv1D` (general: in/out ch, kernel, stride, dilation, pad, groups; incl. causal) +
   `ConvTranspose1D`. Foundation for vocoder, estimator, conformer, TDNN. (Fold `weight_norm` at export → plain weights.)
2. **`FFTCS.compute`** — complex FFT/IFFT (radix-2 Stockham, `float2`) + `STFT`/`iSTFT` wrappers (Hann window,
   framing, overlap-add). Start with the vocoder's tiny n_fft=16; generalize for mel later.
3. **`MelCS.compute`** — mel filterbank (precompute basis CPU → GPU matmul) + log; window generation. (Cloning/analysis only.)
4. **Activations** — `Snake` (+ trainable α buffer), `LeakyReLU` (add to a small shader or inline where used).
5. **`NSFCS.compute`** — SineGen: F0 nearest/linear upsample, cumulative-sum phase, sin/cos, noise, source merge.
6. **(cloning, later)** LSTM cell, FSQ quantizer, stats-pooling — or keep on CPU.

Each new kernel follows the `Gemma3Model.cs` idiom (cache index, set uniforms, bind persistent buffers,
`ceil(N/threads)` groups). **Validate every kernel against a PyTorch reference** (dump numpy inputs/outputs)
before composing — flow-matching + FFT are numerically sensitive (cf. int4 Gemma decode collapse).

---

## 7. Weight export tooling

Extend `LLM/import_params.py`:
- `export_t3` — map T3 Llama tensor names; write **untied** `lm_head.weight` (neither current exporter does this).
- `export_s3gen` — conv weights (fold `remove_weight_norm` → plain), ConvTranspose, ResBlocks, Snake α, flow
  estimator, conformer, F0 predictor, CAMPPlus. Store FP16 (quant optional per-tensor).
- Dump **reference intermediate tensors** (per-stage inputs/outputs) for the validation harness.

---

## 8. Streaming handler + Unity audio bridge

Producer→consumer chain, all as frame-yielding coroutines (fits DeepUnity's per-frame budget, single GPU):

```
Qwen (text) ─▶ [chunker: buffer to sentence/clause boundary] ─▶ T3 ─(onSpeechToken, 25Hz)▶ speech-token buffer
   ─▶ S3Gen chunked (causal: CausalMaskedDiffWithXvec) ─▶ mel chunk ─▶ vocoder ─▶ PCM chunk ─▶ ring buffer
   ─▶ AudioClip.Create(stream:true) + PCMReaderCallback ─▶ AudioSource
```

- **Chunker**: don't synthesize per token — buffer to a clause/sentence boundary (prosody needs lookahead).
  First chunk out = time-to-first-audio (TTFA), the metric that matters.
- **Chunked S3Gen**: exploit the causal decoder for chunk-wise synthesis with small overlap for continuity.
- **Audio bridge**: `AsyncGPUReadback` final waveform buffer → thread-safe **ring buffer** → `PCMReaderCallback`.
  Do **not** assemble whole AudioClips.
- **Single-GPU**: interleave LLM decode and TTS (both yield per frame). Budget VRAM: Qwen (int8/int4) +
  Chatterbox (fp16, 350M). Keep RTF < ~0.5 for playback margin.

---

## 9. Milestones (suggested order for Fable)

- **M0 — Foundations + validation harness.** `ConvCS` (Conv1d/ConvTranspose1d), `FFTCS` (STFT/iSTFT), `MelCS`,
  Snake/LeakyReLU. Unit-test each vs PyTorch numpy dumps.
- **M1 — Vocoder.** Port HiFTGenerator; validate mel→wav against reference on a dumped mel. Proves the whole DSP stack offline.
- **M2 — S3Gen flow.** Upsample-encoder + 1-step estimator; validate speech-tokens→mel vs reference.
- **M3 — T3.** Clone Gemma3→T3, `export_t3`, validate text→speech-tokens (greedy) vs reference.
- **M4 — Baked-voice offline TTS** end-to-end in Unity (text→wav, non-streaming). Conditioning precomputed in Python.
- **M5 — Streaming.** Chunker + ring buffer + `PCMReaderCallback`; T3 `onSpeechToken` tap; chunked S3Gen. Measure TTFA + RTF.
- **M6 — Cloning (optional).** S3Tokenizer + VoiceEncoder + CAMPPlus (CPU ok) for runtime voice capture.
- **M7 — LLM→TTS handler (optional).** Wire Qwen's text stream into the chunker for live speech.

---

## 10. Risks / open items

- **Numerical fidelity**: FFT + 1-step meanflow are sensitive; validate tensor-by-tensor vs reference.
- **weight_norm folding** at export (HiFiGAN) — must `remove_weight_norm` → plain conv weights.
- **Conv2d-with-H=1** is a valid stopgap but inefficient for the vocoder hot path; a real Conv1d is worth building first.
- **Single-GPU VRAM** for Qwen + Chatterbox coexisting — budget with quant; turbo's 350M helps.
- **Untied lm_head** for T3 — new code path in both model + exporter.
- **Watermarking** skipped — fine internally; reconsider if distributing (MIT license, but Resemble ships Perth by default).
