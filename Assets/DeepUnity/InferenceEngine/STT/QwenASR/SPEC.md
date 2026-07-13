# Qwen3-ASR (0.6B / 1.7B) — DeepUnity port spec (D0)

Frozen from: `Qwen/Qwen3-ASR-{0.6B,1.7B}-hf` checkpoints (config.json + safetensors headers, verified
locally in `C:/dev/_model_staging/qwen3asr/`), the transformers `qwen3_asr` sources
(`modeling_qwen3_asr.py`, `feature_extraction_qwen3_asr.py`, `processing_qwen3_asr.py`, HF main,
2026-07-11), and the original `QwenLM/Qwen3-ASR` library (`qwen_asr/inference/qwen3_asr.py`,
`inference/utils.py`). Everything below is code-verified, not model-card prose. Release 2026-01-29,
Apache-2.0 (see LICENSE_CHECK.md). Tech report: arXiv 2601.21337.

## §0 Pipeline

```
mic PCM 16 kHz mono ──▶ §1 log-mel 128 ──▶ §2 audio encoder (windowed transformer, 13 tok/s)
   ──▶ §3 projector (→ LLM hidden) ──▶ §4 Qwen3 decoder (audio embeds spliced into prompt)
   ──▶ greedy decode ──▶ "language {L}<asr_text>{transcript}" ──▶ §5 parse
```

Sizes (all dims from config.json; decoder = **stock Qwen3** — NOT Qwen3.5, no DeltaNet, no hybrid):

| | 0.6B (`Qwen3-ASR-0.6B-hf`) | 1.7B (`Qwen3-ASR-1.7B-hf`) |
|---|---|---|
| encoder d_model / layers / heads / ffn | 896 / 18 / 14 / 3584 | 1024 / 24 / 16 / 4096 |
| decoder hidden / layers / interm | 1024 / 28 / 3072 | 2048 / 28 / 6144 |
| decoder GQA (Q/KV × head_dim) | 16 / 8 × 128 | 16 / 8 × 128 |
| params / bf16 file | ~0.78 B / 1.56 GB | ~2.04 B / 4.08 GB |

## §1 Mel frontend (`Qwen3ASRFeatureExtractor`)

- sr **16000**, **n_fft 400**, **hop 160**, **hann window** (torch default = *periodic*, length 400),
  `torch.stft` with default **center=True → reflect-pad n_fft/2=200 samples both sides**.
- Frames: `1 + floor(len/160)`, then the **last frame is dropped** (`stft[..., :-1]`) → `floor(len/160)`
  frames; 1 s = 100 frames (100 fps).
- Power spectrum `|STFT|²` (201 bins) → mel filterbank **128 bins, slaney-norm + slaney-scale,
  fmin 0, fmax 8000** (matmul `[201,128]ᵀ`). **Export the filterbank matrix as a tensor**
  (`frontend/mel_filters`, [201,128]) — do NOT reimplement slaney in C#.
- `log10(clamp(mel, 1e-10))` → dynamic-range clamp `max(log_spec, log_spec.max() − 8.0)`
  (**global max over the entire clip** — fine for push-to-talk where the utterance is complete;
  a true-streaming mel would not know the future max) → `(x + 4) / 4`.
- Audio shorter than **8000 samples (0.5 s) is zero-padded to 8000** (`min_length`, no mask adjustment —
  intentional, matches training).
- Mel time axis **right-padded with 0 to a multiple of 100** (= 2·n_window); `input_features_mask`
  marks the valid frames (raw-sample mask downsampled `[::160]`).
- dither = 0.0 (off) at inference.

## §2 Audio encoder (`Qwen3ASREncoder`, model_type `qwen3_asr_encoder`)

Whisper-lineage transformer (from Qwen3-Omni's audio tower) with a **per-chunk conv frontend** and
**block-diagonal windowed attention**. No RoPE; no cross-chunk positional encoding.

### 2.1 Conv frontend — per 1-second chunk, NOT continuous
Split padded mel `[128, T]` into `T/100` chunks `[128, 100]`. Each chunk **independently**
(conv padding is per-chunk zero-pad, p=1):
```
[1,128,100] ─conv2d1 (1→480, k3 s2 p1)+GELU─▶ [480,64,50]
            ─conv2d2 (480→480, k3 s2 p1)+GELU─▶ [480,32,25]
            ─conv2d3 (480→480, k3 s2 p1)+GELU─▶ [480,16,13]
permute→[13, 480·16=7680] ─conv_out Linear(7680→d, NO bias)─▶ [13, d]
+= PE[0:13]      # sinusoidal, RESETS every chunk
```
- All conv2d have bias. GELU is **exact (erf)** — `activation_function: "gelu"` → ACT2FN erf-GELU,
  not GPT-2's tanh approx. (T3 uses tanh-GELU — do not reuse blindly; add an erf variant.)
- `SinusoidsPositionEmbedding(length=13, channels=d)`: Whisper-style **concat** layout
  `[sin | cos]` (half channels each), `inv_timescales = exp(−ln(10000)·i/(d/2−1))`,
  `PE[p, i] = sin(p·inv[i])`, `PE[p, d/2+i] = cos(p·inv[i])`. Bake as tensor `enc/pos_emb` [13, d].
- **1 second of audio ⇒ exactly 13 encoder tokens (13 Hz).**
- Partial final chunk (r = T_valid % 100 mel frames): post-CNN valid length = `ceil3(r)` where
  `ceil3` applies `l → floor((l−1)/2)+1` three times (0 stays 0). Keep only the valid positions,
  drop the rest. ⚠ Python floor-div on negatives ≠ C# truncation: handle `r == 0 → 0` explicitly.
- Audio token count for a clip with `T` valid mel frames:
  `N = (T/100)·13 + ceil3(T%100)` — the C# prompt builder needs this same formula (§5).

### 2.2 Windowed transformer
Valid post-CNN positions from all chunks are packed into one flat `[N_tok, d]` sequence.
Attention is **non-causal, block-diagonal**: windows of `13 · (n_window_infer / (2·n_window))
= 13 · (800/100) = 104 tokens` (= **8 s of audio**), last window = remainder. No attention across
windows. (Reference builds `cu_seqlens`; in Unity just loop attention per window, or mask.)

Per layer (pre-LN, 18× or 24×):
```
x += out_proj( MHA( LN_attn(x) ) )        # q/k/v/out Linear ALL WITH bias
x += fc2( GELUerf( fc1( LN_ffn(x) ) ) )   # fc1: d→ffn, fc2: ffn→d, both with bias
```
- heads 14 (0.6B) / 16 (1.7B), head_dim = d/heads = **64**, scale = 64^-0.5, softmax in fp32.
- LayerNorm (with bias) — same kernel family as T3/GPT-2, not RMSNorm.
- Final `ln_post` LayerNorm, then §3. (An fp16-only clamp to ±(65504−1000) exists in reference;
  DeepUnity fp32 activations ⇒ skip.)

## §3 Projector (`Qwen3ASRMultiModalProjector`)

```
audio_embeds = Linear₂( GELUerf( Linear₁(enc_out) ) )   # d→d, then d→decoder_hidden; both with bias
```
Output `[N_tok, hidden]` is **scattered into the decoder's input embeddings at the `<|audio_pad|>`
positions** (one embed per placeholder token, in order). Everything else is the ordinary token
embedding lookup.

## §4 Decoder — stock Qwen3 (`model_type: "qwen3"`)

Per size: 28 layers; hidden 1024/2048; GQA **16 Q / 8 KV heads, head_dim 128**
(q_proj: hidden→2048; k/v_proj: hidden→1024; o_proj: 2048→hidden; **no bias anywhere**);
SwiGLU MLP gate/up/down (interm 3072/6144, SiLU); RMSNorm eps **1e-6** (input + post_attention +
final); **QK-norm**: RMSNorm(128) applied per-head to q and k **before RoPE** (same as Qwen3.5's
q_norm/k_norm); RoPE **full head_dim (128), θ = 1,000,000**, `rope_type: default`, positions
0..N over the whole spliced sequence (audio embeds occupy normal positions); causal;
**tied embeddings** (no lm_head tensor in checkpoint); vocab **151936**; max_position 65536.

Differences vs the existing Qwen3.5 Unity path — all *simplifications*:
no DeltaNet (all 28 layers full attention), no attn output gate, full RoPE instead of partial-0.25,
θ 1e6 not 1e7, head_dim 128 not 256, vocab 151936 not 248320. QK-norm and GQA kernels reuse as-is.

## §5 Prompt format, special tokens, context injection

Token ids (Qwen2 BPE tokenizer, tokenizer.json, verified):

| token | id | | token | id |
|---|---|---|---|---|
| `<\|endoftext\|>` | 151643 (eos₂/pad) | | `<\|audio_start\|>` | 151669 |
| `<\|im_start\|>` | 151644 | | `<\|audio_end\|>` | 151670 |
| `<\|im_end\|>` | 151645 (eos₁) | | `<\|audio_pad\|>` | 151676 (audio placeholder) |
| `<asr_text>` | 151704 (special) | | | |

Prompt (chat_template.jinja, exact — note the template **drops all user text**, audio only):
```
<|im_start|>system\n{SYSTEM_TEXT}<|im_end|>\n<|im_start|>user\n<|audio_start|>{<|audio_pad|> × N}<|audio_end|><|im_end|>\n<|im_start|>assistant\n
```
`N` = audio token count from §2.1. Two ground-truth usage patterns:

1. **HF processor** (`apply_transcription_request`): `SYSTEM_TEXT` = full language name
   (`"English"`, `"Romanian"`, … 30 languages) or empty for auto language-ID.
2. **Original qwen_asr library** (richer — the one to port): `SYSTEM_TEXT` = **context string**
   ("context injection": free text biasing vocabulary/names/jargon; empty if none), and forcing a
   language is done by **pre-filling the assistant turn** with the string `language {Name}<asr_text>`
   after the generation prompt — the model then emits only the transcript.

Output format: `language {Name}<asr_text>{transcript}` then `<|im_end|>`. Silent/empty audio →
`language None<asr_text>`. Parse = take text after `<asr_text>` (id 151704); language = word after
`"language "` prefix before it. Reference also applies a repetition post-fix (collapse >20 identical
consecutive chars / >20 repeats of a ≤20-char pattern) — cheap CPU string pass, port as-is.

Language names (30): Arabic, Cantonese, Chinese, Czech, Danish, Dutch, English, Filipino, Finnish,
French, German, Greek, Hindi, Hungarian, Indonesian, Italian, Japanese, Korean, Macedonian, Malay,
Persian, Polish, Portuguese, Romanian, Russian, Spanish, Swedish, Thai, Turkish, Vietnamese.

Tokenizer needs in C#: **decode always** (id→string table); **encode only for context injection /
language prefill** (byte-level BPE, same algorithm as existing DeepUnity BPE base; new vocab/merges
export from tokenizer.json — build .vocab.txt/.merges.txt twins like the Chatterbox exporter does).
A fixed no-context prompt needs zero encoding (all scaffold ids are constants).

## §6 Decode loop

- **Greedy** (`generation_config: do_sample: false`), `max_new_tokens` 512 default — 128 is plenty
  for push-to-talk utterances. temperature/top-k/top-p: none (vLLM reference uses temperature 0.0).
- Stop on `<|im_end|>` (151645) or `<|endoftext|>` (151643).
- Prefill = system + scaffold + N audio-embed positions + (optional) forced-language prefix;
  decode from there with ordinary KV cache. Nothing exotic: no forced tokens, no timestamps in the
  ASR path (timestamps are a separate ForcedAligner model — out of scope).

## §7 Push-to-talk plan (Unity)

Reference maxima: single-shot input up to **1200 s** (encoder is linear thanks to 8-s attention
windows; decoder sees 13 tok/s ⇒ 15,600 audio tokens at 20 min — fine in 65k ctx). Reference
"streaming" (vLLM-only) is **re-prefill of ALL accumulated audio every 2 s chunk** with a text-prefix
rollback of the last 5 tokens (first 2 chunks: no prefix) — there is no incremental encoder state.

Unity v1 (recommended): **offline per-utterance**, PTT 2–15 s:
1. `Microphone.Start(null, false, 30, 16000)` → poll `GetPosition` → copy float PCM into a ring;
   on release (or VAD tail-silence), slice the utterance buffer.
2. Mel on GPU (§1) — reuse Chatterbox STFT/mel shaders with n_fft 400 / hop 160 / 128 slaney bins /
   reflect-center padding; log10 + global-max−8 clamp + (x+4)/4; pad to 100-frame multiple.
3. Conv frontend per chunk (§2.1) → windowed encoder (§2.2) → projector (§3) — for ≤15 s this is
   ≤195 tokens through 18/24 layers, single window boundary at 104: trivial GPU cost.
4. Build prompt ids (constant scaffold + N pads), prefill with audio embeds scattered in, greedy
   decode, detokenize, parse after `<asr_text>`.

Unity v2 (optional live captions): replicate the reference rollback scheme — every 2 s re-run
mel+encoder on the full accumulated utterance and re-prefill (audio tokens only grow by 26/chunk),
keep decoded text as prompt prefix minus last 5 tokens. Cost per update ≈ one fresh prefill of
(13·seconds + text) tokens — acceptable ≤15 s; defer until v1 is validated.

## §8 Weights inventory (manifest.tsv, ChatterboxWeights format)

Exporter: `validation/import_qwen3asr.py` (standalone — does NOT touch import_params.py) →
`Assets/Resources/DeepUnity/STT/QwenASR/weights_qwen3asr_{0.6b,1.7b}_{fp16,int8}/`
(manifest.tsv `name\tfile\tdtype\tnumel\tshape`, fp16 packed 2-per-uint at load, q8 4-per-uint +
`.scales` — identical loader contract to Chatterbox; a QwenASRWeights.cs can be a near-copy).

```
frontend/mel_filters            [201,128]            f16 (from librosa-equivalent slaney bank)
enc/pos_emb                     [13,d]               f16 (baked sinusoids)
enc/conv2d{1,2,3}.{w,b}         [480,1|480,3,3]      f16 always (conv, small)
enc/conv_out.w                  [d,7680]             mat (no bias)
enc/layer_{i}/ln{1,2}.{w,b}, attn_{q,k,v,out}.{w,b}, fc{1,2}.{w,b}   norms/bias f16; weights mat
enc/ln_post.{w,b}               f16
proj/linear_{1,2}.{w,b}         mat + f16 bias
dec/embed_tokens/part_{0..15}   [151936/16, hidden]  f16 ALWAYS (tied lm_head; 151936 % 16 == 0)
dec/layer_{i}/{q,k,v,o}_proj    mat;  q_norm/k_norm/input_ln/post_attn_ln f16
dec/layer_{i}/mlp_{gate,up,down} mat
dec/norm                        f16
```
`mat` = fp16, or int8 per-output-row scales under `--quant int8` (repo convention: norms, biases,
embeddings, convs stay fp16 in every mode). Source tensors are bf16 → convert to fp16 (values in
range; same as every other DeepUnity export).

## §9 VRAM / latency (RTX 4060 Laptop 8 GB, anchored to measured BENCHMARK.md numbers)

Weights (computed from checkpoint sizes; int8 = decoder+encoder matmuls, embed/norms fp16):

| | 0.6B fp16 | 0.6B int8 | 1.7B fp16 | 1.7B int8 |
|---|---|---|---|---|
| encoder+proj | 0.37 GB | 0.20 GB | 0.64 GB | 0.34 GB |
| decoder | 1.19 GB | 0.75 GB | 3.44 GB | 2.03 GB |
| **weights total** | **1.56 GB** | **~0.95 GB** | **4.08 GB** | **~2.4 GB** |
| KV @1k ctx (fp16; 28L·8KV·128) | 0.12 GB | 0.12 GB | 0.12 GB | 0.12 GB |
| **runtime total (± scratch)** | **~1.8 GB** | **~1.2 GB** | **~4.4 GB** | **~2.7 GB** |

Co-residency on 8 GB with LLM (~1–2 GB) + TTS (~2 GB): **0.6B fits in any mode**;
**1.7B fp16 does NOT fit** (4.4+2+2 ≈ 8.4 GB → shared-memory spill, known death on this card);
**1.7B int8 fits** (~6.7 GB total, tight but viable). Ship 0.6B as default, 1.7B int8 as quality tier.

Latency, 4060 laptop (anchors: qwen3.5-0.8B fp16 = 134 prefill / 31 decode tok/s; 2B = 46 / 17;
int8 speed-neutral). ASR decoders are the same size class; encoder+mel is <0.3 s for ≤15 s audio:

| utterance | 0.6B est. | 1.7B est. |
|---|---|---|
| 5 s (≈80-tok prefill, ~20-tok out) | **~1.5 s** | ~2.5–3.5 s |
| 15 s (≈210-tok prefill, ~60-tok out) | **~2.5–3.5 s** | ~5–8 s |

13 audio tok/s is the structural win: even 15 s of speech is only 195 decoder positions.

## §10 Parity plan

`validation/dump_reference.py` (torch, HF `Qwen3ASRForConditionalGeneration`) dumps per clip:
`mel.npy` (post-norm, pre-pad-to-100 + padded twin), `enc_out.npy` (post-ln_post),
`proj_out.npy`, `logits_step0.npy` (first decode step), `tokens_greedy.npy`, `transcript.txt`.
Unity probes then compare stage-by-stage (corr > 0.99 per stage, Chatterbox precedent).
Watchouts baked into the dumps: erf-GELU (not tanh), reflect-pad STFT, global-max mel clamp,
per-chunk conv boundaries, 104-token attention windows, QK-norm-before-RoPE order.
