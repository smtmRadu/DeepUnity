# Parakeet-TDT 0.6B (v2 / v3) — DeepUnity port spec (E0)

Frozen from: `nvidia/parakeet-tdt-0.6b-v3` HF-native checkpoint (config.json / processor_config.json /
generation_config.json / tokenizer.json, fetched 2026-07-11), the transformers `parakeet` sources
(`modeling_parakeet.py`, `generation_parakeet.py`, `feature_extraction_parakeet.py`,
`convert_nemo_to_hf.py`, HF main 2026-07-11), and NeMo `rnnt_greedy_decoding.py::GreedyTDTInfer`
(canonical TDT loop, cross-checked). Everything below is code-verified, not model-card prose.
v2 = English-only (.nemo-only repo, converted via the official `convert_nemo_to_hf.py`);
v3 = 25 European languages **including Romanian**. Both CC-BY-4.0 (see LICENSE_CHECK.md).
Items marked **[VERIFY]** are confirmed against actual checkpoint tensors during E0 validation runs.

Papers: FastConformer arXiv 2305.05084 · TDT arXiv 2304.06795 (Xu et al., token-and-duration transducer).

## §0 Pipeline

```
mic PCM 16 kHz mono ──▶ §1 log-mel 128 (+ per-feature norm) ──▶ §2 subsample ×8 (conv, 12.5 fps)
   ──▶ §3 FastConformer encoder ×24 (rel-pos MHSA, full attention) ──▶ enc_proj 1024→640
   ──▶ §5/§6 TDT greedy loop { LSTM pred-net + joint head → token + duration, frame-skipping }
   ──▶ §7 detokenize (SentencePiece BPE, decode-only) ──▶ text (+ punctuation, capitalization,
        free per-token timestamps = cumsum(durations) × 80 ms)
```

One architecture, two checkpoints (all dims identical except the vocab):

| | v2 (`parakeet-tdt-0.6b-v2`) | v3 (`parakeet-tdt-0.6b-v3`) |
|---|---|---|
| languages | English (best-in-class WER 6.05%) | 25 EU langs incl. **ro** (leaderboard 6.34%) |
| vocab (incl. blank) | 1025 — VERIFIED (.nemo labels=1024) | 8193 (8192 BPE + `<blank>`=8192) |
| joint head out | 1025 + 5 = 1030 | 8193 + 5 = 8198 |
| params / fp32 ckpt | ~618 M / 2.47 GB | ~627 M / 2.51 GB |
| checkpoint format | `.nemo` (tar) only → convert | HF-native safetensors + `.nemo` |
| durations / blank id | [0,1,2,3,4] / 1024 — VERIFIED | [0,1,2,3,4] / 8192 (config.json) |

Everything in §1–§4 (frontend + encoder) is bit-identical between v2 and v3 — CONFIRMED from the
v2 `.nemo` `model_config.yaml` (conversion run 2026-07-11): `use_bias: False`, `xscaling: False`,
n_layers 24, d_model 1024, prednet {pred_hidden 640, pred_rnn_layers 2}, preprocessor identical
(0.025/0.01 s windows, n_fft 512, 128 feats, per_feature norm); conversion mapped with ZERO
missing/unexpected keys and both exports produced identical 653-entry manifests modulo vocab.

## §1 Mel frontend (`ParakeetFeatureExtractor`, processor_config.json)

- sr **16000**, **n_fft 512**, **win_length 400**, **hop 160**, **128 mel bins**, preemphasis **0.97**.
- Pre-emphasis over the raw clip first: `y[0]=x[0]; y[t]=x[t]−0.97·x[t−1]` (padding region re-zeroed).
- STFT: `torch.stft(n_fft=512, hop=160, win_length=400, window=hann(400, periodic=False),
  center=True, pad_mode="constant")` →
  - **hann is SYMMETRIC** (`periodic=False`) — NOT the periodic hann Chatterbox/Qwen-ASR use;
  - the 400-sample window is **zero-padded centered to 512** (torch.stft rule for win < n_fft:
    56 zeros left, 56 right);
  - center pad = **256 zeros** both sides (HF uses constant/zero; NeMo uses reflect — a first/last-
    frame-only difference that per-utterance normalization washes out; we mirror HF = our reference).
- Power spectrum `|STFT|²` (257 bins) → mel filterbank `librosa.filters.mel(sr=16000, n_fft=512,
  n_mels=128, fmin=0, fmax=8000, norm="slaney")` (slaney scale + slaney norm) →
  **export as tensor `frontend/mel_filters` [128,257]** — do NOT reimplement slaney in C#.
  The 400-pt symmetric hann is exported too (`frontend/window` [400]) for bit-exact trig.
- `log(mel + 2^−24)` (natural log, guard value exact).
- Valid frames: `T = floor(len_samples / 160)` (mask formula `(len + 2·(n_fft/2) − n_fft) // hop`);
  torch.stft actually emits `floor(len/160)+1` frames — the **last frame is masked off**, drop it.
  1 s = 100 frames.
- **Per-feature normalization over the utterance** (the reason the clip must be complete before
  encoding — fine for push-to-talk): per mel bin over valid frames,
  `mean_f = Σx/T`, `var_f = Σ(x−mean)²/(T−1)` (unbiased), `x = (x − mean)/(std + 1e-5)`,
  then padded frames zeroed. dither=0 at inference.

## §2 Subsampling ×8 (`ParakeetEncoderSubsamplingConv2D`, NeMo `dw_striding`)

Input `[1, 1, T, 128]` (mel as 2-D image). All convs 3×3, stride 2, pad 1, **WITH bias**:

```
conv0: Conv2d(1→256, 3×3, s2, p1) + ReLU                     # [256, T/2, 64]
conv1: dw Conv2d(256, 3×3, s2, p1, groups=256) + pw Conv2d(256→256, 1×1) + ReLU   # [256, T/4, 32]
conv2: dw Conv2d(256, 3×3, s2, p1, groups=256) + pw Conv2d(256→256, 1×1) + ReLU   # [256, T/8, 16]
reshape: [256, T', 16] → transpose → [T', 256·16 = 4096]      # CHANNEL-major, freq inner
linear:  4096 → 1024 (bias)                                   # [T', 1024]
```
- Per-stage length: `L' = floor((L−1)/2) + 1` (= `(L + 2·1 − 3)//2 + 1`); freq 128→64→32→16.
- Encoder frame rate: **12.5 fps → one encoder frame = 80 ms**. 15 s clip → T'=187.
- `scale_input=false` → no ×√d after the linear.

## §3 Relative positional encoding (Transformer-XL style, `RelPositionalEncoding`)

- Positions `p = T'−1, T'−2, …, 1, 0, −1, …, −(T'−1)` (length 2T'−1).
- `inv_freq[i] = 10000^(−2i/1024)`, i = 0..511 (over d_model=1024, NOT head_dim).
- `pos_emb[p] = interleave(sin(p·inv_freq), cos(p·inv_freq))` → `[2T'−1, 1024]`,
  layout `[s0,c0,s1,c1,…]` (matches NeMo `pe[:,0::2]=sin, pe[:,1::2]=cos`).
- Computed once per utterance (fp32), shared by all 24 layers. Bake-or-compute: compute in C#
  (cheap), parity-check against the `pos_emb.npy` dump.

## §4 FastConformer encoder — 24 × ConformerBlock (d=1024, 8 heads × 128, ffn 4096)

**No linear biases anywhere in the block** (`attention_bias=false`, `convolution_bias=false` —
applies to FF linears, q/k/v/o, rel-pos proj, and conv-module convs). All five norms are standard
**LayerNorm(1024)** with weight+bias, pre-norm. Per block:

```
x += 0.5 · FF1( LN_ff1(x) )                 # Linear 1024→4096 (no b) → SiLU → Linear 4096→1024 (no b)
x += MHSA( LN_att(x), pos_emb )             # rel-pos attention, below
x += Conv( LN_conv(x) )                     # conv module, below
x += 0.5 · FF2( LN_ff2(x) )                 # same shape as FF1
x  = LN_out(x)                              # per-block final norm (block 23's IS the encoder output)
```

### 4.1 Rel-pos MHSA (`ParakeetEncoderAttention` = NeMo `RelPositionMultiHeadAttention`)
Same math family as the **already-validated Chatterbox S3Gen RelPosAttention kernel**
(`ChatterboxS3GenCS.compute`): pos_bias_u/pos_bias_v ("bias_u/bias_v"), linear_pos
("relative_k_proj"), Transformer-XL rel-shift.

```
q,k,v = x·Wq, x·Wk, x·Wv                       # [T,8,128] each, no bias
p     = pos_emb · Wpos                         # [2T−1, 8, 128], no bias
AC    = (q + bias_u) · kᵀ                      # content term    [8, T, T]
BD    = rel_shift( (q + bias_v) · pᵀ )[…, :T]  # position term   [8, T, 2T−1] → [8, T, T]
attn  = softmax( (AC + BD) / √128 ) · v        # fp32 softmax
out   = concat(attn) · Wo                      # no bias
```
rel_shift (Transformer-XL appendix B): pad the last axis left by 1 → view `[8, 2T, T]` → drop
row 0 → view `[8, T, 2T−1]`; then slice the first T columns. After the shift, entry `[t, s]`
reads relative position `t−s` (positive = past, negative = future). Full bidirectional attention,
no mask for single-utterance batch-1. **[VERIFY at build-out]** whether the Chatterbox kernel's
shift/slice convention matches exactly (its 2T−1 window vs ESPnet legacy slicing) — the E0 dumps
include per-layer attention inputs to settle this.

### 4.2 Conv module (kernel 9)
```
pw1: Conv1d(1024→2048, k1, no b)  → GLU(dim=ch): a·σ(b)         # [T,1024]
dw : Conv1d(1024, k9, pad 4, groups=1024, no b)                  # depthwise, SAME
bn : BatchNorm1d(1024)  — inference = per-channel affine:        # FOLD AT EXPORT:
       scale = γ/√(running_var+eps), shift = β − running_mean·scale   (eps 1e-5)
       exporter emits conv.bn.scale/.shift [1024] f16; no BN kernel needed
silu → pw2: Conv1d(1024→1024, k1, no b)
```

## §5 Prediction network + joint (`ParakeetRNNTDecoder` / `ParakeetTDTJointNetwork`)

```
embedding : Embedding(V, 640)         # V=8193/1025; row[blank] == 0 (NeMo blank_as_pad,
                                      #   padding_idx=blank) — VERIFIED v3: max|row| = 0.00e+00
lstm      : 2-layer LSTM, input 640, hidden 640, batch_first; torch gate order i,f,g,o;
            weight_ih_l{0,1} [2560,640], weight_hh_l{0,1} [2560,640], bias_ih+bias_hh [2560] each
            h0 = c0 = zeros
pred_proj : Linear(640→640, bias)     # NeMo joint.pred
enc_proj  : Linear(1024→640, bias)    # NeMo joint.enc — precompute for ALL frames once per clip
joint     : head( relu( enc_proj[t] + pred_proj(lstm_out) ) )   # head: Linear(640→V+5, bias)
```
Joint logit layout: `[0..V−2]` = real tokens, `[V−1]` = blank (8192 / 1024), `[V..V+4]` = duration
logits for durations `[0,1,2,3,4]` (duration value == bin index). `generation_config` confirms
`decoder_start_token_id = blank`, suppress ids V..V+4 in the token argmax (we split explicitly).

## §6 TDT greedy decode loop (the C# loop, exact)

Reference semantics = NeMo `GreedyTDTInfer._greedy_decode` (label-history conditioned; HF
`ParakeetTDTGenerationMixin` matches; differences flagged below):

```
t = 0; last = BLANK; (h,c) = zeros; predOut = pred_proj(lstm(embed[BLANK]))   # embed[BLANK]=0
tokens = []; frames = []; durs = []
while t < T_enc:
    symbolsAtFrame = 0
    while true:                                        # inner loop at frame t
        logits = head(relu(encProj[t] + predOut))      # [V+5]
        k  = argmax(logits[0..V-1])                    # token (incl. blank at V-1)
        d  = durations[ argmax(logits[V..V+4]) ]       # 0..4
        if k != BLANK:
            tokens.add(k); frames.add(t); durs.add(d)
            (h,c), lstmOut = lstmStep(embed[k], h, c)  # state advances ONLY on non-blank
            predOut = pred_proj(lstmOut)
            last = k
        symbolsAtFrame += 1
        t += d
        if d > 0: break                                # move to the new frame
        if k == BLANK: { t += 1; break }               # blank+dur0 would spin: force +1  (HF rule)
        if symbolsAtFrame >= 10: { t += 1; break }     # max_symbols_per_step=10 (NeMo guard)
transcript = detokenize(tokens)                        # §7
timestamps[i] ≈ frames[i] · 0.080 s                    # free word/token timestamps
```
- Blank emissions do NOT touch the LSTM (both refs: NeMo keeps `hidden_prime` unused on blank;
  HF reuses the cached pred output). So `predOut` is reused across consecutive blanks — with the
  frame loop above, blanks cost ONE joint head evaluation each, no LSTM.
- Divergence note: on blank+dur0, NeMo burns inner iterations until `max_symbols` then advances 1;
  HF forces dur=1 immediately. Identical transcripts (repeated blank adds no tokens); we adopt the
  HF rule (cheaper). On 10-symbol overflow NeMo advances +1 — we adopt that too.
- Loop length: ≤ ~1.5·T_enc joint evaluations in practice (~90 for 15 s); token count ≈ 2–4/word.
  Measured (v3, 4.6 s clips): T_enc=58 covered in 23–30 steps — the duration head skips hard.
- No EOS concept — decoding ends when the frame pointer walks off T_enc.
- **VALIDATED**: this exact loop (dump_reference.py `manual_tdt_greedy`) reproduces transformers'
  `model.generate()` transcripts verbatim on all E0 clips (`manual_matches_generate: true`).

## §7 Tokenizer — SentencePiece BPE, DECODE-ONLY in C#

We never encode text → no BPE merges, no normalizer needed. Decode = table lookup:
- v3 `tokenizer.json` (HF tokenizers, model type BPE): 8192 vocab entries; ids **0–273 are
  specials** (`<unk>`, `<pad>`=2, `<|endoftext|>`=3, Canary-style task/lang tags `<|en|>`,`<|ro|>`…,
  `<|spltoken*|>`); text tokens carry `▁` word-boundary marks; decoder = **Metaspace**
  (`▁`→space, prepend_scheme "always"). `<blank>` added at 8192 — never emitted as text.
- C# decode rule: for each emitted id → vocab string; skip ids in the specials set (0–273 for v3)
  and blank; concat; replace `▁` with `' '`; trim the single leading space. Byte-fallback tokens
  (`<0xNN>`): **VERIFIED v3 — zero byte tokens in the vocab** → plain per-token string decode; the
  exporter still records the flag in `tokenizer/specials.tsv` (guards v2 too).
- v2: SentencePiece `.model` inside the .nemo (1024 tokens — VERIFIED, 0 byte-fallback); exporter
  emits the same `vocab.txt` (one token per line, id order) + `specials.tsv` so C# has ONE code
  path. Quirk: the CONVERTED v2 tokenizer appends `<pad>`@1024 + `<blank>`@1025 while the MODEL's
  blank logit is 1024 — C# must take blank = vocab_size−1 from config, never look `<blank>` up by
  string (rows 1024/1025 are flagged special in specials.tsv → skipped in decode either way).
- Exporter artifacts per variant, next to the manifest: `tokenizer/vocab.txt`,
  `tokenizer/specials.tsv` (id, content, is_special, is_byte).

## §8 Streaming / push-to-talk plan (Unity)

These checkpoints are **offline full-attention** models (not NeMo's cache-aware streaming
checkpoints). NVIDIA's "streaming" example is buffered chunked re-inference (2 s chunks,
10 s left / 2 s right context) — i.e. re-run, not incremental state. For PTT 2–15 s clips the
whole-utterance offline decode is the correct and simplest mode (24-min single-pass capacity is
irrelevant here; per-feature normalization needs the full clip anyway).

Unity v1 (IMPLEMENTED at E1, `ParakeetSTT.Transcribe`):
1. Game code: `Microphone.Start(null, false, 30, 16000)` → poll `GetPosition` → float PCM ring;
   on PTT release (or VAD tail silence) slice the utterance and call `Transcribe(samples, cb)`.
2. CPU (background task): preemph + STFT(512/400sym/160, zero-center-pad, radix-2 FFT) + mel(128,
   baked filters) + log + per-feature norm — `ParakeetCPU.Mel`. Went CPU instead of the originally
   sketched GPU mel: the whole frontend is ~2 M FLOP-scale (≪1% of the encoder), harness-graded at
   corr 1.000, and it removes every STFT-shader delta risk (symmetric hann, win<n_fft embed).
3. GPU (`ParakeetCS.compute`, one dispatch burst ≈ 450 dispatches): subsampling convs → 24
   conformer blocks → enc_proj → `AsyncGPUReadback` of `[T', 640]` fp32 (~0.5 MB @15 s).
4. CPU (background task): the entire §6 loop — LSTM (2×640, "small-net LSTM = CPU-appropriate"
   repo precedent), pred_proj, joint head (640×8198 ≈ 5.2 M MAC/step), argmax, detokenize. Zero
   GPU dispatches during decode → no dispatch-bound tail, no per-step readbacks.
5. Later (optional): live captions via 2 s chunked re-encode of the accumulated clip (encoder is
   the only GPU cost; decoder restarts each pass) — defer until v1 is validated.

## §9 VRAM / latency (RTX 4060 Laptop 8 GB, anchored to BENCHMARK.md measurements)

Component params (v3; v2 ≈ −9 M in embed+head): subsampling 4.3 M, conformer 24×25.2 = 604.7 M,
enc_proj 0.7 M, pred-net+joint 17.5 M → 627 M total (matches 2.51 GB fp32 ckpt).

| | v2 fp16 | v2 int8 | v3 fp16 | v3 int8 |
|---|---|---|---|---|
| encoder (GPU) | 1.22 GB | ~0.65 GB | 1.22 GB | ~0.65 GB |
| pred+joint (CPU RAM, fp32) | ~36 MB | ~36 MB | ~70 MB | ~70 MB |
| activations @15 s (T'=187) | ~40 MB | ~40 MB | ~40 MB | ~40 MB |
| **GPU total** | **~1.26 GB** | **~0.7 GB** | **~1.26 GB** | **~0.7 GB** |

int8 = the 604 M encoder matmul weights (per-row scales, repo convention; norms/convs-dw/biases/
subsampling stay fp16). No KV cache exists in this architecture. Co-residency on 8 GB with
LLM (~1–2 GB) + TTS (~2 GB): **fits comfortably in every mode** — the lightest STT option on the
table (Qwen3-ASR-0.6B is ~1.8 GB runtime).

Latency (anchor: qwen3.5-0.8B fp16 prefill 134 tok/s ⇒ ~214 GFLOP/s effective; encoder forward =
2·609M·T' FLOPs; decode CPU ~1–2 ms/step):

| utterance | T_enc | encoder | decode (CPU) | **total** | Qwen3-ASR-0.6B ref |
|---|---|---|---|---|---|
| 2 s | 25 | ~0.1 s | ~0.03 s | **~0.2 s** | ~1 s |
| 5 s | 62 | ~0.35 s | ~0.08 s | **~0.5 s** | ~1.5 s |
| 15 s | 187 | ~1.1 s | ~0.2 s | **~1.3 s** | ~2.5–3.5 s |

Structural win vs the LLM-decoder ASR: no autoregressive LLM pass — one encoder sweep + a
CPU-only transducer loop. Expected 2–3× faster end-to-end at equal quality for dictation-style text.

## §10 Weights inventory (manifest.tsv, ChatterboxWeights format)

Exporter: `validation/import_parakeet.py` (standalone; does NOT touch import_params.py) →
`Assets/Resources/DeepUnity/STT/Parakeet/weights_parakeet_tdt_0.6b_{v2,v3}_{fp16,int8}/`
(manifest.tsv `name\tfile\tdtype\tnumel\tshape`; fp16 packed 2-per-uint at load, q8 4-per-uint +
`.scales` f16 sibling — identical loader contract to ChatterboxWeights/CosyVoiceWeights).
Sources: v3 = `model.safetensors` (pure-numpy reader, no torch); v2 = HF-converted checkpoint from
`convert_nemo_to_hf.py` (or directly the `.nemo` tar's `model_weights.ckpt` — the converted path is
preferred so BOTH variants flow through identical HF names).

```
frontend/mel_filters                  [128,257]  f16   (librosa slaney, baked)
frontend/window                       [400]      f16   (symmetric hann, baked)
sub/conv0.{w,b}                       [256,1,3,3]      f16
sub/conv{1,2}_dw.{w,b} / _pw.{w,b}    dw [256,1,3,3], pw [256,256,1,1]   f16
sub/linear.{w,b}                      [1024,4096]      mat
layer_{0..23}/ff1.{ln.w,ln.b}, ff1.lin1.w, ff1.lin2.w       norms f16, weights mat
layer_{0..23}/attn.{ln.w,ln.b, q.w,k.w,v.w,o.w, pos.w, bias_u, bias_v}   biases/norms f16, mats mat
layer_{0..23}/conv.{ln.w,ln.b, pw1.w, dw.w, bn.scale, bn.shift, pw2.w}   dw/bn f16, pw mat
layer_{0..23}/ff2.*                   (same as ff1)
layer_{0..23}/out_ln.{w,b}            f16
dec/embedding                         [V,640]    f16 ALWAYS (row blank == 0)
dec/lstm.{wih0,whh0,bih0,bhh0,wih1,whh1,bih1,bhh1}          f16 (CPU-side, small)
dec/pred_proj.{w,b}                   [640,640]  f16
joint/enc_proj.{w,b}                  [640,1024] mat
joint/head.{w,b}                      [8198,640] f16 (CPU-side; int8 optional later)
tokenizer/vocab.txt + specials.tsv    (id-ordered; §7)
```
`mat` = fp16, or int8 per-output-row under `--quant int8`. CPU-side tensors (dec/*, joint/head)
are read via the loader's `ReadFloats` path, not uploaded.

### C# design (IMPLEMENTED at E1)
- `ParakeetSTT : DeepUnity.STT` — Unity runtime (`InputSampleRate=16000`,
  `Transcribe(float[], Action<string>)` coroutine; residency API forwarded to the loader;
  `Warmup()` = 0.5 s silence through the full pipeline; per-utterance GPU scratch, released after
  each call; single-utterance `busy` gate).
- `ParakeetWeights` — verbatim `CosyVoiceWeights.cs` residency pattern (budgeted UploadPump,
  epoch-guarded Defetch, reload-after-defetch, pooled IO); Parakeet delta: `dec/*`,
  `joint/head.*`, `frontend/*` are CPU-side and never uploaded.
- `ParakeetConfig.cs` — compile-time dims + `ParakeetVariant.V2/V3` (vocab/blank/folder).
- `ParakeetCPU.cs` / `ParakeetTensors.cs` / `ParakeetTokenizer.cs` — PURE C# (no UnityEngine):
  mel frontend + full reference encoder + TDT decode + detok, shared verbatim by the dotnet
  parity harness (`validation/harness/`, ALL PASS) and the Unity runtime (which uses Mel +
  Decode + tokenizer from it and swaps the encoder for the GPU path).
- `Assets/Resources/ComputeShaders/ParakeetCS.compute` — Conv2dSub (full+dw), Pointwise2d,
  FlattenSub, LinearBias, LayerNormT, AddScaled, GLU, DepthwiseConvBnSilu (folded BN),
  RelPosAttention (head_dim-128 port of the validated Chatterbox kernel, rel-shift folded into
  direct P indexing — convention PROVEN equivalent by the harness layer-0/enc_out grades).

## §11 Parity plan (`validation/dump_reference.py`)

HF `ParakeetForTDT` chain (transformers ≥ TDT support, WSL), fp32, per clip and per variant dumps:
`mel.npy` (post-norm), `pos_emb.npy`, `sub_out.npy` (post-linear), `enc_layer0.npy`, `enc_out.npy`,
`enc_proj.npy`, `joint_logits_first8.npy` ([8, V+5] — the first 8 greedy-loop evaluations),
`tokens.npy` + `durations.npy` + `frames.npy` (full emission sequence incl. blanks),
`transcript.txt`, `meta.json` (T_mel, T_enc, timings, blank-row-max check §5). Unity probes then
compare stage-by-stage, corr > 0.99 (Chatterbox precedent). Watchouts baked into dumps: symmetric
hann, constant STFT pad, channel-major subsampling flatten, interleaved sin/cos rel-pos, rel-shift
slice, BN folding, blank-row-zero embedding, LSTM gate order.

## §12 Risks

1. **Rel-shift convention mismatch** vs the Chatterbox S3Gen kernel (ESPnet legacy vs NeMo slicing)
   — settled by `pos_emb.npy`+layer-0 dumps before any kernel work; worst case is a new 30-line shift.
2. **BatchNorm folding** — running stats must come from the checkpoint (`running_mean/var`), eval
   mode only; a train-mode slip poisons every layer (dump `enc_layer0` catches it immediately).
3. **v2 conversion trust** — official HF script, but v2's NeMo config may flip a flag (use_bias,
   durations). Mitigation: the exporter asserts config equality with v3 modulo vocab, and the
   dumps for v2 come from the SAME converted checkpoint we export from.
4. **SentencePiece byte-fallback** (v3 multilingual) — decode-only handling is ~15 lines if
   present; specials.tsv records the flag. Romanian diacritics are the test case (clip planned).
5. **LSTM on CPU** — precedent says fine (2×640 is tiny); worst case ~3 ms/step scalar still meets
   budget. fp32 weights from fp16 file — no precision risk at this size.
6. **Symmetric-hann / win-pad STFT deltas** — parametrize the existing shader rather than fork it;
   mel dump is stage 1 of parity, cheap to iterate.
7. **Activation magnitudes** — the subsampling-linear output reaches ±5.4e3 (pre-LN, measured in
   the E0 dumps) and would saturate fp16 storage in later layers' intermediates if activations were
   ever halved; DeepUnity's fp32-activations convention already covers this — do not "optimize"
   encoder activations to fp16.
8. Encoder is dense 604 M matmuls — the one real GPU cost. If 4060 latency disappoints, int8 the
   encoder mats (validated-neutral scheme repo-wide) and/or fp16 activations later; no exotic ops
   anywhere (SiLU/GLU/LayerNorm/BN-affine/depthwise convs all exist or are trivial).

## §13 Recommendation

Bring up **v3 first** (HF-native checkpoint = zero conversion risk; multilingual incl. Romanian;
same code path then absorbs v2 by swapping vocab/head exports). Ship v3 fp16 as default
(~1.26 GB), v2 as the English-quality option, int8 encoder as the co-residency headroom mode.
