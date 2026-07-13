# CosyVoice3 (Fun-CosyVoice3-0.5B-2512) — DeepUnity port spec

Frozen from: released `cosyvoice3.yaml`, `CosyVoice-BlankEN/config.json`, and FunAudioLLM/CosyVoice
sources (`llm/llm.py`, `flow/flow.py`, `flow/flow_matching.py`, `flow/DiT/{dit,modules}.py`,
`transformer/upsample_encoder.py::PreLookaheadLayer`). Items marked **[VERIFY]** are confirmed
against checkpoint tensor shapes / cloned-repo code during A0-A1 before use.
Constants live in `CosyVoiceConfig.cs`; this file explains dataflow and formulas.

## §0 Pipeline

```
text ──Qwen BPE──▶ §1 LM (AR, 25Hz FSQ tokens) ──▶ §2 Flow (DiT CFM, tokens→mel 80@50Hz)
                                                      ──▶ §3 CausalHiFT (mel→24kHz wav)
```
Baked voice (zero-shot conditioning, precomputed in Python by make_voice.py):
- `prompt_text_tokens` — transcript of reference audio, Qwen BPE (must contain <|endofprompt|> per §4)
- `prompt_speech_tokens` — FSQ tokens of reference audio (speech_tokenizer_v3.onnx, 25Hz)
- `prompt_feat` — mel 80@50Hz of reference audio (n_fft 1920 / hop 480 / win 1920, center=False)
- `embedding` — campplus.onnx x-vector [192], consumed L2-normalized by the flow

## §1 LM — CosyVoice3LM (Qwen2.5-0.5B backbone)

Backbone (`Qwen2ForCausalLM`, hidden states only — Qwen's own lm_head is unused):
24 layers, hidden 896, GQA 14Q/2KV, head_dim 64, MLP 4864 SiLU (gate/up/down), RMSNorm eps 1e-6,
FULL RoPE θ=1e6, text embed_tokens vocab 151936. **[VERIFY]** q/k/v projections carry bias (Qwen2
convention), o_proj does not.

Speech side:
- `speech_embedding`: Embedding(6761, 896) — rows 0..6560 = FSQ tokens, 6561=sos, 6562=eos(stop),
  6563=task_id, 6564=fill, rest unused-ish (200-row tail; ALL ids ≥ 6561 are stop ids).
- `llm_decoder`: Linear(896 → 6761, **no bias**) — the untied speech head over the final hidden.

Inference sequence (offline / non-bistream — what we port):
```
emb = [ speech_emb[sos] | embed_tokens(prompt_text ++ text) | speech_emb[task_id] | speech_emb[prompt_speech_tokens] ]
```
then AR decode: hidden → llm_decoder → log_softmax → ras_sampling; sampled token re-embedded via
`speech_embedding` (NOT text embeddings). Stop when id ≥ 6561 (suppress EOS while
produced < 2×textLen; hard cap 20×textLen). Text must contain <|endofprompt|> (id 151646) —
the frontend appends it to prompt_text (§4).

RAS sampling (per step, over full 6761 logits):
1. nucleus: keep smallest prefix of sorted probs with cumsum < top_p(0.8) capped at top_k(25); sample.
2. if that id appeared ≥ 1 time in the last 10 emitted tokens → resample via plain multinomial
   over softmax with the candidate masked to -inf **[VERIFY exact random_sampling domain]**.

KV prefix reuse: everything up to and including task_id + prompt_speech_tokens is text-independent
per voice EXCEPT prompt_text precedes text… NOTE: unlike Chatterbox, the *text* sits BETWEEN sos
and task_id, so the reusable KV prefix is only `[sos | embed(prompt_text)]` — per-voice save/restore
covers sos+prompt_text; the utterance text + task_id + prompt tokens re-prefill each call.
(prompt_speech_tokens FOLLOW the full text, so they can't be cached across utterances.)

## §2 Flow — CausalMaskedDiffWithDiT

Per synthesis segment (offline = whole utterance; streaming = growing prefix, chunked):
```
tok = prompt_speech_tokens ++ generated_tokens            # ids < 6561
e   = Embedding(6561, 80)[tok]                            # input_embedding (masked by len)
h   = PreLookaheadLayer(e)                                # k4 rightpad3 leaky → k3 leftpad2 → +residual
      # streaming non-final chunks: last 3 tokens are passed as `context`, not part of output rows
h   = repeat_interleave(h, 2, dim=time)                   # 25Hz → 50Hz mel frames
spk = Linear192→80( L2normalize(embedding) )
conds[0:promptMel] = prompt_feat; conds[promptMel:] = 0   # [T_mel, 80]
mel = CFM(mu=h, spks=spk, cond=conds, prompt_len=promptMelLen)[ promptMelLen: ]
```

CFM (Euler, n_timesteps=10, cosine schedule t=1−cos(t·π/2)):
- x₀ = FIXED noise: torch.randn(1,80,15000) under seed 0, sliced to T — exported as tensor
  `flow/rand_noise` → bit-reproducible parity with no injection hooks.
- Per step, CFG batch-2 estimator: row0=(x,mu,spk,cond), row1=(x,0,0,0);
  dxdt = 1.7·row0 − 0.7·row1; x += dt·dxdt. (inference_cfg_rate 0.7.)
- Streaming continuity: z/mu cache pins [prompt frames + last 34 frames] of the previous chunk so
  the re-solved ODE agrees on the overlap **[VERIFY exact 2512 cache mechanics in cloned repo —
  GitHub main's ConditionalCFM.forward carries (z,mu) cache; the yaml's CausalConditionalCFM
  variant uses the fixed-noise buffer; reconcile which path CosyVoice3Model drives]**.

DiT estimator (per forward, batch 2):
- input_embed: cat[x, cond, mu, spk_broadcast] (4×80=320) → Linear(320→1024) → x +=
  CausalConvPosEmbed(x): 2× [leftpad30 conv1d(k31, groups16) + Mish].
- t-embed: sinus256(scale 1000: emb=exp(-log(10000)·i/127)·1000·t, [sin|cos]) → Linear256→1024 → SiLU → Linear1024→1024.
- 22 × DiTBlock(dim 1024, 16h×64d, ff 2048):
  AdaLN-Zero: (shift,scale,gate)×2 from Linear(SiLU(t))→6144;
  attn: LN(no-affine, eps 1e-6)·(1+scale)+shift → MHA (q/k/v/out Linear WITH bias **[VERIFY]**,
  RoPE x_transformers over all 64 dims θ=10000 **[VERIFY]**) → x += gate_msa·attn;
  ff: LN·(1+scale_mlp)+shift_mlp → Linear1024→2048 → GELU(tanh) → Linear2048→1024 → x += gate_mlp·ff.
- final: AdaLN-Final (scale,shift from Linear(SiLU(t))→2048) → proj_out Linear(1024→80).
- Attention mask: offline → full bidirectional over T; streaming → chunk mask: frame i attends to
  frames < (⌊i/50⌋+1)·50 (50-frame chunks, ALL left context, no right).

## §3 Vocoder — CausalHiFTGenerator (RESOLVED from sources + hift.pt, 2026-07-11)

Same NSF+iSTFT skeleton as the validated Chatterbox HiFT port; geometry identical
(upsample [8,5,3] k[16,11,7], resblocks k[3,7,11] d[1,3,5] Snake, source resblocks k[7,7,11],
base 512, iSTFT n_fft 16 hop 4, nsf α .1 σ .003 vthr 10, audio_limit .99, lrelu .1).
Checkpoint: 328 tensors, NEW-style weight norm (`parametrizations.weight.original0/1` — exporter
folds g·v/||v||). Causal deltas vs Chatterbox:

- **CausalConv1d** (stride1): pad = int((k·d−d)/2)·2 + (k+1)%2, ALL on one side.
  `causal_type='left'` → past-pad; `'right'` → FUTURE-pad (lookahead). Streaming non-finalize:
  the pad slots are filled from explicit cache/context tensors instead of zeros.
- **conv_pre**: CausalConv1d(80→512, k5, RIGHT) → 4 mel frames lookahead (`conv_pre_look_right=4`).
  Non-finalize chunks run conv_pre on x[:-4] with x[-4:] as the context, and trim the source
  spectrum by 120·4 samples' frames correspondingly.
- **ups**: CausalConv1dUpsample = nearest-neighbor ×stride THEN CausalConv1d(k, LEFT-pad k−1).
  Checkpoint layout [out,in,k] plain Conv1d — NOT ConvTranspose. Port = RepeatTime + Conv1D
  kernels (both already exist).
- **source_downs**: i0 = CausalConv1dDownSample(18→256, k30, s15) [left-pad s−1=14],
  i1 = (18→128, k6, s3) [pad2], i2 = CausalConv1d(18→64, k1) [pad0]. (18 = n_fft/2+1 = 9 complex.)
- **ResBlock(causal=True)** with Snake α per channel; conv_post = CausalConv1d(64→18, k7, LEFT).
- **reflection_pad (1,0)** on x before the LAST stage's fusion (same as chatterbox).
- **Head**: mag = exp(x[:9]); **phase = sin(x[9:])** (direct, no atan2); iSTFT(mag, phase);
  clamp ±0.99. Real = mag·cos(phase), imag = mag·sin(phase).
- **NSF source = SineGen2** (sr 24000 ≠ 22050 → type '2'), causal=True, cumsum-phase generation
  (CumsumPhase/SineMerge kernel family already ported); harmonics 8+1 → m_source.l_linear(9→1)+tanh.
- **f0_predictor = CausalConvRNNF0Predictor — NO RNN despite the name**: 5 causal convs
  [k4 RIGHT(lookahead 3) → 4× k3 LEFT] each +ELU, then Linear(512→1) + abs. Reference runs it in
  float64 ("precision crucial for causal inference") — port plan: fp32 GPU kernel first, parity
  gate vs dump; if F0 drifts, CPU double-precision C# fallback (tiny: ~5 convs over T frames).
- **Chunked/streaming decode** (finalize=False): trim lookahead tails (conv_pre 4 mel frames;
  istft hop·prod(ups)=480·4 samples) and carry per-conv caches. Reference test rebuilds from the
  full prefix each chunk (O(T²)); the real per-conv-cache streaming plumbing (hift cache in
  cli/model.py token2wav) is finalized at A5.

## §4 Text frontend & tokenizer

- Tokenizer: Qwen2.5 BPE (CosyVoice-BlankEN vocab.json + merges.txt, vocab 151936) + specials;
  extends DeepUnity Base/BPETokenizer. allowed_special: all. <|endofprompt|> = 151646.
- Zero-shot prompt assembly **[VERIFY exact frontend.py format at A0]**: prompt_text transcript
  followed by <|endofprompt|>, then the utterance text; instruct mode inserts instruction text
  before <|endofprompt|> instead.
- EN-min normalization v1 (mirror PuncNorm approach): whitespace collapse, unusual-punct
  replacement, capitalize, trailing period; number spell-out optional later.

## §5 Streaming model (A5)

Reference (cli/model.py **[VERIFY hop schedule]**): LM emits continuously; every CHUNK_TOKENS=25
new tokens (+3 lookahead tokens held back) → flow chunk (streaming mask, z/mu overlap cache,
prompt always pinned) → mel chunk → chunked HiFT (mel overlap 8, source cache) → PCM → ring buffer.
First audio ≈ after 28 tokens ≈ 1.1 s of LM decode + 1 chunk of flow+voc — the LM decode rate
(≥25 tok/s needed) and per-chunk DiT cost are the two real-time budgets on the 4060.

## §6 Weights inventory (manifest.tsv, ChatterboxWeights format)

From llm.rl.pt: qwen backbone (24 layers: q/k/v(+bias)/o, gate/up/down, norms, embed_tokens) +
speech_embedding [6761,896] + llm_decoder [6761,896].
From flow.pt: input_embedding [6561,80], spk_embed_affine [80,192](+b), pre_lookahead conv1/conv2,
DiT: input_proj [1024,320](+b), convpos (2×[1024,1024/16,31]+b), time MLP, 22 blocks ×
{adaLN Linear[6144,1024](+b), q/k/v/out [1024,1024](+b), ff1 [2048,1024](+b), ff2 [1024,2048](+b)},
final adaLN [2048,1024](+b), proj_out [80,1024](+b), rand_noise [1,80,15000] (from seed-0 dump).
From hift.pt: full CausalHiFT tree (names at A1).
Voices: `voices/<name>/{prompt_text_tokens(i32), prompt_speech_tokens(i32), prompt_feat(f16 [T,80]),
embedding(f16 [192])}`.
Quant: fp16 everywhere; int8 = LM matmuls (+DiT matmuls if needed) with per-row .scales, norms/
embeddings/heads stay fp16 (repo convention).

## §7 Parity plan

Stage dumps (dump_reference.py, greedy LM sampling top_k=1 for determinism):
text_tokens, lm_logits_step0, speech_tokens, flow: {h_after_lookahead, mu_50hz, dit_in_step0,
dxdt_step0/cond+uncond, mel}, hift: {f0, source, wav}. Expected corr >0.99 per stage (chatterbox
precedent: mu 1.000, wav 0.999). The fixed noise buffer removes all stochasticity from the flow;
NSF noise/phases still need dump-injection like Chatterbox (nsf σ noise + phase init).
